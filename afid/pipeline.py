"""
AFID pipeline: encoder, AFID-X and AFID-Q on a shared backbone pass.

    afid = load_afid()
    embedding = afid.embed("fingermark.png")           # [768]
    features = afid.extract_features("fingermark.png") # minutiae, segmentation, orientation
    quality = afid.assess_quality("fingermark.png")    # q_n, q_e, expert_scores, fusion, afid_q

Images are file paths or numpy arrays (grayscale or BGR as OpenCV delivers it, pixel values 0-255;
floats scaled to 0-1 are not rescaled). Every method also takes a list of images and then returns a
list of results, processed batch_size images at a time. Results are plain dicts.

load_afid(afid_x=False, afid_q=False) loads the identity encoder only.
"""
from pathlib import Path

import cv2
import numpy as np
import torch

from . import checkpoint, minutiae_decode, preprocess
from .calibration import calibrate, fuse
from .models import AFIDEncoder, AFIDQ, AFIDX, EMBEDDING_DIM

# Folder with one subfolder per weights release, named <release>_<backbone id>
# (e.g. afid_768_noncommercial_a19b77afeecd/).
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

# Minutiae operating point: detection threshold, NMS window (model px).
MINUTIAE_PROB_THRESHOLD = 0.35
MINUTIAE_PEAK_DISTANCE = 7

# Gaussian sigmas in model px (tuned for the 768 px input): segmentation is blurred before
# resizing (10 px ~ 0.5 mm at 500 ppi), orientation after, scaled to image size.
SEG_SMOOTH_SIGMA = 10.0
ORI_SMOOTH_SIGMA = 5.0


def find_release(checkpoint_dir=None, release=None):
    """The weights folder to use: the release called `release` inside `checkpoint_dir`
    (default: checkpoints/), or the only release there if `release` is not given."""
    root = Path(checkpoint_dir) if checkpoint_dir is not None else CHECKPOINT_DIR
    if (root / "afid_encoder.pth").exists():
        raise FileNotFoundError(f"{root} is a weights release itself; pass the folder that contains it "
                                f"as checkpoint_dir and its name as release")
    folders = [p for p in sorted(root.glob("*")) if (p / "afid_encoder.pth").exists()]
    names = [p.name.rsplit("_", 1)[0] for p in folders]  # folder name without the backbone id
    if release is None:
        if len(folders) == 1:
            return folders[0]
        if not folders:
            raise FileNotFoundError(f"no weights found in {root}; unzip a weights release into it")
        raise FileNotFoundError(f"several weights releases in {root}, pick one with release=: " + ", ".join(names))
    matches = [p for p, name in zip(folders, names) if release in (name, p.name)]
    if len(matches) != 1:
        raise FileNotFoundError(f"release '{release}' not found in {root}; available: " + (", ".join(names) or "none"))
    return matches[0]


class AFID:
    """Identity encoder plus optional feature extraction (AFID-X) and quality
    assessment (AFID-Q). Components switched off are neither built nor loaded."""

    def __init__(self, device="cpu", afid_x=True, afid_q=True):
        self.device = torch.device(device)
        self.img_size = None  # model input size, read from the encoder checkpoint
        self.backbone_id = None  # set by load_checkpoints
        self.release = None
        self.afid_encoder = AFIDEncoder().to(self.device).eval()
        self.afid_x = AFIDX().to(self.device).eval() if afid_x else None
        self.afid_q = AFIDQ().to(self.device).eval() if afid_q else None

    def load_checkpoints(self, checkpoint_dir=None, release=None):
        checkpoint_dir = find_release(checkpoint_dir, release)
        path = checkpoint_dir / "afid_encoder.pth"
        sd, metadata = checkpoint.load_checkpoint(path, self.device)
        bid = checkpoint.backbone_id(sd)
        checkpoint.check_backbone(metadata, path, bid)  # file integrity
        self.afid_encoder.load_state_dict(sd)
        self.backbone_id = bid
        if "img_size" not in metadata:
            raise RuntimeError(f"{path}: metadata has no 'img_size'; re-assemble the release")
        self.img_size = metadata["img_size"]
        self.release = metadata.get("release")

        # The other files hold heads only; the backbone is the encoder's.
        for module, name in ((self.afid_x, "afid_x_decoders.pth"),
                             (self.afid_q, "afid_q_regressor.pth")):
            if module is None:
                continue
            path = checkpoint_dir / name
            sd, metadata = checkpoint.load_checkpoint(path, self.device)
            checkpoint.check_backbone(metadata, path, bid)
            module.load_state_dict(sd)
        return self

    # ------------------------------------------------------------ public API

    def embed(self, images, batch_size=1):
        """Raw identity embedding: float32 [768], or [N, 768] for a list. Its L2
        norm is Q_n; normalise it before computing cosine similarity."""
        results = self._process(images, batch_size, embed=True)
        if isinstance(results, dict):
            return results["embedding"]
        return np.stack([r["embedding"] for r in results]) if results else np.empty((0, EMBEDDING_DIM), np.float32)

    def extract_features(self, images, batch_size=1, minutiae_prob_threshold=MINUTIAE_PROB_THRESHOLD,
                         minutiae_peak_distance=MINUTIAE_PEAK_DISTANCE):
        """Needs AFID-X. Returns a dict with
            minutiae      list of {x, y, theta, quality}, image px, theta in degrees
            segmentation  [H, W] float32 foreground probability, 0-1
            orientation   [H, W] float32 ridge orientation in degrees, 0-180
        segmentation and orientation are at the image's own resolution. Orientation has period 180;
        to average or interpolate it, use the doubled-angle field (cos 2t, sin 2t).
        minutiae_prob_threshold and minutiae_peak_distance set the minutiae detection threshold
        and the minimum distance between minutiae (in model pixels)."""
        self._require(self.afid_x, "afid_x")
        return self._process(images, batch_size, extract=True, minutiae_prob_threshold=minutiae_prob_threshold,
                             minutiae_peak_distance=minutiae_peak_distance)

    def assess_quality(self, images, batch_size=1):
        """Needs AFID-Q. Returns a dict with
            q_n            embedding norm (MagFace quality)
            q_e            mean of the expert scores, 0-1
            expert_scores  [10] float32, 0-1
            fusion         max(q_n - 10, 0) * q_e
            afid_q         final quality, integer 0-100 (ISO/IEC 29794-1)"""
        self._require(self.afid_q, "afid_q")
        return self._process(images, batch_size, quality=True)

    def run(self, images, batch_size=1, minutiae_prob_threshold=MINUTIAE_PROB_THRESHOLD,
            minutiae_peak_distance=MINUTIAE_PEAK_DISTANCE):
        """Everything the loaded components compute, from one backbone pass:
        'embedding' plus the keys of extract_features and assess_quality."""
        return self._process(images, batch_size, embed=True, extract=self.afid_x is not None,
                             quality=self.afid_q is not None, minutiae_prob_threshold=minutiae_prob_threshold,
                             minutiae_peak_distance=minutiae_peak_distance)

    @torch.no_grad()
    def encode(self, image):
        """The four backbone stage maps ([1, C, H, W] tensors), e.g. as input
        for your own head."""
        tensor, _ = preprocess.preprocess(preprocess.load_image(image), self.img_size)
        return self.afid_encoder.encoder(tensor.unsqueeze(0).to(self.device))

    # -------------------------------------------------------------- internals

    @staticmethod
    def _require(module, name):
        if module is None:
            raise RuntimeError(f"loaded without {name}; use load_afid({name}=True)")

    @torch.no_grad()
    def _process(self, images, batch_size, embed=False, extract=False, quality=False,
                 minutiae_prob_threshold=MINUTIAE_PROB_THRESHOLD, minutiae_peak_distance=MINUTIAE_PEAK_DISTANCE):
        """Run the requested components, batch by batch. One result dict per image;
        a single image (not a list) gives a single dict."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        single = not isinstance(images, (list, tuple))
        if single:
            images = [images]
        results = []
        for i in range(0, len(images), batch_size):

            # Load images
            imgs = [preprocess.load_image(im) for im in images[i:i + batch_size]]

            # Preprocess images
            tensors, pads = zip(*[preprocess.preprocess(img, self.img_size) for img in imgs])

            # Backbone pass
            stages = self.afid_encoder.encoder(torch.stack(tensors).to(self.device))
            batch = [{} for _ in imgs]
            raw = self.afid_encoder.embed(stages)  # [B, 768]

            if embed:
                for out, e in zip(batch, raw.float().cpu().numpy()):
                    out["embedding"] = e
            if extract:
                ori, minu, seg = self.afid_x(stages)
                ori = ori.float().cpu().numpy()
                minu = torch.sigmoid(minu).float().cpu().numpy()
                seg = torch.sigmoid(seg[:, 0]).float().cpu().numpy()
                for b, out in enumerate(batch):
                    out.update(self._decode_features(ori[b], minu[b], seg[b], pads[b], imgs[b].shape,
                                                     minutiae_prob_threshold, minutiae_peak_distance))
            if quality:
                scores = self.afid_q(stages)  # [B, 10]
                for b, out in enumerate(batch):
                    out.update(self._quality(raw[b], scores[b]))
            results += batch
        return results[0] if single else results

    def _decode_features(self, ori, min_prob, seg, pad, size, minutiae_prob_threshold, minutiae_peak_distance):
        """Raw decoder outputs of one image (model grid) -> features at the image's own resolution."""
        H, W = size
        seg = cv2.GaussianBlur(seg, (0, 0), SEG_SMOOTH_SIGMA)
        segmentation = preprocess.unpad_and_resize(seg, pad, H, W)

        # Orientation = doubled-angle field (ux, uy): blur the components, not the wrapping angle.
        ux, uy = preprocess.unpad_and_resize(ori, pad, H, W)
        sigma = max(1, int(round(max(H, W) / self.img_size * ORI_SMOOTH_SIGMA)))  # model px -> image px
        ux = cv2.GaussianBlur(ux, (0, 0), sigma)
        uy = cv2.GaussianBlur(uy, (0, 0), sigma)
        mag = np.sqrt(ux ** 2 + uy ** 2) + 1e-8
        orientation = ((np.degrees(0.5 * np.arctan2(uy / mag, ux / mag)) + 180) % 180).astype(np.float32)

        peaks = minutiae_decode.decode_minutiae(min_prob, minutiae_prob_threshold, minutiae_peak_distance)
        minutiae = minutiae_decode.to_orig_coords(peaks, pad, self.img_size, H, W)
        return {"minutiae": minutiae, "segmentation": segmentation, "orientation": orientation}

    @staticmethod
    def _quality(raw_embedding, scores):
        q_n = torch.norm(raw_embedding, p=2).item()
        q_e = scores.mean().item()
        fusion = fuse(q_n, q_e)
        return {"q_n": q_n, "q_e": q_e, "expert_scores": scores.float().cpu().numpy(),
                "fusion": fusion, "afid_q": calibrate(fusion)}


def load_afid(device=None, checkpoint_dir=None, release=None, afid_x=True, afid_q=True) -> AFID:
    """checkpoint_dir: the folder that contains the weights releases (default: checkpoints/).
    release: name of the release to load, e.g. "afid_768_noncommercial"; optional if there is only one.
    afid_x=False / afid_q=False skip loading that component. The encoder is always loaded."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return AFID(device=device, afid_x=afid_x, afid_q=afid_q).load_checkpoints(checkpoint_dir, release)
