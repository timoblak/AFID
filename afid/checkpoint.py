"""
Checkpoint format {"model": state_dict, "metadata": {...}} and backbone check.

The backbone is frozen and shared by all components, so only afid_encoder.pth stores it. The other
files hold their heads only. Every file records the backbone_id (a hash of the backbone weights) of
the backbone it belongs to, and the loader refuses files from different releases.
"""
import hashlib
import torch

BACKBONE_PREFIX = "encoder."


def backbone_id(state_dict, prefix=BACKBONE_PREFIX):
    h = hashlib.sha256()
    for k in sorted(k for k in state_dict if k.startswith(prefix)):
        t = state_dict[k].detach().cpu().contiguous().float()
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()[:12]


def load_checkpoint(path, map_location="cpu"):
    """Load and validate a checkpoint (weights_only, so no arbitrary unpickling)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    metadata = ckpt.get("metadata")
    if metadata is None:
        raise RuntimeError(f"{path}: missing 'metadata' block, unknown version.")
    return ckpt["model"], metadata


def check_backbone(metadata, path, actual_id):
    if metadata.get("backbone_id") != actual_id:
        raise RuntimeError(
            f"{path}: was trained on backbone {metadata.get('backbone_id')} but the loaded "
            f"afid_encoder.pth has backbone {actual_id}. Files from different releases "
            f"cannot be mixed.")
