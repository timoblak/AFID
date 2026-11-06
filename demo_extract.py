#!/usr/bin/env python3
"""
AFID demo: feature extraction, quality assessment and visualization.

    python demo_extract.py image.png                        # features + results.txt -> results/<name>/
    python demo_extract.py folder/ --recursive              # every image in a folder (and its subfolders)
    python demo_extract.py image.png --mode quality         # print the quality scores, save nothing
    python demo_extract.py folder/ --mode quality --csv q.csv
    python demo_extract.py image.png --mode features        # save only the features

The features are saved per image as
    features.png      segmentation, ridge orientation and minutiae side by side
    minutiae.csv      x, y (image px), theta (degrees), quality
    mask.png          binary foreground mask (segmentation > 0.5), 0/255, image size
    orientation.npz   array "orientation": ridge orientation in degrees (0-180), image size
For fingermark matching see demo_match.py.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from afid import load_afid
from afid.preprocess import load_image
from afid.visualize import render_feature_panel

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

LEGEND = """\
Scores (higher is better):
  AFID-Q  final quality, integer 0-100 (ISO/IEC 29794-1); 50 = ~50% rank-1
  F       fusion max(Q_n - 10, 0) * Q_e; 0 = no usable ridges, >= 18.95 -> 100
  Q_e     expert-ensemble regressor, 0-1
  Q_n     embedding norm; ~5-10 = background, 10-35+ = friction ridge"""


def list_images(path, recursive):
    """The image itself, or the images in a folder (sorted; with subfolders if recursive)."""
    if path.is_file():
        return [path]
    files = path.rglob("*") if recursive else path.glob("*")
    return sorted(p for p in files if p.suffix.lower() in IMAGE_EXTS)


def describe(result):
    """One line with whatever was computed (quality scores and/or minutiae count)."""
    parts = []
    if "afid_q" in result:  # AFID-Q was loaded
        parts.append(f"AFID-Q={result['afid_q']}/100  F={result['fusion']:.4f}  "
                     f"Q_e={result['q_e']:.4f}  Q_n={result['q_n']:.4f}")
    if "minutiae" in result:  # AFID-X was loaded
        parts.append(f"minutiae={len(result['minutiae'])}")
    return "  ".join(parts)


def save_features(result, out_dir):
    """Write the extracted features of one image into out_dir."""
    # Minutiae: one row each, rounded to keep the file readable.
    with open(out_dir / "minutiae.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "theta", "quality"])
        for m in result["minutiae"]:
            writer.writerow([f"{m['x']:.1f}", f"{m['y']:.1f}", f"{m['theta']:.1f}", f"{m['quality']:.3f}"])

    # Segmentation: probability > 0.5 is foreground (255), the rest background (0).
    mask = (result["segmentation"] > 0.5).astype(np.uint8) * 255
    cv2.imwrite(str(out_dir / "mask.png"), mask)

    # Orientation: ridge angle in degrees (0-180) for every pixel.
    np.savez_compressed(out_dir / "orientation.npz", orientation=result["orientation"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="an image, or a folder of images")
    ap.add_argument("--mode", choices=["quality", "features", "all"], default="all",
                    help="quality: print scores only; features: save the features only; all: both (default)")
    ap.add_argument("--out_dir", type=Path, default=Path("results"), help="output folder (default: results)")
    ap.add_argument("--recursive", "-r", action="store_true", help="also search subfolders of a folder")
    ap.add_argument("--csv", type=Path, help="also write the scores of all images to this CSV")
    ap.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--checkpoints", type=Path, help="folder with the weights releases (default: checkpoints/)")
    ap.add_argument("--release", help="name of the weights release (default: the only one in the folder)")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"error: {args.input} not found")
    paths = list_images(args.input, args.recursive)
    if not paths:
        sys.exit(f"error: no images found in {args.input}" + ("" if args.recursive else " (try --recursive)"))

    # Load only what the mode needs: AFID-X for the features, AFID-Q for the quality.
    want_features = args.mode in ("features", "all")
    want_quality = args.mode in ("quality", "all")
    afid = load_afid(device=args.device, checkpoint_dir=args.checkpoints, release=args.release,
                     afid_x=want_features, afid_q=want_quality)
    print(f"Weights: {afid.release} ({afid.backbone_id})")
    if want_quality:
        print(LEGEND)

    rows = []  # one summary row per image, for --csv
    for path in paths:
        # Extract features and/or quality.
        try:
            image = load_image(path)
            result = afid.run(image)  # one backbone pass for everything that is loaded
        except Exception as e:
            print(f"{path.name}: FAILED ({e})", file=sys.stderr)
            continue
        print(f"{path.name}: {describe(result)}")

        # Save the results into results/<name>/. In a folder the subfolder structure is kept.
        if want_features:
            if args.input.is_dir():
                name = path.relative_to(args.input).with_suffix("")
            else:
                name = Path(path.stem)
            out_dir = args.out_dir / name
            out_dir.mkdir(parents=True, exist_ok=True)

            panel, _ = render_feature_panel(image, result)  # overview picture
            cv2.imwrite(str(out_dir / "features.png"), panel)
            save_features(result, out_dir)
            if want_quality:
                (out_dir / "results.txt").write_text(
                    f"afid_q={result['afid_q']} fusion={result['fusion']:.6f} "
                    f"q_e={result['q_e']:.6f} q_n={result['q_n']:.6f}\n")

        # Summary row: image path, the quality scores (if computed) and the minutiae count (if computed).
        row = {"image": path}
        for key in ("afid_q", "fusion", "q_e", "q_n"):
            if key in result:
                row[key] = result[key]
        if "minutiae" in result:
            row["n_minutiae"] = len(result["minutiae"])
        rows.append(row)

    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {args.csv}")


if __name__ == "__main__":
    main()
