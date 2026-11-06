#!/usr/bin/env python3
"""
AFID demo: fingermark matching with the identity embedding.

    python demo_match.py a.png b.png              # 1:1  one similarity score
    python demo_match.py probe.png gallery/       # 1:n  best matches of one image in a folder
    python demo_match.py folder/                  # n:n  every image against every other image
    python demo_match.py probes/ gallery/         # m:n  every probe against every gallery image
    python demo_match.py pairs.txt                # list of pairs, one score per pair

Each input is an image or a folder. With one input, its images are compared
with each other; with two, the first are the probes and the second the gallery.
A single input that is not an image is a pair list: one pair per line, two full
image paths separated by a comma, a tab or whitespace (use a tab if a path
contains spaces). Empty lines and lines starting with # are ignored.

Step 1, enrollment: every image becomes a 768-d embedding (computed in batches).
Step 2, matching: the similarity of two images is the cosine similarity of their
embeddings (afid.similarity), -1 to 1; higher means more likely the same finger.
For feature extraction and quality see demo_extract.py.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

from afid import load_afid, similarity

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(path, recursive):
    """The image itself, or the images in a folder (sorted; with subfolders if recursive)."""
    if not path.exists():
        sys.exit(f"error: {path} not found")
    if path.is_file():
        return [path]
    files = path.rglob("*") if recursive else path.glob("*")
    return sorted(p for p in files if p.suffix.lower() in IMAGE_EXTS)


def read_pairs(path):
    """Read a pair list: one line per pair, giving the two image paths. Returns [(path_a, path_b), ...]."""
    pairs = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):  # empty line or comment
            continue
        # Comma or tab if there is one (paths may contain spaces), otherwise whitespace.
        if "," in line or "\t" in line:
            fields = [field.strip() for field in re.split("[,\t]", line)]
        else:
            fields = line.split()
        if len(fields) < 2:
            sys.exit(f"error: {path}, line {line_number}: expected two image paths")
        pairs.append((Path(fields[0]), Path(fields[1])))  # further columns are ignored

    # Check all files now, so a typo does not surface after minutes of embedding.
    for a, b in pairs:
        for image in (a, b):
            if not image.is_file():
                sys.exit(f"error: image not found: {image}")
    return pairs


def write_csv(path, rows):
    """rows: (probe, gallery, similarity) tuples."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["probe", "gallery", "similarity"])
        for probe, gallery, score in rows:
            writer.writerow([probe, gallery, f"{score:.6f}"])
    print(f"Saved {path}")


def match_pairs(afid, pairs, args):
    """Similarity of every listed pair."""
    if not pairs:
        sys.exit("error: no pairs in the list")

    # Enrollment: an image that appears in several pairs is embedded only once.
    images = set()
    for a, b in pairs:
        images.update([a, b])
    images = sorted(images)
    print(f"Enrolling {len(images)} images...")
    embedding_of = dict(zip(images, afid.embed(images, args.batch_size)))

    # Matching: cosine similarity of the two embeddings of each pair.
    rows = []
    for a, b in pairs:
        score = similarity(embedding_of[a], embedding_of[b])
        print(f"{a.name}  {b.name}  {score:.4f}")
        rows.append((a, b, score))
    if args.csv:
        write_csv(args.csv, rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", type=Path, nargs="+", metavar="input", help="one or two images or folders, or a pair list")
    ap.add_argument("--top", type=int, default=5, help="matches to show per probe (default: 5)")
    ap.add_argument("--csv", type=Path, help="write all scores to this CSV")
    ap.add_argument("--batch_size", type=int, default=16, help="images per batch (default: 16)")
    ap.add_argument("--recursive", "-r", action="store_true", help="also search subfolders of a folder")
    ap.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--checkpoints", type=Path, help="folder with the weights releases (default: checkpoints/)")
    ap.add_argument("--release", help="name of the weights release (default: the only one in the folder)")
    args = ap.parse_args()
    if len(args.inputs) > 2:
        ap.error("give one or two inputs")

    afid = load_afid(device=args.device, checkpoint_dir=args.checkpoints, release=args.release,
                     afid_x=False, afid_q=False)  # embeddings only
    print(f"Weights: {afid.release} ({afid.backbone_id})")

    # A single input that is not an image is a pair list.
    if len(args.inputs) == 1 and args.inputs[0].is_file() and args.inputs[0].suffix.lower() not in IMAGE_EXTS:
        match_pairs(afid, read_pairs(args.inputs[0]), args)
        return

    # Probes and gallery. With one input the images are matched with each other (n:n),
    # so the probes are also the gallery.
    n_to_n = len(args.inputs) == 1
    probe_paths = list_images(args.inputs[0], args.recursive)
    gallery_paths = probe_paths if n_to_n else list_images(args.inputs[1], args.recursive)
    if not probe_paths or not gallery_paths or (n_to_n and len(probe_paths) < 2):
        sys.exit("error: not enough images to match")

    # Enrollment: one embedding per image.
    n_images = len(probe_paths) if n_to_n else len(probe_paths) + len(gallery_paths)
    print(f"Enrolling {n_images} images...")
    probes = afid.embed(probe_paths, args.batch_size)
    gallery = probes if n_to_n else afid.embed(gallery_paths, args.batch_size)

    # Matching: scores[i, j] = similarity of probe i and gallery image j.
    scores = similarity(probes, gallery)
    if n_to_n:
        np.fill_diagonal(scores, -np.inf)  # an image is not matched with itself

    # Print: one score for 1:1, otherwise the best matches of each probe.
    if len(probe_paths) == 1 and len(gallery_paths) == 1:
        print(f"Similarity: {scores[0, 0]:.4f}")
    else:
        for i, probe in enumerate(probe_paths):
            print(probe.name)
            best_first = np.argsort(-scores[i])
            for rank, j in enumerate(best_first[:args.top], 1):
                print(f"  {rank}. {gallery_paths[j].name}  {scores[i, j]:.4f}")

    if args.csv:
        rows = []
        for i, probe in enumerate(probe_paths):
            for j, gal in enumerate(gallery_paths):
                if n_to_n and j <= i:  # n:n: write each pair once
                    continue
                rows.append((probe, gal, scores[i, j]))
        write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
