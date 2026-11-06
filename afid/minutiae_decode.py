"""Minutiae decoding from the AFID-X minutiae head output."""
import math

import cv2
import numpy as np

from .models import MINUTIAE_CH as N_BINS  # angular bins over 360 deg


def decode_minutiae(min_prob, prob_threshold=0.35, min_peak_distance=7):
    """
    min_prob: [N_BINS,H,W] sigmoid probabilities, raw AFID-X output.

    Returns a list of dicts {x, y, theta, quality} in the same pixel grid:
    theta in degrees [0, 360), counter-clockwise on screen. Quality is the
    detection map value at the peak.
    """
    detection = min_prob.sum(axis=0)
    dilated = cv2.dilate(detection, np.ones((min_peak_distance, min_peak_distance), np.uint8))
    ys, xs = np.where((detection >= dilated - 1e-6) & (detection > prob_threshold))

    angles = np.radians(np.arange(N_BINS) * 360 / N_BINS)
    out = []
    for y, x in zip(ys, xs):
        w = min_prob[:, y, x]
        theta = math.degrees(math.atan2(np.sum(w * np.sin(angles)), np.sum(w * np.cos(angles)))) % 360
        out.append({"x": float(x), "y": float(y), "theta": theta, "quality": float(detection[y, x])})
    return out


def to_orig_coords(minutiae, pad, img_size, orig_h, orig_w):
    """Map minutiae from the padded model grid to original image pixels.
    pad = (top, bot, left, right)."""
    scale = img_size / max(orig_h, orig_w)
    return [{**m, "x": (m["x"] - pad[2]) / scale, "y": (m["y"] - pad[0]) / scale} for m in minutiae]
