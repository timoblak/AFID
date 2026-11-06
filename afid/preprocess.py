"""
Preprocessing shared by all components: fit to a square of the model's input size
(img_size, from the checkpoint metadata) keeping aspect ratio, linear resize, black padding.

Note: AFID-Q was trained with white padding and area resize; the effect of the
shared convention is small (<= 1 point of AFID-Q) and goes away at retraining.
"""
from pathlib import Path

import cv2
import numpy as np
import torch

# Means and std of the training set
MEAN, STD = 0.709, 0.265


def _resize_pad(img_u8, img_size, pad_value_u8=0):
    """Returns (padded image, (top, bot, left, right))."""
    H, W = img_u8.shape
    scale = img_size / max(H, W)
    # Truncate, not round: matches training.
    new_H, new_W = max(1, int(H * scale)), max(1, int(W * scale))
    img_r = cv2.resize(img_u8, (new_W, new_H), interpolation=cv2.INTER_LINEAR)

    pad_H, pad_W = img_size - new_H, img_size - new_W
    top, left = pad_H // 2, pad_W // 2
    bot, right = pad_H - top, pad_W - left
    img_p = np.pad(img_r, ((top, bot), (left, right)), mode="constant", constant_values=pad_value_u8)
    return img_p, (top, bot, left, right)


def _to_tensor(img_p_u8):
    norm = (img_p_u8.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(norm).float().unsqueeze(0).repeat(3, 1, 1)


def preprocess(img_u8, img_size):
    """Returns (normalized [3,H,W] tensor, padding (top, bot, left, right))."""
    img_p, pad = _resize_pad(img_u8, img_size)
    return _to_tensor(img_p), pad


def load_image(image):
    """Path or numpy image -> grayscale uint8."""
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"could not read {image}")
        return img
    return to_gray_u8(image)


def to_gray_u8(image):
    """Numpy image -> grayscale uint8"""
    img = np.asarray(image)
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 3:
        code = {3: cv2.COLOR_BGR2GRAY, 4: cv2.COLOR_BGRA2GRAY}.get(img.shape[2])
        if code is None:
            raise ValueError(f"expected 1, 3 or 4 channels, got shape {img.shape}")
        img = cv2.cvtColor(img, code)
    if img.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale or 3-channel image, got shape {img.shape}")
    return img


def unpad_and_resize(arr, pad, orig_h, orig_w):
    """Crop the padding of a [H,W] or [C,H,W] model-grid map and resize to the original size."""
    pad_top, pad_bot, pad_left, pad_right = pad
    h, w = arr.shape[-2], arr.shape[-1]
    if arr.ndim == 3:
        cropped = np.transpose(arr[:, pad_top:h - pad_bot, pad_left:w - pad_right], (1, 2, 0))
        resized = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        if resized.ndim == 2:
            resized = resized[:, :, None]
        return np.transpose(resized, (2, 0, 1))
    cropped = arr[pad_top:h - pad_bot, pad_left:w - pad_right]
    return cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
