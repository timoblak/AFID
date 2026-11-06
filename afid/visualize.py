"""
Feature visualization: segmentation, ridge orientation and minutiae, as one row
of square panels (render_feature_panel).
"""
import cv2
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Paul Tol "bright" colours; the cyclic map wraps because 0 and 180 deg are the
# same ridge orientation.
TOL_BRIGHT = {
    'blue':   '#4477AA',
    'cyan':   '#66CCEE',
    'green':  '#228833',
    'yellow': '#CCBB44',
    'red':    '#EE6677',
    'purple': '#AA3377',
    'grey':   '#BBBBBB',
}
TOL_CYCLIC = ['#4477AA', '#66CCEE', '#228833', '#CCBB44', '#EE6677', '#AA3377']


def tol_cyclic_cmap(n=256):
    return LinearSegmentedColormap.from_list('tol_cyclic', TOL_CYCLIC + TOL_CYCLIC[:1], N=n)


def hex_to_bgr(hex_color):
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


SEG_BGR = hex_to_bgr(TOL_BRIGHT['green'])
MINUTIAE_BGR = hex_to_bgr(TOL_BRIGHT['red'])
OUTLINE_DARK = (25, 25, 25)

PANEL_TILE = 512
PANEL_GUTTER = 8
PANEL_BG = 255

_ORI_LUT = (np.asarray([tol_cyclic_cmap(256)(i / 256.0)[:3] for i in range(256)])[:, ::-1] * 255
            ).astype(np.uint8)  # RGB -> BGR


def _glyph_scale(shape, tile=PANEL_TILE):
    """Source px per tile px -- so `n * _glyph_scale(...)` renders as n px on the tile."""
    return max(1e-3, max(shape[0], shape[1]) / float(tile))


def _glyph_px(nominal_tile_px, scale, minimum=1):
    return max(minimum, int(round(nominal_tile_px * scale)))


def overlay_segmentation(base_bgr, seg_map, alpha=0.4, color=SEG_BGR, border=None):
    """Translucent mask fill plus an opaque outline in the same colour."""
    mask = (seg_map > 0.5).astype(np.uint8)
    overlay = base_bgr.copy()
    overlay[mask == 1] = (
        (1 - alpha) * overlay[mask == 1] + alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)
    if border is None:
        border = _glyph_px(4, _glyph_scale(base_bgr.shape[:2]))
    if border > 0:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, border, cv2.LINE_AA)
    return overlay


def ridge_gain(base_bgr, sigma=None, strength=0.9, lo=0.20, hi=1.35):
    """Local ridge contrast as a multiplier centred on 1.0 -- the image is
    high-passed (subtract a local mean) and divided by local std, so the
    result tracks ridge CONTRAST rather than absolute brightness. Mean gain
    is ~1.0 everywhere, so colour keeps its strength on dark and bright
    backgrounds alike, with ridges riding on top as light/dark modulation."""
    g = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    if sigma is None:
        sigma = max(1.5, 6.0 * _glyph_scale(base_bgr.shape[:2]))
    mean = cv2.GaussianBlur(g, (0, 0), sigma)
    detail = g - mean
    std = np.sqrt(cv2.GaussianBlur(detail * detail, (0, 0), sigma)) + 1e-3
    dn = np.clip(detail / (2.0 * std), -1.0, 1.0)
    return np.clip(1.0 + strength * dn, lo, hi)


def dense_orientation_map(theta_deg, seg_mask, base_bgr, mode="ridge", alpha=0.55, strength=0.9):
    """Ridge orientation, angle-coded through the cyclic Tol map, laid over
    the source image; left untouched outside the segmentation.

    mode="ridge" (default): colour scaled by LOCAL ridge contrast (see
        ridge_gain) -- ridges stay crisp and the colour doesn't wash out over
        dark background, at the cost of equalising contrast (a faint region
        reads as boldly as a clear one -- read it for orientation, not as
        evidence of ridge quality).
    mode="blend": flat alpha composite -- even colour strength, but ridges
        wash out as alpha rises.
    """
    idx = np.rint(np.mod(theta_deg, 180.0) / 180.0 * 255.0).astype(np.uint8)
    ori = _ORI_LUT[idx].astype(np.float32)
    back = base_bgr.astype(np.float32)
    if mode == "ridge":
        vis = ori * ridge_gain(base_bgr, strength=strength)[..., None]
    elif mode == "blend":
        vis = alpha * ori + (1.0 - alpha) * back
    else:
        raise ValueError(f"unknown mode {mode!r}; use 'ridge' or 'blend'")
    if seg_mask is not None:
        a = np.clip(seg_mask, 0.0, 1.0)[..., None].astype(np.float32)
        vis = vis * a + back * (1.0 - a)
    return np.clip(vis, 0, 255).astype(np.uint8)


def draw_minutiae_points(base_bgr, minutiae, radius=None, stick_len=None, thickness=None,
                          outline=OUTLINE_DARK, outline_w=None):
    """minutiae: list of {x, y, theta, quality} dicts in base_bgr's pixel grid."""
    gs = _glyph_scale(base_bgr.shape[:2])
    if radius is None:
        radius = _glyph_px(6, gs, 2)
    if stick_len is None:
        stick_len = _glyph_px(18, gs, 4)
    if thickness is None:
        thickness = _glyph_px(2, gs)
    if outline_w is None:
        outline_w = _glyph_px(2, gs)

    vis = base_bgr.copy()
    for m in minutiae:
        x, y, theta = m["x"], m["y"], m["theta"]
        c = (int(round(x)), int(round(y)))
        th = np.radians(theta)
        end = (int(round(x + stick_len * np.cos(th))), int(round(y - stick_len * np.sin(th))))
        if outline is not None and outline_w > 0:
            cv2.line(vis, c, end, outline, thickness + 2 * outline_w, cv2.LINE_AA)
            cv2.circle(vis, c, radius + outline_w, outline, -1, cv2.LINE_AA)
        cv2.line(vis, c, end, MINUTIAE_BGR, thickness, cv2.LINE_AA)
        cv2.circle(vis, c, radius, MINUTIAE_BGR, -1, cv2.LINE_AA)
    return vis


def to_square(img, size=PANEL_TILE, bg=PANEL_BG):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    canvas = np.full((side, side, 3), bg, dtype=np.uint8)
    canvas[top:top + h, left:left + w] = img
    interp = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
    return cv2.resize(canvas, (size, size), interpolation=interp)


def make_feature_panel(views, size=PANEL_TILE, gutter=PANEL_GUTTER, bg=PANEL_BG):
    """views side by side as square tiles, separated by gutters."""
    tiles = [to_square(v, size, bg) for v in views]
    if gutter <= 0:
        return np.hstack(tiles)
    sep = np.full((size, gutter, 3), bg, dtype=np.uint8)
    parts = [sep] * (2 * len(tiles) - 1)
    parts[0::2] = tiles
    return np.hstack(parts)


def render_feature_panel(image, features, tile=PANEL_TILE):
    """Build the [segmentation | orientation | minutiae] row for one image, at
    original-image resolution.

    image: grayscale uint8; features: the dict from AFID.extract_features or
    AFID.run. Returns (panel, views), where views is a dict of the three
    panels at native resolution."""
    base_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    seg_view = overlay_segmentation(base_bgr, features["segmentation"])
    ori_view = dense_orientation_map(features["orientation"], features["segmentation"], base_bgr, mode="ridge")
    min_view = draw_minutiae_points(base_bgr, features["minutiae"])

    panel = make_feature_panel([seg_view, ori_view, min_view], size=tile)
    views = {"segmentation": seg_view, "orientation": ori_view, "minutiae": min_view}
    return panel, views
