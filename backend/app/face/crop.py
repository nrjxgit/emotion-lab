"""Face cropping with margin (research profile margin25 = 0.25 per side)."""

import numpy as np


def crop_face(img: np.ndarray, box: dict, margin: float = 0.25) -> np.ndarray:
    if margin < 0:
        raise ValueError("margin must be >= 0")
    x, y, w, h = (int(box[k]) for k in ("x", "y", "w", "h"))
    if w <= 0 or h <= 0:
        raise ValueError("empty face box")
    H, W = img.shape[:2]
    x0 = max(0, int(x - w * margin))
    y0 = max(0, int(y - h * margin))
    x1 = min(W, int(x + w * (1 + margin)))
    y1 = min(H, int(y + h * (1 + margin)))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError("crop outside image")
    return crop
