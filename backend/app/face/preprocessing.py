"""Canonical face crop + tensor preprocessing (single implementation).

Validated semantics (v0.5 margin25 profile): float YuNet boxes, floor/ceil edge
clamping, resize to 64x64, upstream [-1, 1] normalization. Used by both the serving
API and benchmark studies — identical inputs give identical tensors by construction
(see tests/test_unified_preprocessing.py).

Legacy serving ([0, 1] + tight box + AREA) is intentionally NOT here; it lives
in fer_model.preprocess defaults + main.py legacy branch for reproducibility.
"""

import cv2
import numpy as np

MARGIN25 = 0.25


def crop_pixels(
    img: np.ndarray, box: dict, margin: float = MARGIN25, square: bool = False
) -> np.ndarray:
    x, y, w, h = (float(box[k]) for k in ("x", "y", "w", "h"))
    if not np.isfinite([x, y, w, h]).all() or w <= 0 or h <= 0:
        raise ValueError("invalid face box")
    if margin < 0:
        raise ValueError("margin must be >= 0")
    cx, cy = x + w / 2, y + h / 2
    if square:
        w = h = max(w, h)
    w, h = w * (1 + 2 * margin), h * (1 + 2 * margin)
    H, W = img.shape[:2]
    x0, y0 = max(0, int(np.floor(cx - w / 2))), max(0, int(np.floor(cy - h / 2)))
    x1, y1 = min(W, int(np.ceil(cx + w / 2))), min(H, int(np.ceil(cy + h / 2)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("face box outside image")
    return img[y0:y1, x0:x1]


def tensor_from_crop(
    crop: np.ndarray, interpolation=None, v2: bool = True
) -> np.ndarray:
    """Convert an already-cropped face image to the model tensor.

    Resizes to 64x64, converts to grayscale, rescales to [-1, 1] (upstream
    mini-Xception preprocessing). Identical math to
    preprocess_crop(..., "margin25") without the cropping step.
    """
    if interpolation is None:
        interpolation = cv2.INTER_LINEAR
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    x = (
        cv2.resize(gray, (64, 64), interpolation=interpolation).astype(np.float32)
        / 255.0
    )
    if v2:  # upstream oarriaga v2 pipeline (*2-1 op order, matches study artifacts)
        x = x * 2.0 - 1.0
    return x[..., None].astype(np.float32)


def crop_tensor(
    img: np.ndarray,
    box: dict,
    margin: float = MARGIN25,
    square: bool = False,
    interpolation=None,
    v2: bool = True,
) -> np.ndarray:
    return tensor_from_crop(
        crop_pixels(img, box, margin=margin, square=square), interpolation, v2
    )


def serving_crop_tensor(
    img_bgr: np.ndarray, box: dict, margin: float = MARGIN25
) -> np.ndarray:
    return crop_tensor(
        img_bgr,
        box,
        margin=margin,
        square=False,
        interpolation=cv2.INTER_LINEAR,
        v2=True,
    )
