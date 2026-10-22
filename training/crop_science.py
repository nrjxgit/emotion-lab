"""Frozen preprocessing profiles and abstention-aware fixed-label metrics.

No trained weights or dataset pixels live here. Units accept BGR/gray uint8;
coordinates are pixel xywh, margins are per-side fractions, borders are clamped.
"""

import cv2
import numpy as np

FER7 = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
CANDIDATES = (
    "tight_legacy",
    "tight_area",
    "tight_linear",
    "tight_cubic",
    "margin10",
    "margin25",
    "square",
    "aligned_square",
    "source48",
)
CONTROLS = (
    "full_legacy",
    "full_legacy_linear",
    "full_area",
    "full_linear",
    "full_roundtrip",
)


def crop_image(image, box, margin=0.0, square=False):
    x, y, w, h = (float(box[k]) for k in ("x", "y", "w", "h"))
    if not np.isfinite([x, y, w, h]).all() or w <= 0 or h <= 0:
        raise ValueError("invalid face box")
    cx, cy = x + w / 2, y + h / 2
    if square:
        w = h = max(w, h)
    w, h = w * (1 + 2 * margin), h * (1 + 2 * margin)
    H, W = image.shape[:2]
    x0, y0 = max(0, int(np.floor(cx - w / 2))), max(0, int(np.floor(cy - h / 2)))
    x1, y1 = min(W, int(np.ceil(cx + w / 2))), min(H, int(np.ceil(cy + h / 2)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("face box outside image")
    return image[y0:y1, x0:x1]


def align_eyes(image, box):
    eyes = np.asarray([box["landmarks5"]["eye_L"], box["landmarks5"]["eye_R"]], float)
    eyes = eyes[np.argsort(eyes[:, 0])]
    delta = eyes[1] - eyes[0]
    if np.linalg.norm(delta) < 1e-6:
        raise ValueError("degenerate eye locations")
    angle = float(np.degrees(np.arctan2(delta[1], delta[0])))
    center = (box["x"] + box["w"] / 2, box["y"] + box["h"] / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    H, W = image.shape[:2]
    return cv2.warpAffine(
        image, matrix, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    ), matrix


def _canonical():
    """Single shared crop/tensor implementation (backend/app/face/preprocessing)."""
    try:
        from backend.app.face.preprocessing import crop_pixels, tensor_from_crop
    except ImportError:
        try:
            from face.preprocessing import crop_pixels, tensor_from_crop
        except ImportError:
            import sys
            from pathlib import Path

            sys.path.insert(
                0, str(Path(__file__).resolve().parents[1] / "backend" / "app")
            )
            from face.preprocessing import crop_pixels, tensor_from_crop
    return crop_pixels, tensor_from_crop


def preprocess_crop(image, box, strategy, source=None):
    if strategy not in CANDIDATES + CONTROLS:
        raise ValueError(f"unknown frozen strategy: {strategy}")
    interpolation = cv2.INTER_LINEAR
    if strategy in ("tight_legacy", "tight_area", "full_legacy", "full_area"):
        interpolation = cv2.INTER_AREA
    elif strategy == "tight_cubic":
        interpolation = cv2.INTER_CUBIC
    v2 = strategy not in ("tight_legacy", "full_legacy", "full_legacy_linear")
    if strategy.startswith("full_"):
        crop = (
            image
            if strategy == "full_roundtrip"
            else (source if source is not None else image)
        )
        if crop.ndim == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        tensor = (
            cv2.resize(crop, (64, 64), interpolation=interpolation).astype(np.float32)
            / 255
        )
        if v2:
            tensor = tensor * 2 - 1
        return tensor[..., None]
    crop_pixels, tensor_from_crop = _canonical()
    if strategy == "source48":
        if source is None:
            raise ValueError("source48 requires original image")
        ratio_x, ratio_y = (
            source.shape[1] / image.shape[1],
            source.shape[0] / image.shape[0],
        )
        mapped = {
            k: box[k] * (ratio_x if k in ("x", "w") else ratio_y)
            for k in ("x", "y", "w", "h")
        }
        crop = crop_pixels(source, mapped)
    else:
        if strategy == "aligned_square":
            image, _ = align_eyes(image, box)
        crop = crop_pixels(
            image,
            box,
            margin={"margin10": 0.1, "margin25": 0.25}.get(strategy, 0.0),
            square=strategy in ("square", "aligned_square"),
        )
    return tensor_from_crop(crop, interpolation, v2)


def metrics(y, proba, detected, n_classes=7):
    y = np.asarray(y, int)
    p = np.asarray(proba, float)
    detected = np.asarray(detected, bool)
    if len(y) == 0 or p.shape != (len(y), n_classes) or detected.shape != y.shape:
        raise ValueError(
            "metrics require nonempty matching labels/probabilities/detections"
        )
    if not ((y >= 0) & (y < n_classes)).all():
        raise ValueError("labels outside class mapping")
    if detected.any() and (
        not np.isfinite(p[detected]).all()
        or (p[detected] < 0).any()
        or not np.allclose(p[detected].sum(1), 1, atol=1e-5)
    ):
        raise ValueError("invalid probability rows")
    pred = np.where(detected, p.argmax(1), n_classes)
    cm = np.zeros((n_classes, n_classes + 1), int)
    np.add.at(cm, (y, pred), 1)

    def score(cm_, denominator):
        tp = np.diag(cm_[:, :n_classes])
        support = cm_.sum(1)
        precision = np.divide(
            tp,
            cm_[:, :n_classes].sum(0),
            out=np.zeros(n_classes),
            where=cm_[:, :n_classes].sum(0) > 0,
        )
        recall = np.divide(tp, support, out=np.zeros(n_classes), where=support > 0)
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros(n_classes),
            where=precision + recall > 0,
        )
        return {
            "n": int(denominator),
            "accuracy": float(tp.sum() / denominator) if denominator else None,
            "macro_f1": float(f1.mean()) if denominator else None,
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "support": support.tolist(),
            "confusion": cm_.tolist(),
        }

    e2e = score(cm, len(y))
    cond_cm = cm.copy()
    cond_cm[:, -1] = 0
    cond = score(cond_cm, detected.sum())
    cond["ece"] = cond["nll"] = cond["brier"] = None
    if detected.any():
        pp, yy = p[detected], y[detected]
        confidence = pp.max(1)
        correct = pp.argmax(1) == yy
        ece = 0.0
        for i in range(10):
            mask = (confidence >= i / 10) & (
                (confidence < (i + 1) / 10) if i < 9 else True
            )
            if mask.any():
                ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
        cond.update(
            ece=float(ece),
            nll=float(-np.log(np.clip(pp[np.arange(len(yy)), yy], 1e-12, 1)).mean()),
            brier=float(((pp - np.eye(n_classes)[yy]) ** 2).sum(1).mean()),
        )
    return {
        "n": len(y),
        "detected": int(detected.sum()),
        "coverage": float(detected.mean()),
        "classes": FER7[:n_classes],
        "abstentions": int((~detected).sum()),
        "conditional": cond,
        "end_to_end": e2e,
    }


def select_validation_winner(scores, split):
    if split != "validation":
        raise ValueError("selection is allowed only on validation")
    return max(scores, key=lambda name: scores[name]["conditional"]["macro_f1"])
