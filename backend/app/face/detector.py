"""Real face detection: YuNet (opencv_zoo, Apache-2.0) via cv2.dnn. No synthetic boxes."""

import cv2
import numpy as np
import os

MODEL = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "models",
    "face_detection_yunet_2023mar.onnx",
)
MODEL = os.path.abspath(MODEL)
_detector = None
_det_size = None


def model_available() -> bool:
    return os.path.exists(MODEL) and os.path.getsize(MODEL) > 100000


def _get_detector(w: int, h: int):
    global _detector, _det_size
    if not model_available():
        return None
    if _detector is None:
        _detector = cv2.FaceDetectorYN_create(MODEL, "", (w, h), 0.6, 0.3, 5000)
        _det_size = (w, h)
    elif _det_size != (w, h):
        _detector.setInputSize((w, h))
        _det_size = (w, h)
    return _detector


def detect_faces(img: np.ndarray) -> list:
    """Returns [] when no face found. Raises ValueError/TypeError on bad input."""
    if img is None:
        raise ValueError("img is None")
    if not isinstance(img, np.ndarray):
        raise TypeError(f"expected np.ndarray, got {type(img)}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected BGR HxWx3, got shape {img.shape}")
    det = _get_detector(img.shape[1], img.shape[0])
    if det is None:
        return []
    _, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        return []
    out = []
    for f in faces:
        x, y, w, h = (int(f[0]), int(f[1]), int(f[2]), int(f[3]))
        out.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "confidence": float(f[14]),
                "landmarks5": {
                    "eye_R": (float(f[4]), float(f[5])),
                    "eye_L": (float(f[6]), float(f[7])),
                    "nose": (float(f[8]), float(f[9])),
                    "mouth_R": (float(f[10]), float(f[11])),
                    "mouth_L": (float(f[12]), float(f[13])),
                },
            }
        )
    return out
