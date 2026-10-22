import numpy as np


def _d(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.linalg.norm(a - b))


def estimate_aus(pts: dict, baseline: dict | None = None) -> list:
    def get(k, default=(0.5, 0.5)):
        return pts.get(k, baseline.get(k, default) if baseline else default)

    eye_h_L = _d(get("eye_top_L"), get("eye_bot_L"))
    eye_h_R = _d(get("eye_top_R"), get("eye_bot_R"))
    brow_eye_L = _d(get("brow_inner_L"), get("eye_top_L"))
    mouth_w = _d(get("mouth_L"), get("mouth_R"))
    mouth_h = _d(get("mouth_top"), get("mouth_bot"))
    nose_w = _d(get("nose_L"), get("nose_R"))
    b_brow = 0.06
    eye_open = (eye_h_L + eye_h_R) / 2 / 0.05
    brow_raise = (brow_eye_L - b_brow) / 0.03
    mouth_ratio = (mouth_h / max(mouth_w, 1e-6)) / 0.35
    smile_pull = (mouth_w - 0.16) / 0.06
    nose_wrink = max(0.0, (0.055 - nose_w) / 0.03 + max(0.0, mouth_ratio - 1.0) * 0.3)

    def clip(x):
        return float(max(0.0, min(1.0, x)))

    aus = [
        {
            "code": "AU01",
            "name": "inner brow raise",
            "intensity": clip(brow_raise),
            "confidence": 0.7,
            "duration": 0.0,
        },
        {
            "code": "AU02",
            "name": "outer brow raise",
            "intensity": clip(brow_raise * 0.9),
            "confidence": 0.7,
            "duration": 0.0,
        },
        {
            "code": "AU04",
            "name": "brow lower",
            "intensity": clip(-brow_raise),
            "confidence": 0.7,
            "duration": 0.0,
        },
        {
            "code": "AU06",
            "name": "cheek raise",
            "intensity": clip(smile_pull * 0.8),
            "confidence": 0.6,
            "duration": 0.0,
        },
        {
            "code": "AU09",
            "name": "nose wrinkle",
            "intensity": clip(nose_wrink),
            "confidence": 0.6,
            "duration": 0.0,
        },
        {
            "code": "AU12",
            "name": "lip corner pull",
            "intensity": clip(smile_pull),
            "confidence": 0.75,
            "duration": 0.0,
        },
        {
            "code": "AU15",
            "name": "lip corner depress",
            "intensity": clip(-smile_pull),
            "confidence": 0.6,
            "duration": 0.0,
        },
        {
            "code": "AU25",
            "name": "mouth open",
            "intensity": clip(mouth_ratio - 0.6),
            "confidence": 0.8,
            "duration": 0.0,
        },
        {
            "code": "AU43",
            "name": "eye close",
            "intensity": clip(1.0 - eye_open),
            "confidence": 0.8,
            "duration": 0.0,
        },
        {
            "code": "AU05",
            "name": "eye widen",
            "intensity": clip(eye_open - 1.0),
            "confidence": 0.7,
            "duration": 0.0,
        },
    ]
    for a in aus:
        a["quality"] = "ok" if a["confidence"] >= 0.6 else "low"
    return aus


def measure_geometry(pts: dict) -> dict:
    def D(a, b):
        return _d(pts[a], pts[b])

    try:
        face_w = D("cheek_L", "cheek_R")
        return {
            "nose_width_ratio": float(D("nose_L", "nose_R") / max(face_w, 1e-6)),
            "mouth_width_ratio": float(D("mouth_L", "mouth_R") / max(face_w, 1e-6)),
            "brow_distance_ratio": float(
                D("brow_inner_L", "brow_inner_R") / max(face_w, 1e-6)
            ),
            "eye_distance_ratio": float(
                D("eye_top_L", "eye_top_R") / max(face_w, 1e-6)
            ),
            "face_width": float(face_w),
        }
    except KeyError:
        return {
            "nose_width_ratio": 0.0,
            "mouth_width_ratio": 0.0,
            "brow_distance_ratio": 0.0,
            "eye_distance_ratio": 0.0,
            "face_width": 0.0,
        }
