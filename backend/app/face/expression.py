"""Expression: trained mini-Xception first; AU-rule mapping kept as separate observable analysis."""

import numpy as np

CLASSES = [
    "happiness",
    "sadness",
    "anger",
    "surprise",
    "fear",
    "disgust",
    "neutral",
    "unknown",
]
# FER7 (oarriaga order) -> our 8 classes
FERMAP = {
    "happy": "happiness",
    "sad": "sadness",
    "angry": "anger",
    "surprise": "surprise",
    "fear": "fear",
    "disgust": "disgust",
    "neutral": "neutral",
}


class ExpressionPredictor:
    TRAINED_SOURCE = "trained-minixception-fer2013"
    PROFILES = ("corrected", "legacy")
    # T fit on validation_margin25 (n=686 detected), confirmed on frozen test
    # (ECE 0.0597 -> 0.0495). Valid ONLY for corrected profile + mini-X weights.
    TEMPERATURE = 1.045

    def __init__(self, profile: str = "corrected"):
        if profile not in self.PROFILES:
            raise ValueError(f"unknown profile {profile!r}; choices {self.PROFILES}")
        self.profile = profile
        self.source = "unavailable"
        self._w = None
        self._status = "not loaded"
        try:
            try:
                from .fer_model import load_fer, weights_available
            except ImportError:
                from fer_model import load_fer, weights_available
            if weights_available():
                self._w = load_fer()
                self.source = self.TRAINED_SOURCE
                self._status = f"loaded {len(self._w)} layers"
            else:
                self._status = "hdf5 missing"
        except Exception as e:
            self._status = f"load failed: {e}"

    def predict(
        self, face_crop: np.ndarray, aus: list, calibrated: bool = False
    ) -> dict:
        if self._w is not None:
            try:
                try:
                    from .fer_model import preprocess, predict_fer, FER7
                except ImportError:
                    from fer_model import preprocess, predict_fer, FER7
                import cv2 as _cv2

                if self.profile == "corrected":
                    try:
                        from .preprocessing import tensor_from_crop
                    except ImportError:
                        from preprocessing import tensor_from_crop

                    tensor = tensor_from_crop(face_crop, _cv2.INTER_LINEAR, True)
                else:
                    tensor = preprocess(face_crop)
                p7 = [float(v) for v in predict_fer(tensor, self._w)]
                raw7 = list(p7)
                if calibrated:
                    if self.profile != "corrected":
                        raise ValueError(
                            "calibration validated only for corrected profile"
                        )
                    z = np.log(np.clip(np.asarray(p7), 1e-12, 1.0)) / self.TEMPERATURE
                    z = z - z.max()
                    e = np.exp(z)
                    p7 = list(e / e.sum())

                def _finish(p7vals):
                    pr = {c: 0.0 for c in CLASSES}
                    for name, p in zip(FER7, [float(v) for v in p7vals]):
                        pr[FERMAP[name]] += p
                    # unknown mass from entropy: confident -> little unknown
                    mx = max(pr.values())
                    pr["unknown"] = max(0.0, (1.0 - mx) * 0.25)
                    s = sum(pr.values())
                    return {k: v / s for k, v in pr.items()}

                probs = _finish(p7)
                conf = float(max(probs.values()))
                m = {a.get("code"): float(a.get("intensity", 0)) for a in (aus or [])}
                arousal = max(m.get("AU05", 0), m.get("AU25", 0), 0.0)
                express = float(min(1.0, sum(m.values()) / 3.0)) if m else 0.0
                valence = float(
                    probs["happiness"]
                    - probs["sadness"]
                    - probs["anger"] * 0.7
                    - probs["disgust"] * 0.5
                )
                out = {
                    "probs": probs,
                    "valence": max(-1, min(1, valence)),
                    "arousal": float(max(0, min(1, arousal))),
                    "expressiveness": express,
                    "confidence": conf,
                    "uncertainty": float(1 - conf),
                    "source": self.source,
                    "profile": self.profile,
                    "calibrated": calibrated,
                }
                if calibrated:
                    out["probs_raw"] = _finish(raw7)
                    out["temperature"] = self.TEMPERATURE
                    out["note"] = (
                        "calibrated confidence = better probability estimates, "
                        "not certainty about feelings"
                    )
                return out
            except Exception as e:
                return {
                    "probs": {c: (1.0 if c == "unknown" else 0.0) for c in CLASSES},
                    "valence": 0.0,
                    "arousal": 0.0,
                    "expressiveness": 0.0,
                    "confidence": 0.15,
                    "uncertainty": 0.85,
                    "source": f"error: {e}",
                }
        return {
            "probs": {c: (1.0 if c == "unknown" else 0.0) for c in CLASSES},
            "valence": 0.0,
            "arousal": 0.0,
            "expressiveness": 0.0,
            "confidence": 0.15,
            "uncertainty": 0.85,
            "source": self._status,
        }


def observable_feature_analysis(aus: list) -> dict:
    """Heuristic AU->emotion resemblance. Label: observable-feature-analysis (NOT a trained model)."""
    m = {a.get("code"): float(a.get("intensity", 0)) for a in (aus or [])}
    scores = {
        "happiness": m.get("AU12", 0) * 2 + m.get("AU06", 0),
        "sadness": m.get("AU15", 0) * 1.6 + m.get("AU01", 0) * 0.8,
        "anger": m.get("AU04", 0) * 1.8,
        "surprise": m.get("AU05", 0) * 1.4 + m.get("AU25", 0),
        "fear": m.get("AU05", 0) + m.get("AU01", 0) * 0.8,
        "disgust": m.get("AU09", 0) * 2.0,
        "neutral": max(0.0, 0.6 - sum(m.values()) * 0.2),
        "unknown": 0.4,
    }
    s = sum(scores.values()) or 1.0
    return {
        "probs": {k: v / s for k, v in scores.items()},
        "source": "observable-feature-analysis",
    }
