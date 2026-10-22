import numpy as np


def movement_features(track: list) -> dict:
    """track: list of frames, each frame list of (x,y). Returns energy/amplitude/rhythm."""
    if not track or len(track) < 5:
        return {
            "energy": 0.0,
            "amplitude": 0.0,
            "rhythm": 0.0,
            "posture": "unknown",
            "visible": False,
        }
    arr = np.array(track, dtype=float)  # T x K x 2
    diff = np.diff(arr, axis=0)
    energy = float(np.mean(np.linalg.norm(diff, axis=-1)))
    amplitude = float(np.mean(np.std(arr, axis=0)))
    rhythm = float(np.std(np.linalg.norm(diff, axis=-1)))
    return {
        "energy": energy,
        "amplitude": amplitude,
        "rhythm": rhythm,
        "posture": "upright" if energy < 0.2 else "active",
        "visible": True,
    }


def predict_body_emotion(feats: dict) -> dict:
    if not feats.get("visible", False):
        return {
            "probs": {"happiness": 0.1, "neutral": 0.2, "unknown": 0.7},
            "confidence": 0.2,
            "uncertainty": 0.8,
            "visible": False,
            "type": "motion-only",
            "caution": "no body visible; not an emotion reading",
        }
    e = feats.get("energy", 0.0)
    # No fixed psychological labels: high energy → arousal hypothesis only
    if e > 0.08:
        probs = {"happiness": 0.25, "surprise": 0.25, "neutral": 0.2, "unknown": 0.3}
    else:
        probs = {"neutral": 0.5, "sadness": 0.15, "unknown": 0.35}
    # normalize to full 8-class set
    full = {
        "happiness": 0.05,
        "sadness": 0.05,
        "anger": 0.05,
        "surprise": 0.05,
        "fear": 0.05,
        "disgust": 0.05,
        "neutral": 0.1,
        "unknown": 0.1,
    }
    full.update(probs)
    s = sum(full.values())
    full = {k: v / s for k, v in full.items()}
    conf = max(full.values())
    return {
        "probs": full,
        "confidence": float(conf),
        "uncertainty": float(1 - conf),
        "visible": True,
        "movement": feats,
        "type": "motion-only",
        "caution": "movement energy only; NOT a body-emotion prediction (no licensed body-emotion weights)",
    }
