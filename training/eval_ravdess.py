"""RAVDESS filename decode + abstention-aware summarize (pure, no data needed)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_fer import benchmark  # noqa: E402

RAVDESS_MAP = {
    "neutral": 1,
    "calm": 2,
    "happy": 3,
    "sad": 4,
    "angry": 5,
    "fearful": 6,
    "disgust": 7,
    "surprised": 8,
}
ID2EMO = {v: k for k, v in RAVDESS_MAP.items()}

FER7_INDEX = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6,
}
TO_FER7 = {
    "neutral": "neutral",
    "calm": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fearful": "fear",
    "disgust": "disgust",
    "surprised": "surprise",
}


def decode_ravdess(fname: str):
    parts = os.path.basename(fname).replace(".mp4", "").split("-")
    if len(parts) != 7:
        raise ValueError(f"not a RAVDESS filename: {fname}")
    return ID2EMO[int(parts[2])], int(parts[6])


def to_fer7(ravdess_emo: str, include_calm: bool = True):
    if ravdess_emo == "calm" and not include_calm:
        return None
    return TO_FER7[ravdess_emo]


def summarize(rows: list, n_classes: int = 7) -> dict:
    yt, yp, pr = [], [], []
    abst = 0
    for r in rows:
        if not r.get("face") or r.get("pred") is None:
            abst += 1
            continue
        yt.append(r["y"])
        yp.append(r["pred"])
        if "proba" in r:
            pr.append(r["proba"])
    m = benchmark(
        yt, yp, proba=(pr or None), dataset="ravdess-frames", n_classes=n_classes
    )
    return {"n": len(rows), "abstentions": abst, "metrics": m}
