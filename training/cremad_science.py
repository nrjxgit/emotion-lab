"""CREMA-D label semantics. Surprise is absent in targets, not in predictions."""

from pathlib import Path
import csv
import io
import numpy as np
from training.crop_science import metrics

CODE = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "SAD": 4, "NEU": 6}
VOTE = {"A": 0, "D": 1, "F": 2, "H": 3, "S": 4, "N": 6}


def parse_filename(name):
    parts = Path(name).stem.split("_")
    if len(parts) != 4 or parts[2] not in CODE:
        raise ValueError(f"not a six-class CREMA-D filename: {name}")
    return parts[0], CODE[parts[2]]


def parse_face_vote(value):
    return VOTE.get(value.strip())


def read_summary(text):
    return {
        Path(row["FileName"]).stem: row for row in csv.DictReader(io.StringIO(text))
    }


def score_six(y, proba, detected):
    result = metrics(y, proba, detected)
    # Do not renormalize away Surprise: it is a wrong seven-head prediction.
    ids = [0, 1, 2, 3, 4, 6]
    c = result["end_to_end"]
    pp = np.asarray(proba)
    yy = np.asarray(y)
    detected = np.asarray(detected, bool)
    recall = np.array(c["recall"])[ids]
    precision = np.array(c["precision"])[ids]
    f1 = np.divide(
        2 * recall * precision,
        recall + precision,
        out=np.zeros(6),
        where=recall + precision > 0,
    )
    return {
        **result,
        "accuracy": c["accuracy"],
        "macro_f1_six": float(f1.mean()),
        "classes_six": ["angry", "disgust", "fear", "happy", "sad", "neutral"],
        "recall": recall.tolist(),
        "precision": precision.tolist(),
        "confusion": c["confusion"],
        "surprise_predictions": int(((pp.argmax(1) == 5) & detected).sum()),
    }
