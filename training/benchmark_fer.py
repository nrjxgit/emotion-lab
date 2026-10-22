"""Labeled benchmark: accuracy, macroF1, per-class recall, confusion, ECE.

Every result carries its dataset name. No synthetic data may pass as real:
callers label synthetic runs explicitly.
"""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import expected_calibration_error


def benchmark(
    y_true, y_pred, proba=None, dataset: str = "unlabeled", n_classes: int = 7
) -> dict:
    y_true = list(y_true)
    y_pred = list(y_pred)
    if not y_true:
        raise ValueError("empty label set: provide labeled rows, never invent them")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    out = {
        "dataset": dataset,
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macroF1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_recall": {
            int(i): float(r)
            for i, r in enumerate(
                recall_score(
                    y_true,
                    y_pred,
                    labels=list(range(n_classes)),
                    average=None,
                    zero_division=0,
                )
            )
        },
        "confusion": cm,
    }
    if proba is not None:
        out["ece"] = expected_calibration_error(y_true, np.asarray(proba, float))
    else:
        out["ece"] = None
    return out
