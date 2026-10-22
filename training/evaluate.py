"""Scientific evaluation: confusion, macroF1, per-class PR, ECE calibration, abstention."""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    log_loss,
)


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += m.mean() * abs((pred[m] == y_true[m]).mean() - conf[m].mean())
    return float(ece)


def evaluate(y_true, y_pred, proba=None, n_classes: int | None = None) -> dict:
    y_true = list(y_true)
    y_pred = list(y_pred)
    n_classes = n_classes or (max(max(y_true), max(y_pred)) + 1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )
    out = {
        "macroF1": macro,
        "confusion": cm,
        "per_class": {
            i: {"p": float(p[i]), "r": float(r[i]), "f1": float(f[i])}
            for i in range(n_classes)
        },
        "n": len(y_true),
        "abstention_rate": float(
            sum(1 for v in y_pred if v == "unknown") / max(len(y_pred), 1)
        ),
    }
    if proba is not None:
        proba = np.asarray(proba, float)
        out["ece"] = expected_calibration_error(y_true, proba)
        try:
            out["logloss"] = float(
                log_loss(y_true, proba, labels=list(range(n_classes)))
            )
        except Exception:
            out["logloss"] = float("nan")
    else:
        out["ece"] = 0.0
    return out
