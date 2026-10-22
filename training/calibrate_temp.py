"""Temperature scaling (fit on validation only) + confidence abstention curves."""

import numpy as np


def _softmax(z):
    z = np.asarray(z, float)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece_from_proba(proba, y_true, n_bins: int = 10) -> float:
    proba = np.asarray(proba, float)
    y_true = np.asarray(y_true)
    conf, pred = proba.max(1), proba.argmax(1)
    ece = 0.0
    for b in range(n_bins):
        m = (conf > b / n_bins) & (conf <= (b + 1) / n_bins)
        if m.sum():
            ece += m.mean() * abs((pred[m] == y_true[m]).mean() - conf[m].mean())
    return float(ece)


def fit_temperature(logits, y_true) -> float:
    """Grid-search T minimizing NLL (validation data only)."""
    logits = np.asarray(logits, float)
    y_true = np.asarray(y_true)
    best_T, best_nll = 1.0, float("inf")
    for T in np.concatenate([np.linspace(0.05, 5, 200)]):
        p = _softmax(logits / T)
        nll = -np.log(p[np.arange(len(y_true)), y_true] + 1e-12).mean()
        if nll < best_nll:
            best_T, best_nll = T, nll
    return float(best_T)


def apply_temperature(logits, T: float):
    return _softmax(np.asarray(logits, float) / T)


def abstention_curve(proba, y_true, thresholds=(0.0, 0.5, 0.7, 0.9)) -> list:
    proba = np.asarray(proba, float)
    y_true = np.asarray(y_true)
    conf = proba.max(1)
    out = []
    for t in thresholds:
        m = conf >= t
        out.append(
            {
                "threshold": float(t),
                "coverage": float(m.mean()),
                "accuracy": float((proba[m].argmax(1) == y_true[m]).mean())
                if m.sum()
                else 0.0,
                "n": int(m.sum()),
            }
        )
    return out
