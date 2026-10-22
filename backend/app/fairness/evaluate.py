from sklearn.metrics import f1_score, precision_recall_fscore_support
import numpy as np


def stratified_report(y_true, y_pred, groups) -> dict:
    y_true = list(y_true)
    y_pred = list(y_pred)
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out = {"macroF1": macro, "n": len(y_true), "by_group": {}}
    for g in sorted(set(groups)):
        idx = [i for i, x in enumerate(groups) if x == g]
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        out["by_group"][str(g)] = {
            "n": len(idx),
            "macroF1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        }
    # abstention: unknown class = len(CLASSES)-1 placeholder
    out["abstention_rate"] = float(
        sum(1 for p in y_pred if p == "unknown") / max(len(y_pred), 1)
    )
    return out


def anatomy_hypothesis_test(geom_values: list, au_values: list) -> dict:
    """Pearson r with 95% CI (Fisher z). Correlation ≠ causation."""
    from scipy import stats

    x = np.asarray(geom_values, float)
    y = np.asarray(au_values, float)
    if len(x) < 5:
        return {
            "r": 0.0,
            "p": 1.0,
            "ci95": [0, 0],
            "n": len(x),
            "warning": "n<5 underpowered",
        }
    r, p = stats.pearsonr(x, y)
    z = np.arctanh(np.clip(r, -0.999, 0.999))
    se = 1 / np.sqrt(len(x) - 3)
    lo, hi = np.tanh([z - 1.96 * se, z + 1.96 * se])
    return {
        "r": float(r),
        "p": float(p),
        "ci95": [float(lo), float(hi)],
        "n": len(x),
        "warning": "controls needed: angle, age, lighting; do not claim causation",
    }
