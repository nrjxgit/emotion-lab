"""FER2013 CSV loader (Kaggle fer2013.csv: emotion,pixels,Usage). No scraping; user supplies file."""

import csv
import numpy as np


def load_fer_csv(path: str, usage: str | None = None, max_n: int | None = None):
    """Returns (X uint8 Nx48x48, y int). Raises FileNotFoundError with access instructions."""
    try:
        fh = open(path, newline="")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{path} not found. FER2013 licence: Kaggle ToU. Get it: Kaggle 'FER-2013' "
            "(register, accept terms, download fer2013.csv) and pass --csv <path>."
        )
    X, y = [], []
    with fh:
        for row in csv.DictReader(fh):
            if usage and row.get("Usage", "").strip() != usage:
                continue
            px = np.fromstring(row["pixels"], sep=" ", dtype=np.uint8)
            if px.size != 2304:
                continue
            X.append(px.reshape(48, 48))
            y.append(int(row["emotion"]))
            if max_n and len(X) >= max_n:
                break
    if not X:
        raise ValueError(f"no rows loaded from {path} (usage={usage})")
    return np.stack(X), np.array(y)
