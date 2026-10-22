"""Standard adapter evaluation: identical samples, metrics, abstention, timing."""

import time

import numpy as np

from benchmark_fer import benchmark


def evaluate_adapter(
    adapter, X: np.ndarray, y: np.ndarray, dataset: str = "eval"
) -> dict:
    y = np.asarray(y)
    proba, t0 = [], time.time()
    for im in X:
        img = im if im.ndim == 3 else np.stack([im] * 3, -1)
        proba.append(np.asarray(adapter.predict_proba(img), float))
    ms = (time.time() - t0) / max(len(X), 1) * 1000
    proba = np.array(proba)
    m = benchmark(
        list(y), list(proba.argmax(1)), proba=proba, dataset=dataset, n_classes=7
    )
    m["ms_per_image"] = ms
    conf = proba.max(1)
    m["abstention"] = [
        {"threshold": t, "coverage": float((conf >= t).mean())} for t in (0.5, 0.7, 0.9)
    ]
    m["adapter"] = adapter.name
    return m
