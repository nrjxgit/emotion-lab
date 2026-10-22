"""Baseline: untouched FER2013 test split through the frozen mini-Xception.

Writes eval_results/fer_test_baseline.json. Never trains. Never touches train/.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "backend/app/face")
sys.path.insert(0, "training")
from fer_model import FER7, load_fer, preprocess
from fer_images import CLASSES7, load_split
from benchmark_fer import benchmark

assert CLASSES7 == FER7, f"label order mismatch: {CLASSES7} vs {FER7}"

DATA = "eval_data/fer2013/test"
X, y, files = load_split(DATA)
print(
    "test images:",
    X.shape,
    "classes:",
    {c: int((y == i).sum()) for i, c in enumerate(CLASSES7)},
)

w = load_fer()
t0 = time.time()
proba = []
for i in range(len(X)):
    img3 = np.stack([X[i]] * 3, axis=-1)
    from fer_model import predict_fer

    proba.append(predict_fer(preprocess(img3), w))
    if (i + 1) % 2000 == 0:
        print(f"{i + 1}/{len(X)} ({time.time() - t0:.0f}s)")
proba = np.array(proba)
yp = proba.argmax(axis=1)
print(
    f"inference {time.time() - t0:.1f}s ({(time.time() - t0) / len(X) * 1000:.1f} ms/img)"
)
np.savez_compressed(
    "eval_results/fer_test_proba.npz", proba=proba, y=y, files=np.array(files)
)

m = benchmark(
    list(y), list(yp), proba=proba, dataset="fer2013-test-untouched", n_classes=7
)
m["sec_total"] = round(time.time() - t0, 1)

# disgust investigation: class index 1
d = proba[:, 1]
m["disgust"] = {
    "mean_prob_all": float(d.mean()),
    "max_prob": float(d.max()),
    "mean_prob_on_disgust_rows": float(d[y == 1].mean()),
    "recall_disgust": float(m["per_class_recall"][1]),
    "n_disgust_test": int((y == 1).sum()),
    "predicted_disgust_count": int((yp == 1).sum()),
}
os.makedirs("eval_results", exist_ok=True)
json.dump(
    m,
    open("eval_results/fer_test_baseline.json", "w"),
    indent=1,
    default=lambda o: o.tolist() if hasattr(o, "tolist") else o,
)
print(
    "accuracy:",
    round(m["accuracy"], 4),
    "macroF1:",
    round(m["macroF1"], 4),
    "ece:",
    round(m["ece"], 4),
)
print("per-class recall:", {k: round(v, 3) for k, v in m["per_class_recall"].items()})
print("wrote eval_results/fer_test_baseline.json")
