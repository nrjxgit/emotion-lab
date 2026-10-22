"""Fit T on TRAIN sample; report ECE/abstention on frozen TEST proba.

Test labels are used only for scoring, never fitting.
"""

import json
import sys

import numpy as np

sys.path.insert(0, "backend/app/face")
sys.path.insert(0, "training")
from fer_images import load_split
from fer_model import load_fer, preprocess, predict_fer
from calibrate_temp import (
    abstention_curve,
    apply_temperature,
    ece_from_proba,
    fit_temperature,
)

Xtr, ytr, _ = load_split("eval_data/fer2013/train")
rng = np.random.default_rng(0)
idx = rng.choice(len(Xtr), 2000, replace=False)
w = load_fer()
Ptr = np.array([predict_fer(preprocess(np.stack([Xtr[i]] * 3, -1)), w) for i in idx])
print("train sample done")
logits_tr = np.log(Ptr + 1e-12)
T = fit_temperature(logits_tr, ytr[idx])
print("T =", round(T, 3))

zt = np.load("eval_results/fer_test_proba.npz")
Pte, yte = zt["proba"], zt["y"]
logits_te = np.log(Pte + 1e-12)
Pcal = apply_temperature(logits_te, T)
out = {
    "T": T,
    "n_fit": 2000,
    "ece_before": ece_from_proba(Pte, yte),
    "ece_after": ece_from_proba(Pcal, yte),
    "acc_before": float((Pte.argmax(1) == yte).mean()),
    "acc_after": float((Pcal.argmax(1) == yte).mean()),
    "abstention": abstention_curve(Pcal, yte),
}
json.dump(out, open("eval_results/calibration.json", "w"), indent=2)
print("ece", round(out["ece_before"], 4), "->", round(out["ece_after"], 4))
for c in out["abstention"]:
    print(f"thr={c['threshold']}: coverage={c['coverage']:.3f} acc={c['accuracy']:.3f}")
