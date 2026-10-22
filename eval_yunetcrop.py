"""Deployment-path eval: upscale 48px stills, YuNet-detect, FER on the crop.

Reports detection rate + accuracy on detected crops (vs pre-cropped baseline).
Slow (~3 forwards/img); run unattended. Subset flag for smoke runs.
"""

import argparse
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "backend/app/face")
sys.path.insert(0, "training")
from detector import detect_faces
from fer_images import CLASSES7, load_split
from fer_model import load_fer, preprocess, predict_fer

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=0, help="0 = full test set")
ap.add_argument("--up", type=int, default=192)
a = ap.parse_args()

X, y, _ = load_split("eval_data/fer2013/test")
if a.n:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), a.n, replace=False)
    X, y = X[idx], y[idx]
w = load_fer()
det, ok = 0, 0
for i, im in enumerate(X):
    big = cv2.resize(
        cv2.cvtColor(np.stack([im] * 3, -1), cv2.COLOR_BGR2RGB), (a.up, a.up)
    )[:, :, ::-1]
    faces = detect_faces(cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
    if not faces:
        continue
    b = max(faces, key=lambda f: f["w"] * f["h"])
    H, W, _ = big.shape
    x0, y0 = max(0, b["x"]), max(0, b["y"])
    x1, y1 = min(W, b["x"] + b["w"]), min(H, b["y"] + b["h"])
    if x1 <= x0 or y1 <= y0:
        continue
    det += 1
    crop = big[y0:y1, x0:x1][:, :, ::-1]
    ok += int(predict_fer(preprocess(crop), w).argmax() == y[i])
    if (i + 1) % 500 == 0:
        print(f"{i + 1}/{len(X)} det={det}", flush=True)
out = {
    "n": len(X),
    "upscale": a.up,
    "detected": det,
    "det_rate": det / len(X),
    "acc_on_detected": (ok / det) if det else 0.0,
}
json.dump(out, open("eval_results/yunetcrop.json", "w"), indent=1)
print(out)
