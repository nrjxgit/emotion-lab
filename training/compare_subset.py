"""Same-subset model comparison (stratified 50/class, seed 0).

Usage: python training/compare_subset.py --model minixception [--model hf-vit]
Writes eval_results/compare_<model>.json. Comparison is valid only across
JSONs produced on the same subset file (eval_results/compare_subset.npz).
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "backend/app/face")
sys.path.insert(0, "training")
from fer_images import CLASSES7, load_split

SUBSET = "eval_results/compare_subset.npz"


def build_subset(per_class: int = 50, seed: int = 0):
    X, y, _ = load_split("eval_data/fer2013/test")
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(7):
        ii = np.where(y == c)[0]
        idx.extend(rng.choice(ii, min(per_class, len(ii)), replace=False).tolist())
    idx = np.array(idx)
    np.savez_compressed(SUBSET, X=X[idx], y=y[idx])
    print("subset:", X[idx].shape, "saved", SUBSET)


def run_minixception():
    from fer_model import load_fer, preprocess, predict_fer

    d = np.load(SUBSET)
    X, y = d["X"], d["y"]
    w = load_fer()
    proba = np.array([predict_fer(preprocess(np.stack([im] * 3, -1)), w) for im in X])
    np.savez_compressed("eval_results/compare_minixception.npz", proba=proba, y=y)
    print("minixception subset acc:", float((proba.argmax(1) == y).mean()))


def run_hfvit():
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    import torch

    d = np.load(SUBSET)
    X, y = d["X"], d["y"]
    mid = "dima806/facial_emotions_image_detection"
    proc = AutoImageProcessor.from_pretrained(mid)
    model = AutoModelForImageClassification.from_pretrained(mid).eval()
    labels = [
        model.config.id2label[i].lower() for i in range(len(model.config.id2label))
    ]
    print("hf labels:", labels)
    probs = []
    import torch as _t

    with _t.no_grad():
        for im in X:
            rgb = np.stack([im] * 3, -1)
            inp = proc(images=rgb, return_tensors="pt")
            logits = model(**inp).logits[0]
            probs.append(_t.softmax(logits, -1).numpy())
    np.savez_compressed(
        "eval_results/compare_hfvit.npz",
        proba=np.array(probs),
        y=y,
        labels=np.array(labels),
    )
    print(
        "hf-vit subset acc (native head):",
        float((np.array(probs).argmax(1) == y).mean()),
        "(head mapping documented in report)",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["minixception", "hf-vit"])
    ap.add_argument("--build-subset", action="store_true")
    a = ap.parse_args()
    if a.build_subset or not os.path.exists(SUBSET):
        build_subset()
    {"minixception": run_minixception, "hf-vit": run_hfvit}[a.model]()
