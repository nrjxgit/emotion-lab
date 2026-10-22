"""Train/test exact-duplicate analysis + ViT abstention/ECE from saved arrays.

No inference, no fitting: reads pixels for hashing + saved proba arrays.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from calibrate_temp import abstention_curve, ece_from_proba
from fer_images import load_split


def flag_test_rows_in_train(test_hashes, train_hash_set) -> np.ndarray:
    return np.array([h in train_hash_set for h in test_hashes])


def main():
    Xtr, _, _ = load_split(str(ROOT / "eval_data" / "fer2013" / "train"))
    train_hashes = {hashlib.sha256(im.tobytes()).hexdigest() for im in Xtr}
    Xte, yte, _ = load_split(str(ROOT / "eval_data" / "fer2013" / "test"))
    test_hashes = [hashlib.sha256(im.tobytes()).hexdigest() for im in Xte]
    dup = flag_test_rows_in_train(test_hashes, train_hashes)
    proba = np.load(ROOT / "eval_results" / "fer_test_proba.npz")
    y, p = proba["y"], proba["proba"]
    assert (y == yte).all()
    vit = np.load(ROOT / "eval_results" / "vit_test_full.npz")
    vp, vy = vit["proba"], vit["y"]
    out = {
        "n_train": len(Xtr),
        "n_test": len(yte),
        "n_test_with_exact_train_duplicate": int(dup.sum()),
        "mini_acc_on_duplicates": float((p[dup].argmax(1) == y[dup]).mean())
        if dup.any()
        else None,
        "mini_acc_on_rest": float((p[~dup].argmax(1) == y[~dup]).mean()),
        "mini_ece_on_duplicates": ece_from_proba(p[dup], y[dup]) if dup.any() else None,
        "vit_abstention_full_test": abstention_curve(
            vp, vy, [0.0, 0.5, 0.7, 0.9, 0.95]
        ),
        "vit_ece_full_test": ece_from_proba(vp, vy),
        "vit_overall_acc": float((vp.argmax(1) == vy).mean()),
        "warning": "duplicate rows inflate the score of any model that saw FER2013 train; "
        "ViT's own training membership remains unverified (see VIT_FORENSICS.md)",
    }
    Path("eval_results/v0.6").mkdir(parents=True, exist_ok=True)
    Path("eval_results/v0.6/overlap.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
