"""Paired FER full-set metrics, plus pinned-checkpoint spot parity on saved scores."""

import json
import hashlib
from pathlib import Path
import numpy as np
from backend.app.face.adapters import ViTAdapter
from training.crop_science import metrics
from training.fer_images import load_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval_results/v0.5"


def main():
    import torch

    torch.set_num_threads(2)
    mini = np.load(ROOT / "eval_results/fer_test_proba.npz")
    vit = np.load(ROOT / "eval_results/vit_test_full.npz")
    X, y, files = load_split(str(ROOT / "eval_data/fer2013/test"))
    if not np.array_equal(y, mini["y"]) or not np.array_equal(y, vit["y"]):
        raise ValueError("full model sample ordering does not match")
    if not np.array_equal(files, mini["files"]):
        raise ValueError("canonical filename order changed")
    adapter = ViTAdapter()
    errors = []
    for idx in (0, len(y) // 2, len(y) - 1):
        p = adapter.predict_proba(np.stack([X[idx]] * 3, -1))
        errors.append(float(np.max(np.abs(p - vit["proba"][idx]))))
    if max(errors) > 1e-5:
        raise ValueError(
            f"pinned checkpoint differs from saved full evaluation: {errors}"
        )
    scores = {
        "checkpoint_spot_parity_max": max(errors),
        "vit_revision": adapter.revision,
        "identical_sample_ids_sha256": hashlib.sha256(
            "\n".join(files).encode()
        ).hexdigest(),
        "n": len(y),
        "overlap_status": "potentially contaminated FER; NOT independent validation",
        "mini_legacy": metrics(y, mini["proba"], np.ones(len(y), bool)),
        "vit": metrics(y, vit["proba"], np.ones(len(y), bool)),
    }
    (OUT / "full_models.json").write_text(json.dumps(scores, indent=2))
    print({k: scores[k]["conditional"] for k in ("mini_legacy", "vit")}, flush=True)


if __name__ == "__main__":
    main()
