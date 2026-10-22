"""Full FER2013-test ViT eval THROUGH the pinned-revision adapter.

Supersedes the unpinned historical run: identical sample order, mapped head,
provenance recorded. Slow (~40 min CPU); run unattended.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "backend/app/face")
sys.path.insert(0, "training")
from adapters import ViTAdapter, FER7
from fer_images import CLASSES7, load_split

assert CLASSES7 == FER7
torch.set_num_threads(2)
X, y, files = load_split("eval_data/fer2013/test")
adapter = ViTAdapter()  # pinned revision VIT_REVISION
assert adapter.revision == "747cf16692eea925b54b7b543cf436848128b68d"
proba = []
for i, im in enumerate(X):
    proba.append(adapter.predict_proba(np.stack([im] * 3, -1)))
    if (i + 1) % 1000 == 0:
        print(f"{i + 1}/{len(X)}", flush=True)
proba = np.array(proba)
np.savez_compressed(
    "eval_results/vit_test_full_pinned.npz",
    proba=proba,
    y=y,
    files=np.array(files),
    revision=adapter.revision,
)
print("done", proba.shape, float((proba.argmax(1) == y).mean()))
