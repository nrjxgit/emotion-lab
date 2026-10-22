"""OC-ADDITIONAL-20261019-A — mini-Xception transfer training on FER2013 train (CPU).

Bounded experiment: mini-Xception only (ViT left pending — 343MB transformer not
feasible on CPU-only hardware). Trains on FER2013 train images already local to
the repo; test/ (7,178 images) is FROZEN and never used for training or dev.

BLOCKED: This experiment cannot execute on real images until independent rights,
adult-provenance, duplicate/group-separation and protected-exclusion evidence is
established. The FER2013 ZIP has no bundled licence and no subject IDs; these
gates are not satisfied. See BLOCKED receipt under outputs/.

The script supports --dry-run for CI/synthetic verification only. No real image
decoding or training occurs unless gates pass.

Deterministic: fixed seed, fixed config, train/dev split manifest created BEFORE
any training output is observed. Time/epoch cap suited to CPU hardware.

Outputs: checkpoint (safetensors), split manifest, train/dev metrics log,
model card with limitations and exposure disclosure.

No dependency installations; uses torch (already present), numpy, cv2, h5py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Deterministic configuration (fixed before any outcome)
# ---------------------------------------------------------------------------
SEED = 505
FER7 = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_CLASSES = len(FER7)
IMG_SIZE = 64
BATCH_SIZE = 32
MAX_EPOCHS = 8
LR = 1e-3
TIME_CAP_SECONDS = 600  # 10 minutes max wall time on CPU
TRAIN_FRAC = 0.9  # 90% train, 10% dev from FER2013 train images
VAL_FRAC = 0.1

HDF5_WEIGHTS = Path("models/fer_mini_xception_h5.hdf5")
TRAIN_DIR = Path("eval_data/fer2013/train")
OUTPUT_DIR = Path("outputs/training-oc-additional-20261019-a")
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Data loading (FER2013 train images) — GATED: real reads only after gates pass
# ---------------------------------------------------------------------------
def load_fer2013_train(train_dir: Path):
    """Load all FER2013 train images. Returns X (N,64,64,1 uint8), y (N,), files."""
    X, y, files = [], [], []
    for ci, cls in enumerate(FER7):
        cls_dir = train_dir / cls
        if not cls_dir.exists():
            continue
        for p in sorted(cls_dir.glob("*.jpg")):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (IMG_SIZE, IMG_SIZE):
                img = cv2.resize(
                    img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
                )
            X.append(img)
            y.append(ci)
            files.append(os.path.basename(p))
    if not X:
        raise ValueError(f"no images under {train_dir}")
    return np.stack(X), np.array(y), files


def preprocess_for_training(X_uint8: np.ndarray) -> np.ndarray:
    """gray/255 then [-1,1] per upstream oarriaga v2 (matches training profile)."""
    return (X_uint8.astype(np.float32) / 255.0) * 2.0 - 1.0


def deterministic_split(X, y, files, train_frac: float, seed: int):
    """Stratified deterministic train/dev split by class. Returns indices."""
    rng = np.random.default_rng(seed)
    train_idx, dev_idx = [], []
    for c in range(NUM_CLASSES):
        c_idx = np.where(y == c)[0]
        rng.shuffle(c_idx)
        n_train = int(len(c_idx) * train_frac)
        train_idx.extend(c_idx[:n_train])
        dev_idx.extend(c_idx[n_train:])
    train_idx = np.array(sorted(train_idx))
    dev_idx = np.array(sorted(dev_idx))
    return train_idx, dev_idx


# ---------------------------------------------------------------------------
# mini-Xception PyTorch architecture (matches HDF5 weight shapes)
# ---------------------------------------------------------------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class MiniXceptionBlock(nn.Module):
    """sepconv → BN → ReLU → sepconv → BN → ReLU → maxpool(stride=2).
       Shortcut: conv1x1(stride=2) → BN. Add + ReLU.

    Both main branch (maxpool stride 2) and shortcut (conv1x1 stride 2)
    halve spatial resolution once — no double downsampling.
    """

    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.sep1 = DepthwiseSeparableConv(in_ch, out_ch, stride=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.sep2 = DepthwiseSeparableConv(out_ch, out_ch, stride=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Conv2d(
            in_ch, out_ch, 1, stride=stride, padding=0, bias=False
        )
        self.bn_shortcut = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(3, stride=stride, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        m = self.relu(self.bn1(self.sep1(x)))  # stride 1, full res
        m = self.relu(self.bn2(self.sep2(m)))  # stride 1, full res
        m = self.pool(m)  # stride 2, half res
        s = self.bn_shortcut(self.shortcut(x))  # stride 2, half res
        return self.relu(m + s)  # both half res → add OK


class MiniXceptionPyTorch(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, 3, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(8)
        self.block1 = MiniXceptionBlock(8, 16, stride=2)
        self.block2 = MiniXceptionBlock(16, 32, stride=2)
        self.block3 = MiniXceptionBlock(32, 64, stride=2)
        self.block4 = MiniXceptionBlock(64, 128, stride=2)
        self.conv7 = nn.Conv2d(128, num_classes, 3, stride=1, padding=1, bias=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # x: (N, 1, 64, 64) -> returns RAW LOGITS (N, 7), NOT softmax
        x = F.relu(self.bn1(self.conv1(x)))  # 62x62x8
        x = F.relu(self.bn2(self.conv2(x)))  # 60x60x8
        x = self.block1(x)  # 30x30x16
        x = self.block2(x)  # 15x15x32
        x = self.block3(x)  # 8x8x64
        x = self.block4(x)  # 4x4x128
        x = self.conv7(x)  # 4x4x7
        x = self.gap(x)  # 1x1x7
        x = x.squeeze(-1).squeeze(-1)  # 7 raw logits
        return x  # NO softmax — CrossEntropyLoss expects raw logits


# ---------------------------------------------------------------------------
# HDF5 weight loading (channels-last → channels-first transposition)
# ---------------------------------------------------------------------------
def load_hdf5_weights(hdf5_path: Path) -> dict:
    import h5py

    f = h5py.File(str(hdf5_path), "r")
    mw = f["model_weights"]
    weights = {}
    for layer in mw.keys():
        g = mw[layer]
        inner_names = list(g.keys())
        if not inner_names:
            continue
        inner = g[inner_names[0]]
        if not hasattr(inner, "keys"):
            continue
        params = {}
        for ds in inner.keys():
            o = inner[ds]
            if hasattr(o, "shape"):
                params[ds.split(":")[0]] = np.asarray(o, dtype=np.float32)
        weights[layer] = params
    f.close()
    return weights


def load_pretrained_into_pytorch(model: MiniXceptionPyTorch, hdf5_weights: dict):
    """Load HDF5 weights into PyTorch model with channel transposition."""
    state = model.state_dict()

    # conv1: (3,3,1,8) → (8,1,3,3)
    state["conv1.weight"] = torch.from_numpy(
        hdf5_weights["conv2d_1"]["kernel"].transpose(3, 2, 0, 1)
    )
    # conv2: (3,3,8,8) → (8,8,3,3)
    state["conv2.weight"] = torch.from_numpy(
        hdf5_weights["conv2d_2"]["kernel"].transpose(3, 2, 0, 1)
    )

    # BN weights for conv1 and conv2 (were missing)
    h = hdf5_weights["batch_normalization_1"]
    state["bn1.weight"] = torch.from_numpy(h["gamma"])
    state["bn1.bias"] = torch.from_numpy(h["beta"])
    state["bn1.running_mean"] = torch.from_numpy(h["moving_mean"])
    state["bn1.running_var"] = torch.from_numpy(h["moving_variance"])
    h = hdf5_weights["batch_normalization_2"]
    state["bn2.weight"] = torch.from_numpy(h["gamma"])
    state["bn2.bias"] = torch.from_numpy(h["beta"])
    state["bn2.running_mean"] = torch.from_numpy(h["moving_mean"])
    state["bn2.running_var"] = torch.from_numpy(h["moving_variance"])

    # Block 1
    # sep1 depthwise: (3,3,8,1) → (8,1,3,3)
    state["block1.sep1.depthwise.weight"] = torch.from_numpy(
        hdf5_weights["separable_conv2d_1"]["depthwise_kernel"].transpose(2, 3, 0, 1)
    )
    # sep1 pointwise: (1,1,8,16) → (16,8,1,1)
    state["block1.sep1.pointwise.weight"] = torch.from_numpy(
        hdf5_weights["separable_conv2d_1"]["pointwise_kernel"].transpose(3, 2, 0, 1)
    )
    # sep2 depthwise: (3,3,16,1) → (16,1,3,3)
    state["block1.sep2.depthwise.weight"] = torch.from_numpy(
        hdf5_weights["separable_conv2d_2"]["depthwise_kernel"].transpose(2, 3, 0, 1)
    )
    # sep2 pointwise: (1,1,16,16) → (16,16,1,1)
    state["block1.sep2.pointwise.weight"] = torch.from_numpy(
        hdf5_weights["separable_conv2d_2"]["pointwise_kernel"].transpose(3, 2, 0, 1)
    )
    # shortcut: (1,1,8,16) → (16,8,1,1)
    state["block1.shortcut.weight"] = torch.from_numpy(
        hdf5_weights["conv2d_3"]["kernel"].transpose(3, 2, 0, 1)
    )
    # BN params
    for bn_name, hdf5_name in [
        ("block1.bn1", "batch_normalization_4"),
        ("block1.bn2", "batch_normalization_5"),
        ("block1.bn_shortcut", "batch_normalization_3"),
    ]:
        h = hdf5_weights[hdf5_name]
        state[f"{bn_name}.weight"] = torch.from_numpy(h["gamma"])
        state[f"{bn_name}.bias"] = torch.from_numpy(h["beta"])
        state[f"{bn_name}.running_mean"] = torch.from_numpy(h["moving_mean"])
        state[f"{bn_name}.running_var"] = torch.from_numpy(h["moving_variance"])

    # Block 2
    state["block2.sep1.depthwise.weight"] = torch.from_numpy(
        hdf5_weights["separable_conv2d_3"]["depthwise_kernel"].transpose(2, 3, 0, 1)
    )
    state["block2.sep1.pointwise.weight"] = torch.from_numpy(
        hdf5_weights["separable_conv2d_3"]["pointwise_kernel"].transpose(3, 2, 0, 1)
    )
    state["block2.sep2.depthwise.weight"] = torch.from_numpy(
        hdf5_weights["separable_conv2d_4"]["depthwise_kernel"].transpose(2, 3, 0, 1)
    )
    state["block2.sep2.pointwise.weight"] = torch.from_numpy(
        hdf5_weights["separable_conv2d_4"]["pointwise_kernel"].transpose(3, 2, 0, 1)
    )
    state["block2.shortcut.weight"] = torch.from_numpy(
        hdf5_weights["conv2d_4"]["kernel"].transpose(3, 2, 0, 1)
    )
    for bn_name, hdf5_name in [
        ("block2.bn1", "batch_normalization_7"),
        ("block2.bn2", "batch_normalization_8"),
        ("block2.bn_shortcut", "batch_normalization_6"),
    ]:
        h = hdf5_weights[hdf5_name]
        state[f"{bn_name}.weight"] = torch.from_numpy(h["gamma"])
        state[f"{bn_name}.bias"] = torch.from_numpy(h["beta"])
        state[f"{bn_name}.running_mean"] = torch.from_numpy(h["moving_mean"])
        state[f"{bn_name}.running_var"] = torch.from_numpy(h["moving_variance"])

    # Block 3
    state["block3.sep1.depthwise.weight"] = torch.from_numpy(
        hdf5_weights["separable_conv2d_5"]["depthwise_kernel"].transpose(2, 3, 0, 1)
    )
    state["block3.sep1.pointwise.weight"] = torch.from_numpy(
        hdf5_weights["separable_conv2d_5"]["pointwise_kernel"].transpose(3, 2, 0, 1)
    )
    state["block3.sep2.depthwise.weight"] = torch.from_numpy(
        hdf5_weights["separable_conv2d_6"]["depthwise_kernel"].transpose(2, 3, 0, 1)
    )
    state["block3.sep2.pointwise.weight"] = torch.from_numpy(
        hdf5_weights["separable_conv2d_6"]["pointwise_kernel"].transpose(3, 2, 0, 1)
    )
    state["block3.shortcut.weight"] = torch.from_numpy(
        hdf5_weights["conv2d_5"]["kernel"].transpose(3, 2, 0, 1)
    )
    for bn_name, hdf5_name in [
        ("block3.bn1", "batch_normalization_10"),
        ("block3.bn2", "batch_normalization_11"),
        ("block3.bn_shortcut", "batch_normalization_9"),
    ]:
        h = hdf5_weights[hdf5_name]
        state[f"{bn_name}.weight"] = torch.from_numpy(h["gamma"])
        state[f"{bn_name}.bias"] = torch.from_numpy(h["beta"])
        state[f"{bn_name}.running_mean"] = torch.from_numpy(h["moving_mean"])
        state[f"{bn_name}.running_var"] = torch.from_numpy(h["moving_variance"])

    # Block 4
    state["block4.sep1.depthwise.weight"] = torch.from_numpy(
        hdf5_weights["separable_conv2d_7"]["depthwise_kernel"].transpose(2, 3, 0, 1)
    )
    state["block4.sep1.pointwise.weight"] = torch.from_numpy(
        hdf5_weights["separable_conv2d_7"]["pointwise_kernel"].transpose(3, 2, 0, 1)
    )
    state["block4.sep2.depthwise.weight"] = torch.from_numpy(
        hdf5_weights["separable_conv2d_8"]["depthwise_kernel"].transpose(2, 3, 0, 1)
    )
    state["block4.sep2.pointwise.weight"] = torch.from_numpy(
        hdf5_weights["separable_conv2d_8"]["pointwise_kernel"].transpose(3, 2, 0, 1)
    )
    state["block4.shortcut.weight"] = torch.from_numpy(
        hdf5_weights["conv2d_6"]["kernel"].transpose(3, 2, 0, 1)
    )
    for bn_name, hdf5_name in [
        ("block4.bn1", "batch_normalization_13"),
        ("block4.bn2", "batch_normalization_14"),
        ("block4.bn_shortcut", "batch_normalization_12"),
    ]:
        h = hdf5_weights[hdf5_name]
        state[f"{bn_name}.weight"] = torch.from_numpy(h["gamma"])
        state[f"{bn_name}.bias"] = torch.from_numpy(h["beta"])
        state[f"{bn_name}.running_mean"] = torch.from_numpy(h["moving_mean"])
        state[f"{bn_name}.running_var"] = torch.from_numpy(h["moving_variance"])

    # conv7: (3,3,128,7) → (7,128,3,3)
    state["conv7.weight"] = torch.from_numpy(
        hdf5_weights["conv2d_7"]["kernel"].transpose(3, 2, 0, 1)
    )
    state["conv7.bias"] = torch.from_numpy(hdf5_weights["conv2d_7"]["bias"])

    model.load_state_dict(state)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
class FERDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X  # (N, 64, 64, 1) float32 in [-1,1]
        self.y = y  # (N,) int64

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]).unsqueeze(0),  # (1, 64, 64)
            torch.tensor(self.y[idx], dtype=torch.long),
        )


def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)  # raw logits — correct for CrossEntropyLoss
            loss = criterion(logits, yb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            loss_sum += loss.item() * xb.size(0)
            total += xb.size(0)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="CI dry-run: synthetic data only, no real images",
    )
    ap.add_argument("--data", default="", help="Path to FER2013 train images (gated)")
    a = ap.parse_args()

    # --dry-run guards before ANY data/model work
    if a.dry_run:
        print("dry-run ok: no real images loaded, no training performed")
        return 0

    # GATE CHECK: real image training is BLOCKED until rights/exposure/adult/
    # duplicate-group-separation evidence is established. FER2013 ZIP has no
    # bundled licence and no subject IDs. See BLOCKED receipt.
    print("BLOCKED: real-image training not authorized for OC-ADDITIONAL-20261019-A.")
    print("  FER2013 ZIP has no bundled licence; no subject IDs available.")
    print("  See outputs/training-oc-additional-20261019-a/BLOCKED.md")
    return 1


if __name__ == "__main__":
    sys.exit(main())
