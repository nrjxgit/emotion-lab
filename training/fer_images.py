"""FER2013 image-folder loader (Kaggle FER-2013 images: train/<cls>/*.jpg).

Order matches oarriaga/FER2013: angry, disgust, fear, happy, sad, surprise, neutral.
Test split is NEVER used for training: helpers only read; assert_disjoint guards mixes.
"""

import glob
import os

import cv2
import numpy as np

CLASSES7 = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def load_split(split_dir: str, max_per_class: int | None = None):
    X, y, files = [], [], []
    for ci, cls in enumerate(CLASSES7):
        paths = sorted(glob.glob(os.path.join(split_dir, cls, "*.jpg")))
        if max_per_class:
            paths = paths[:max_per_class]
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (48, 48):
                img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
            X.append(img)
            y.append(ci)
            files.append(os.path.basename(p))
    if not X:
        raise ValueError(f"no images under {split_dir}")
    return np.stack(X), np.array(y), files


def assert_disjoint(train_files: list, test_files: list):
    overlap = set(train_files) & set(test_files)
    if overlap:
        raise ValueError(
            f"train/test filename overlap ({len(overlap)}): test split contaminated"
        )
