"""Reproduce the preregistered v0.5 crop study; no training, no test selection.

Run from repo root: python -m training.run_crop_study
Resumable, versioned artifacts in eval_results/v0.5/crops. Existing predictions
are reused only when their fingerprint matches the current protocol and files.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"

import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np

from training.crop_science import (
    CANDIDATES,
    CONTROLS,
    FER7,
    crop_image,
    preprocess_crop,
    metrics,
    select_validation_winner,
)
from backend.app.face.detector import detect_faces, MODEL as YUNET
from backend.app.face.fer_model import HDF5, load_fer, predict_fer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval_results" / "v0.5" / "crops"


def sha(path):
    return hashlib.file_digest(open(path, "rb"), "sha256").hexdigest()


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_rows(split):
    rows = []
    base = ROOT / "eval_data" / "fer2013"
    for i, cls in enumerate(FER7):
        for p in sorted((base / split / cls).glob("*.jpg")):
            image = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (48, 48):
                raise ValueError(f"corrupt/unexpected image: {p}")
            rows.append(
                {
                    "id": p.relative_to(base).as_posix(),
                    "y": i,
                    "pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                    "file_sha256": sha(p),
                    "image": image,
                }
            )
    return rows


def manifest(rows):
    return [{k: v for k, v in r.items() if k != "image"} for r in rows]


def detections(rows, split, fingerprint):
    file = OUT / f"{split}_detections.json"
    if file.exists():
        d = json.loads(file.read_text())
        if d["fingerprint"] != fingerprint:
            raise ValueError(
                "detection manifest mismatch; use a new versioned output directory"
            )
        return d["boxes"]
    boxes = []
    for i, row in enumerate(rows):
        big = cv2.cvtColor(
            cv2.resize(row["image"], (192, 192), interpolation=cv2.INTER_LINEAR),
            cv2.COLOR_GRAY2BGR,
        )
        faces = detect_faces(big)
        box = max(faces, key=lambda b: b["w"] * b["h"]) if faces else None
        if box:
            try:
                crop_image(big, box)
            except ValueError:
                box = None
        boxes.append(box)
        if (i + 1) % 1000 == 0:
            print(split, "detection", i + 1, flush=True)
    json_write(
        file,
        {"fingerprint": fingerprint, "ids": [r["id"] for r in rows], "boxes": boxes},
    )
    return boxes


def evaluate_profiles(rows, boxes, split, model, fingerprint):
    detected = np.asarray([b is not None for b in boxes])
    y = np.asarray([r["y"] for r in rows])
    scores = {}
    for strategy in CANDIDATES + CONTROLS:
        path = OUT / f"{split}_{strategy}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as d:
                if str(d["fingerprint"]) != fingerprint:
                    raise ValueError(f"stale prediction file: {path}")
                p = d["proba"]
        else:
            tensors, idx = [], []
            for i, (r, b) in enumerate(zip(rows, boxes)):
                if b is None:
                    continue
                big = cv2.resize(r["image"], (192, 192), interpolation=cv2.INTER_LINEAR)
                tensors.append(preprocess_crop(big, b, strategy, source=r["image"]))
                idx.append(i)
            p = np.zeros((len(rows), 7))
            for start in range(0, len(tensors), 128):
                batch = np.asarray(tensors[start : start + 128])
                p[np.asarray(idx[start : start + 128])] = np.asarray(
                    model(batch, training=False)
                )
            np.savez_compressed(
                path,
                proba=p,
                y=y,
                detected=detected,
                ids=np.asarray([r["id"] for r in rows]),
                fingerprint=fingerprint,
            )
        scores[strategy] = metrics(y, p, detected)
        print(
            split,
            strategy,
            "accuracy",
            round(scores[strategy]["conditional"]["accuracy"], 5),
            "F1",
            round(scores[strategy]["conditional"]["macro_f1"], 5),
            flush=True,
        )
    json_write(OUT / f"{split}_scores.json", scores)
    return scores


def main():
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    test = load_rows("test")
    test_hashes = {r["pixels_sha256"] for r in test}
    train = load_rows("train")
    eligible = [r for r in train if r["pixels_sha256"] not in test_hashes]
    validation = []
    for c in range(7):
        pool = sorted(
            (r for r in eligible if r["y"] == c),
            key=lambda r: hashlib.sha256(("505:" + r["id"]).encode()).hexdigest(),
        )
        # Also deduplicate validation rows within each class.
        seen = set()
        unique = []
        for r in pool:
            if r["pixels_sha256"] not in seen:
                unique.append(r)
                seen.add(r["pixels_sha256"])
        validation.extend(unique[:100])
    protocol = ROOT / "docs" / "experiments" / "v0.5-crop-protocol.md"
    frozen_protocol = protocol.read_text().split("## Execution ledger")[0]
    evidence = {
        "protocol_sha256": hashlib.sha256(frozen_protocol.encode()).hexdigest(),
        "study_code_sha256": sha(__file__),
        "ops_sha256": sha(ROOT / "training/crop_science.py"),
        "checkpoint_sha256": sha(HDF5),
        "detector_sha256": sha(YUNET),
        "test_manifest": manifest(test),
        "validation_manifest": manifest(validation),
        "excluded_train_exact_test_duplicates": len(train) - len(eligible),
        "engine": {
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
        "subject_split": "unverifiable; FER2013 has no subject IDs",
    }
    fingerprint = hashlib.sha256(
        json.dumps(evidence, sort_keys=True).encode()
    ).hexdigest()
    json_write(OUT / "manifest.json", {**evidence, "fingerprint": fingerprint})
    model = tf.keras.models.load_model(HDF5, compile=False)
    # Independent forward parity for each range, before scoring.
    weights = load_fer()
    parity = {}
    for range_ in ("legacy", "upstream"):
        errors = []
        for r in validation[:3]:
            x = cv2.resize(r["image"], (64, 64)).astype(np.float32) / 255
            if range_ == "upstream":
                x = x * 2 - 1
            x = x[..., None]
            errors.append(
                float(
                    np.max(
                        np.abs(
                            predict_fer(x, weights)
                            - np.asarray(model(x[None], training=False))[0]
                        )
                    )
                )
            )
        parity[range_] = max(errors)
    if max(parity.values()) >= 1e-5:
        raise ValueError(f"forward parity failed: {parity}")
    json_write(OUT / "parity.json", parity)
    vb = detections(validation, "validation", fingerprint)
    validation_scores = evaluate_profiles(
        validation, vb, "validation", model, fingerprint
    )
    winner = select_validation_winner(
        {k: validation_scores[k] for k in CANDIDATES}, "validation"
    )
    lock = OUT / "selection_lock.json"
    selection = {
        "winner": winner,
        "fingerprint": fingerprint,
        "validation_macro_f1": validation_scores[winner]["conditional"]["macro_f1"],
        "candidate_order": list(CANDIDATES),
        "criterion": "validation conditional seven-class macro-F1",
    }
    if lock.exists():
        old = json.loads(lock.read_text())
        if any(old[k] != v for k, v in selection.items()):
            raise ValueError(
                "selection already locked; cannot change after test scoring"
            )
    else:
        json_write(
            lock,
            {
                **selection,
                "locked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
    print("LOCKED WINNER:", winner, flush=True)
    tb = detections(test, "test", fingerprint)
    evaluate_profiles(test, tb, "test", model, fingerprint)
    print("COMPLETE", fingerprint, flush=True)


if __name__ == "__main__":
    main()
