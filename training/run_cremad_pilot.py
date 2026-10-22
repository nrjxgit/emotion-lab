"""Frozen visual-only CREMA-D pilot: 24 hash-selected actors x DFA x 6 emotions.

No training/calibration on CREMA-D. Fixed quarter/half/three-quarter frames, same
YuNet crops for every model. Filename intention AND visual crowd-majority scores.
ODbL/DbCL; keep video data local and publish only metrics/manifests.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
import csv
import hashlib
import json
from pathlib import Path
import urllib.request
import cv2
import numpy as np

from backend.app.face.detector import detect_faces
from backend.app.face.fer_model import load_fer, predict_fer, preprocess
from backend.app.face.adapters import ViTAdapter
from training.crop_science import crop_image
from training.cremad_science import parse_filename, parse_face_vote, score_six

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "cremad_repo"
OUT = ROOT / "eval_results/v0.5/cremad"
REV = "1658cd342dff90010aa843eaeebd53610a08b1dc"
VIT_REV = "747cf16692eea925b54b7b543cf436848128b68d"


def download_text(relative):
    url = f"https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/{REV}/{relative}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "expression-research-audit"}
    )
    return urllib.request.urlopen(req, timeout=60).read().decode("utf-8-sig")


def main():
    import torch

    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    text = download_text("processedResults/summaryTable.csv")
    (OUT / "LICENSE.txt").write_text(download_text("LICENSE.txt"))
    import io

    from training.cremad_science import read_summary

    votes = read_summary(text)
    fs = sorted(
        p
        for p in (DATA / "VideoFlash").glob("*_DFA_*_XX.flv")
        if p.stat().st_size > 1000
    )
    actors = sorted(
        {p.name[:4] for p in fs},
        key=lambda a: hashlib.sha256(("505:" + a).encode()).hexdigest(),
    )[:24]
    fs = [p for p in fs if p.name[:4] in actors]
    manifest = {
        "git_revision": REV,
        "actors": actors,
        "clips": [p.name for p in fs],
        "video_sha256": {
            p.name: hashlib.file_digest(open(p, "rb"), "sha256").hexdigest() for p in fs
        },
        "selection": "hash(seed505,actor) first24; DFA utterance all6; frames .25,.5,.75",
        "crop": "clamped rectangle + .10 per-side margin (frozen independently of CREMA labels)",
        "frame_aggregation": "mean probability of available detected frames; zero frames => abstention",
        "training": "none on CREMA-D; all selected actors evaluation-only",
        "labels": "filename intended emotion; secondary FaceVote single-majority, ties excluded explicitly",
        "vit_revision": VIT_REV,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    vit = ViTAdapter()
    vit.revision = VIT_REV
    w = load_fer()
    rows = []
    for i, path in enumerate(fs):
        cache = OUT / "clips" / (path.stem + ".json")
        if cache.exists():
            rows.append(json.loads(cache.read_text()))
            continue
        actor, y = parse_filename(path.name)
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        predictions = {name: [] for name in ("mini_legacy", "mini_upstream", "vit")}
        frame_info = []
        for fraction in (0.25, 0.5, 0.75):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * fraction))
            ok, frame = cap.read()
            if not ok:
                frame_info.append({"fraction": fraction, "error": "decode"})
                continue
            faces = detect_faces(frame)
            if not faces:
                frame_info.append({"fraction": fraction, "error": "no_face"})
                continue
            box = max(faces, key=lambda b: b["w"] * b["h"])
            try:
                crop = crop_image(frame, box, margin=0.1)
            except ValueError:
                frame_info.append({"fraction": fraction, "error": "invalid_box"})
                continue
            x = preprocess(crop)
            predictions["mini_legacy"].append(predict_fer(x, w).tolist())
            predictions["mini_upstream"].append(predict_fer(x * 2 - 1, w).tolist())
            predictions["vit"].append(vit.predict_proba(crop).tolist())
            frame_info.append(
                {"fraction": fraction, "box": box, "crop_shape": list(crop.shape)}
            )
        cap.release()
        r = {
            "file": path.name,
            "actor": actor,
            "intended": y,
            "face_vote": parse_face_vote(votes.get(path.stem, {}).get("FaceVote", "")),
            "frame_info": frame_info,
            "probabilities": {
                name: np.mean(p, axis=0).tolist() if p else [0.0] * 7
                for name, p in predictions.items()
            },
            "detected": bool(predictions["mini_legacy"]),
        }
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(r))
        rows.append(r)
        if (i + 1) % 12 == 0:
            print("CREMA-D", i + 1, "/", len(fs), flush=True)
    scores = {}
    for name in ("mini_legacy", "mini_upstream", "vit"):
        p = np.asarray([r["probabilities"][name] for r in rows])
        mask = np.asarray([r["detected"] for r in rows])
        y = [r["intended"] for r in rows]
        intended = score_six(y, p, mask)
        visual_idx = [i for i, r in enumerate(rows) if r["face_vote"] is not None]
        visual = score_six(
            [rows[i]["face_vote"] for i in visual_idx], p[visual_idx], mask[visual_idx]
        )
        scores[name] = {
            "intended": intended,
            "visual_majority": visual,
            "visual_ties_or_missing": len(rows) - len(visual_idx),
            "per_actor": {
                a: score_six(
                    [r["intended"] for r in rows if r["actor"] == a],
                    p[[i for i, r in enumerate(rows) if r["actor"] == a]],
                    mask[[i for i, r in enumerate(rows) if r["actor"] == a]],
                )
                for a in actors
            },
        }
    (OUT / "scores.json").write_text(json.dumps(scores, indent=2, allow_nan=False))
    print(
        "COMPLETE CREMA-D",
        {k: v["intended"]["accuracy"] for k, v in scores.items()},
        flush=True,
    )


if __name__ == "__main__":
    main()
