"""CREMA-D expansion: all 91 actors x DFA x 6 emotions, validated margin25 crops.

Per-frame probs saved for all models (temporal comparison: middle vs mean-3).
T=1.045 cross-dataset ECE (no fitting on CREMA-D). Slow (~15 min); unattended.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
import csv
import hashlib
import io
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from backend.app.face.detector import detect_faces
from backend.app.face.adapters import ViTAdapter
from training.crop_science import preprocess_crop
from training.cremad_science import parse_filename, parse_face_vote, score_six

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "cremad_repo"
OUT = ROOT / "eval_results" / "v0.7" / "cremad"
REV = "1658cd342dff90010aa843eaeebd53610a08b1dc"


def main():
    import torch
    from backend.app.face.fer_model import load_fer, predict_fer

    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/{REV}/processedResults/summaryTable.csv"
    text = (
        urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "audit"}), timeout=120
        )
        .read()
        .decode("utf-8-sig")
    )
    votes = {
        r["FileName"].rsplit(".", 1)[0]: r for r in csv.DictReader(io.StringIO(text))
    }
    fs = sorted(
        p
        for p in (DATA / "VideoFlash").glob("*_DFA_*_XX.flv")
        if p.stat().st_size > 1000
    )
    actors = sorted({p.name[:4] for p in fs})
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "git_revision": REV,
                "n_actors": len(actors),
                "n_clips": len(fs),
                "actors": actors,
                "clips": [p.name for p in fs],
                "crop": "validated margin25/LINEAR/[-1,1] for mini; same crop for ViT",
                "frames": [0.25, 0.5, 0.75],
                "aggregation": ["middle", "mean3"],
                "training": "none on CREMA-D",
                "temperature_note": "T=1.045 applied post-hoc for cross-dataset ECE only, never fit here",
            },
            indent=2,
        )
    )
    vit = ViTAdapter()
    w = load_fer()
    rows = []
    for i, path in enumerate(fs):
        cache = OUT / "clips" / (path.stem + ".json")
        if cache.exists():
            rows.append(json.loads(cache.read_text()))
            continue
        actor, y = parse_filename(path.name)
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frames = []
        for fraction in (0.25, 0.5, 0.75):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * fraction))
            ok, frame = cap.read()
            if not ok:
                continue
            faces = detect_faces(frame)
            if not faces:
                continue
            box = max(faces, key=lambda b: b["w"] * b["h"])
            try:
                t = preprocess_crop(frame, box, "margin25")
            except ValueError:
                continue
            pm = predict_fer(t, w)
            H, W, _ = frame.shape
            x0, y0 = max(0, box["x"]), max(0, box["y"])
            x1, y1 = min(W, box["x"] + box["w"]), min(H, box["y"] + box["h"])
            if x1 <= x0 or y1 <= y0:
                continue
            frames.append(
                {
                    "mini": [float(v) for v in pm],
                    "vit": [float(v) for v in vit.predict_proba(frame[y0:y1, x0:x1])],
                }
            )
        cap.release()
        r = {
            "file": path.name,
            "actor": actor,
            "intended": y,
            "face_vote": parse_face_vote(votes.get(path.stem, {}).get("FaceVote", "")),
            "n_frames": len(frames),
            "frames": frames,
            "detected": bool(frames),
        }
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(r))
        rows.append(r)
        if (i + 1) % 50 == 0:
            print(f"CREMA-D full {i + 1}/{len(fs)}", flush=True)

    def agg(rows, model, how):
        P, Y, M = [], [], []
        for r in rows:
            if not r["detected"]:
                M.append(False)
                P.append([0.0] * 7)
                Y.append(r["intended"])
                continue
            M.append(True)
            Y.append(r["intended"])
            probs = [f[model] for f in r["frames"]]
            mid = probs[len(probs) // 2]
            P.append(list(np.mean(probs, axis=0)) if how == "mean3" else mid)
        return score_six(Y, np.array(P), np.array(M))

    out = {}
    for model in ("mini", "vit"):
        out[model] = {how: agg(rows, model, how) for how in ("middle", "mean3")}
    # cross-dataset temperature (post-hoc, no fitting)
    from training.calibrate_temp import ece_from_proba, apply_temperature

    pm = np.array(
        [
            r["frames"][len(r["frames"]) // 2]["mini"] if r["detected"] else [0.0] * 7
            for r in rows
        ]
    )
    ym = np.array([r["intended"] for r in rows])
    mm = np.array([r["detected"] for r in rows])
    pc = apply_temperature(np.log(np.clip(pm[mm], 1e-12, 1)), 1.045)
    out["cross_dataset_T"] = {
        "ece_before": ece_from_proba(pm[mm], ym[mm]),
        "ece_after": ece_from_proba(pc, ym[mm]),
    }
    (OUT / "scores.json").write_text(json.dumps(out, indent=2, allow_nan=False))
    print(
        "COMPLETE",
        {
            m: {h: round(out[m][h]["accuracy"], 4) for h in ("middle", "mean3")}
            for m in ("mini", "vit")
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
