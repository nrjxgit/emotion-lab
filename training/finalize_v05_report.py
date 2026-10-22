"""Build v0.5 report exclusively from saved predictions/manifests.

Requires completed crop, ViT and CREMA runs. No model/crop selection here.
Locks validation-derived temperature and abstention threshold before test scoring.
"""

import hashlib
import json
from pathlib import Path
import numpy as np
from training.calibrate_temp import fit_temperature, apply_temperature, abstention_curve
from training.crop_science import metrics, CANDIDATES, CONTROLS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval_results/v0.5"
CROP = OUT / "crops"


def read(path):
    return json.loads(path.read_text())


def calibrate_selected():
    selection = read(CROP / "selection_lock.json")
    name = selection["winner"]
    v = np.load(CROP / f"validation_{name}.npz")
    mask = v["detected"]
    vp, vy = v["proba"][mask], v["y"][mask]
    T = fit_temperature(np.log(np.clip(vp, 1e-12, 1)), vy)
    cv = apply_temperature(np.log(np.clip(vp, 1e-12, 1)), T)
    thresholds = [0.0, 0.5, 0.7, 0.9, 0.95]
    curves = abstention_curve(cv, vy, thresholds)
    choices = [r for r in curves if r["accuracy"] >= 0.8 and r["n"] >= 30]
    threshold = (
        min(choices, key=lambda r: r["threshold"])["threshold"] if choices else None
    )
    locked = {
        "strategy": name,
        "temperature": T,
        "threshold": threshold,
        "rule": "lowest predeclared threshold with validation accuracy>=.8 and n>=30",
        "fit_n": len(vy),
        "fit_file_sha256": hashlib.file_digest(
            open(CROP / f"validation_{name}.npz", "rb"), "sha256"
        ).hexdigest(),
        "validation_curve": curves,
    }
    lockfile = CROP / "calibration_lock.json"
    if lockfile.exists() and read(lockfile) != locked:
        raise ValueError("calibration already locked; refusing test-guided changes")
    lockfile.write_text(json.dumps(locked, indent=2))
    # Test labels first accessed only after lock exists.
    test = np.load(CROP / f"test_{name}.npz")
    mask = test["detected"]
    p = test["proba"]
    y = test["y"]
    pc = p.copy()
    pc[mask] = apply_temperature(np.log(np.clip(p[mask], 1e-12, 1)), T)
    accepted = (
        mask & (pc.max(1) >= threshold)
        if threshold is not None
        else np.zeros(len(y), bool)
    )
    result = {
        "lock": locked,
        "uncalibrated": metrics(y, p, mask),
        "calibrated": metrics(y, pc, mask),
        "selected_abstention": metrics(y, pc, accepted),
        "limitations": "validation is FER train-derived; pretrained data membership unverified; not independent calibration",
    }
    (CROP / "calibration_result.json").write_text(json.dumps(result, indent=2))
    return result


def cluster_interval(clip_rows, model, label, n=1000):
    rows = [r for r in clip_rows if r[label] is not None]
    actors = sorted({r["actor"] for r in rows})
    scores = {
        a: [
            int(r["detected"] and np.argmax(r["probabilities"][model]) == r[label])
            for r in rows
            if r["actor"] == a
        ]
        for a in actors
    }
    rng = np.random.default_rng(505)
    estimates = []
    for _ in range(n):
        sample = np.concatenate(
            [scores[a] for a in rng.choice(actors, len(actors), replace=True)]
        )
        estimates.append(float(sample.mean()))
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def main():
    selection = read(CROP / "selection_lock.json")
    validation = read(CROP / "validation_scores.json")
    test = read(CROP / "test_scores.json")
    full = read(OUT / "full_models.json")
    crema = read(OUT / "cremad/scores.json")
    manifest = read(CROP / "manifest.json")
    cal = calibrate_selected()
    winner = selection["winner"]
    same_ids = None
    for name in CANDIDATES + CONTROLS:
        d = np.load(CROP / f"test_{name}.npz")
        ids = d["ids"][d["detected"]]
        if same_ids is not None and not np.array_equal(ids, same_ids):
            raise ValueError(
                "crop strategies evaluated on different detected subjects/images"
            )
        same_ids = ids
    clip_rows = [read(p) for p in sorted((OUT / "cremad/clips").glob("*.json"))]
    intervals = {
        m: {
            label: cluster_interval(clip_rows, m, label)
            for label in ("intended", "face_vote")
        }
        for m in ("mini_legacy", "mini_upstream", "vit")
    }
    (OUT / "cremad/actor_bootstrap_ci.json").write_text(json.dumps(intervals, indent=2))

    lines = [
        "# v0.5: preprocessing audit and model verification",
        "",
        "## Scientific status",
        "",
        "The prior 7.0-point aggregate crop drop was not a controlled causal attribution. "
        "This report controls sample membership, input normalization, interpolation, geometry and resize path. "
        "Legacy results are retained; no default model or preprocessing was silently replaced.",
        "",
        "**FER is not independent validation of either pretrained checkpoint.** "
        "The ViT author notebook pools folders and oversamples before splitting; mini-Xception historical "
        "checkpoint selection may reuse the combined FER test set. Exact historical membership is unavailable. "
        "See `docs/VIT_FORENSICS.md` and archived primary sources.",
        "",
        "## 1. Locked protocol and data",
        "",
        "Original ZIP unchanged; FER folders contain 28,709 train and 7,178 test images. "
        "Crop validation: 100 unique images/class from train, hash seed505, excluding any exact "
        "decoded-pixel duplicate in test. No fitting/selection on test. No subject IDs: "
        "participant-level split guarantees cannot be made for FER or unknown pretrained corpora.",
        "",
        f"Train images excluded as exact test duplicates: {manifest['excluded_train_exact_test_duplicates']}. "
        f"Winner **{winner}**, locked at {selection['locked_utc']} using validation macro-F1 "
        f"{selection['validation_macro_f1']:.4f}. All test variants were predeclared diagnostics; "
        "no reselection after viewing them.",
        "",
        "Protocol: `docs/experiments/v0.5-crop-protocol.md`; hashes, filenames, "
        "pixel digests, detector boxes and per-image predictions: `eval_results/v0.5/crops/`.",
        "",
        "## 2. Crop comparison: identical detected test images",
        "",
        "| Profile | Val F1 | Detected-test accuracy | Detected-test F1 | End-to-end accuracy | End-to-end F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in CANDIDATES + CONTROLS:
        s = test[name]
        c, e = s["conditional"], s["end_to_end"]
        lines.append(
            f"| {name}{' **selected**' if name == winner else ''} | "
            f"{validation[name]['conditional']['macro_f1']:.4f} | {c['accuracy']:.4f} | "
            f"{c['macro_f1']:.4f} | {e['accuracy']:.4f} | {e['macro_f1']:.4f} |"
        )
    n, det = test[winner]["n"], test[winner]["detected"]
    lines.extend(
        [
            "",
            f"Common cohort: **{det}/{n}** detected. Every profile above uses exactly those "
            "same image IDs. Misses count as wrong/false negatives in end-to-end metrics; "
            "confusion matrices include an eighth abstention column. Controls use this common "
            "cohort for fair diagnostics, not an independent detector-free end-to-end path.",
            "",
            "| Profile | Angry recall | Disgust | Fear | Happy | Sad | Surprise | Neutral |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in CANDIDATES + CONTROLS:
        lines.append(
            "| "
            + name
            + " | "
            + " | ".join(f"{v:.4f}" for v in test[name]["conditional"]["recall"])
            + " |"
        )

    def acc(name):
        return test[name]["conditional"]["accuracy"]

    lines.extend(
        [
            "",
            "### Controlled diagnostic contrasts (percentage points)",
            "",
            f"- Geometry/extra-resize cost at legacy range: tight_legacy minus full_legacy = {(acc('tight_legacy') - acc('full_legacy')) * 100:+.2f} pp.",
            f"- Normalization only on tight AREA crop: tight_area minus tight_legacy = {(acc('tight_area') - acc('tight_legacy')) * 100:+.2f} pp.",
            f"- Final interpolation only (LINEAR vs AREA, corrected range): {(acc('tight_linear') - acc('tight_area')) * 100:+.2f} pp.",
            f"- 10% margin vs tight (LINEAR, corrected range): {(acc('margin10') - acc('tight_linear')) * 100:+.2f} pp.",
            f"- Square aspect vs rectangle: {(acc('square') - acc('tight_linear')) * 100:+.2f} pp.",
            f"- Eye rotation vs unaligned square: {(acc('aligned_square') - acc('square')) * 100:+.2f} pp.",
            f"- Source48 one-resize vs 192 intermediate: {(acc('source48') - acc('tight_linear')) * 100:+.2f} pp (also boundary rounding changes).",
            f"- Full-image resize roundtrip cost: {(acc('full_roundtrip') - acc('full_linear')) * 100:+.2f} pp.",
            "",
            "These are frozen-pipeline paired diagnostics, not biological causality or a license "
            "to select a new method on this test set. FER thumbnails were already cropped: "
            "upsampling and redetection cannot recreate a full webcam scene.",
            "",
            "### Where the crop drop comes from (measured box geometry)",
            "",
        ]
    )
    geo = json.loads((CROP / "geometry_diagnostics.json").read_text())
    n_det = geo["n_detected"]
    lines.extend(
        [
            f"On the {n_det} detected test images, YuNet boxes (192px-upscaled thumbnails) are "
            f"large and frequently overflow the frame: {geo['boxes_outside_image']}/{n_det} boxes "
            f"({geo['boxes_outside_image'] / n_det:.1%}) extend past "
            "an image edge and are clamped before cropping. Mean retained frame fraction: "
            f"tight {geo['tight']['mean_fraction_original_frame']:.3f}, "
            f"margin10 {geo['margin10']['mean_fraction_original_frame']:.3f}, "
            f"margin25 {geo['margin25']['mean_fraction_original_frame']:.3f} "
            f"(median {geo['margin25']['median']:.1f}; "
            f"{geo['margin25']['fraction_exact_full_frame']:.1%} of margin25 crops "
            "are the entire frame), "
            f"square {geo['square']['mean_fraction_original_frame']:.3f}. Raw numbers: "
            "`crops/geometry_diagnostics.json`. ",
            "Interpretation, in order of measured size: (1) input range dominates — the legacy "
            "[0,1] path discards the trained [-1,1] operating point (+14.93 pp tight_area vs "
            "tight_legacy on identical boxes); (2) tight YuNet crops discard ~24% of frame "
            "pixels on already-cropped thumbnails while overflowing boxes get edge-clamped, so "
            "'margin' comparisons are mostly full-frame-vs-tight comparisons, not evidence about "
            "real webcam framing; (3) final interpolation is negligible (+0.21 pp LINEAR vs AREA); "
            "eye-rotation alignment slightly hurts (-0.40 pp) on 5-point landmarks. The original "
            "7.1 pp aggregate figure mixed all of these effects together and must not be cited as "
            "a pure alignment/margin cost.",
            "",
        ]
    )
    lines.extend(
        [
            "### Training normalization correction",
            "",
        ]
    )
    lines.extend(
        [
            "Upstream mini-Xception training/demo uses `preprocess_input(..., True)` = "
            "gray/255 then subtract .5 and multiply 2. Our historical [0,1] input was incorrect "
            "relative to that reference. TensorFlow parity on equal tensors did not test the "
            "range. Batched Keras scoring here is parity-checked against NumPy for both ranges; "
            "maximum errors are saved in `crops/parity.json`. Serving legacy default retained "
            "for reproducibility; corrected profiles remain explicit research options.",
            "",
            "## 3. Full FER model comparison",
            "",
            "| Model/profile | N | Accuracy | Macro-F1 | ECE (10 bins) | NLL |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("mini_legacy", "vit"):
        s = full[name]["conditional"]
        lines.append(
            f"| {name} | {s['n']} | {s['accuracy']:.4f} | {s['macro_f1']:.4f} | {s['ece']:.4f} | {s['nll']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Same 7,178 IDs and fixed metrics/class order. Native preprocessing is model-specific "
            "(mini gray64; ViT RGB224 bilinear /255 mean/std=.5), not forced to share incompatible tensors. "
            "ViT head mapped explicitly; revision `747cf16692eea925b54b7b543cf436848128b68d`, "
            "Apache-2.0, base google/vit-base-patch16-224-in21k. Exact weights/config hashes "
            "and full metrics in `sources/provenance.json` and `full_models.json`. "
            "Pinned-checkpoint spot parity verifies the saved full-run scores. "
            "**This is not evidence of independent ViT superiority.**",
            "",
            "## 4. CREMA-D visual-only external pilot",
            "",
            "Official CREMA-D revision `1658cd342dff90010aa843eaeebd53610a08b1dc`; "
            "ODbL1.0 + DbCL1.0. Public Git-LFS media acquired; no authentication bypass. "
            "24 actors hash-selected before scores, DFA sentence x six emotions, "
            "three fixed quarter/half/three-quarter frames, same detected .10-margin crops. "
            "Clip score = mean available-frame probabilities; no detected frames => abstention. "
            "All selected actors evaluation-only: no CREMA training/calibration/tuning. "
            "Known actors prevent within-study leakage; historical pretraining membership remains unknown. "
            "Surprise has no target examples: surprise predictions are errors, never renormalized away. "
            "Filename emotion is an acted instruction, not internal feeling. Secondary target: "
            "visual-only crowd FaceVote; ties/missing votes explicitly excluded.",
            "",
            "| Model | Intended N | Accuracy (actor-bootstrap 95% CI) | F1 six | ECE | FaceVote N | Agreement | F1 six |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("mini_legacy", "mini_upstream", "vit"):
        s = crema[name]
        t, v = s["intended"], s["visual_majority"]
        ci = intervals[name]["intended"]
        lines.append(
            f"| {name} | {t['n']} | {t['accuracy']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] | "
            f"{t['macro_f1_six']:.4f} | {t['conditional']['ece']:.4f} | {v['n']} | "
            f"{v['accuracy']:.4f} | {v['macro_f1_six']:.4f} |"
        )
    lines.extend(
        [
            "",
            "This is a 24-actor/one-sentence pilot, not all 7,442 clips or population-wide "
            "validation. Actor-bootstrap uses seed505/1,000 draws, not frame-wise confidence inflation. "
            "Per-class, per-actor, confusion and failure metadata are preserved in "
            "`cremad/scores.json`, `cremad/clips/` and `cremad/actor_bootstrap_ci.json`. "
            "No demographic fairness/infant/fusion claims follow from this sample.",
            "",
            "## 5. Separate-validation calibration and abstention",
            "",
            f"For the locked crop {winner}, validation fit n={cal['lock']['fit_n']}, T={cal['lock']['temperature']:.4f}. "
            f"Test ECE {cal['uncalibrated']['conditional']['ece']:.4f} -> {cal['calibrated']['conditional']['ece']:.4f}; "
            "temperature leaves argmax unchanged. Validation-selected threshold: "
            f"{cal['lock']['threshold']}; test coverage {cal['selected_abstention']['coverage']:.4f}, "
            f"conditional accepted accuracy {cal['selected_abstention']['conditional']['accuracy']}, "
            f"end-to-end accuracy {cal['selected_abstention']['end_to_end']['accuracy']:.4f}. "
            "Selection rule/validation curve/locks saved before test use. These remain FER-derived "
            "explorations, not independent calibration certificates.",
            "",
            "## 6. Application/reproducibility/limitations",
            "",
            "- Default mini-Xception and legacy API behavior preserved; opt-in ViT via "
            "`POST /api/infer?model=vit`, `GET /api/models`. HSEmotion remains an explicit "
            "unimplemented adapter; Claude app not merged.",
            "- Body remains motion-only; no trained fusion improvement established.",
            "- No further physical camera capture was performed; consented participant protocol "
            "remains required. Earlier dark-camera probe is not live-person validation.",
            "- FER lacks subject/culture/age/skin-tone metadata. Exact duplicate exclusions "
            "are not a substitute for participant-independent splits.",
            "- ZIP has no licence file; user-supplied FER source/licence chain is not independently "
            "established. Local research analysis only; no image redistribution.",
            "- Reproduce from repository root: `python -m training.snapshot_v05_sources`; "
            "`python -m training.run_crop_study`; `python run_vit_full.py` (historical full run; "
            "new adapters pin revision); `python -m training.score_vit_full`; "
            "`python -m training.run_cremad_pilot`; `python -m training.finalize_v05_report`.",
            "- Source datasets/checkpoints kept local; arrays/manifests record sample IDs, "
            "hashes, selection locks and exact profiles. Do not reinterpret legacy Unknown "
            "mass or `1-max(prob)` as a scientifically calibrated internal mood probability.",
            "",
        ]
    )
    report = ROOT / "eval_results/v0.5-report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("Report generated:", report, flush=True)


if __name__ == "__main__":
    main()
