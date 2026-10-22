"""T + abstention threshold from VALIDATION margin25; confirmed on TEST.

No test labels used for fitting. Rerunnable from saved v0.5 arrays.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "training")
from calibrate_temp import (
    apply_temperature,
    abstention_curve,
    ece_from_proba,
    fit_temperature,
)

CROP = Path("eval_results/v0.5/crops")
v = np.load(CROP / "validation_margin25.npz")
t = np.load(CROP / "test_margin25.npz")
vm, ym = v["detected"], v["y"]
tm, yt = t["detected"], t["y"]
T = fit_temperature(np.log(np.clip(v["proba"][vm], 1e-12, 1)), ym[vm])
curves = abstention_curve(
    apply_temperature(np.log(np.clip(v["proba"][vm], 1e-12, 1)), T),
    ym[vm],
    [0.0, 0.5, 0.7, 0.9, 0.95],
)
cands = [c for c in curves if c["accuracy"] >= 0.8 and c["n"] >= 30]
thr = min(cands, key=lambda c: c["threshold"])["threshold"]
P = apply_temperature(np.log(np.clip(t["proba"][tm], 1e-12, 1)), T)
out = {
    "T": T,
    "threshold": thr,
    "fit": "validation_margin25",
    "ece_test_uncal": ece_from_proba(t["proba"][tm], yt[tm]),
    "ece_test_cal": ece_from_proba(P, yt[tm]),
    "abstention_test": abstention_curve(P, yt[tm], [0.0, 0.5, 0.7, 0.9, 0.95]),
}
Path("eval_results/v0.6").mkdir(parents=True, exist_ok=True)
Path("eval_results/v0.6/calibration_validation.json").write_text(
    json.dumps(out, indent=2)
)
print("T =", round(T, 4), "thr =", thr)
print("test ECE", round(out["ece_test_uncal"], 4), "->", round(out["ece_test_cal"], 4))
for c in out["abstention_test"]:
    print(
        f"thr={c['threshold']}: cov={c['coverage']:.3f} acc={c['accuracy']:.3f} n={c['n']}"
    )
