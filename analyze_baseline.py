"""Deep-dive on saved baseline proba (no re-inference)."""

import json
import numpy as np

z = np.load("eval_results/fer_test_proba.npz")
proba, y = z["proba"], z["y"]
CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
yp = proba.argmax(axis=1)

print("== confusion (rows=true, cols=pred) ==")
print(" " * 9 + "".join(f"{c[:4]:>6}" for c in CLASSES))
for i, c in enumerate(CLASSES):
    row = [(yp[y == i] == j).sum() for j in range(7)]
    print(f"{c:>9}" + "".join(f"{v:6d}" for v in row))

print("\n== disgust deep-dive ==")
d = proba[:, 1]
print(
    f"mean P(disgust): all={d.mean():.4f} on-disgust-rows={d[y == 1].mean():.4f} max={d.max():.4f}"
)
rank = (proba > d[:, None]).sum(axis=1)
print(
    "rank of disgust prob per true-disgust row:",
    np.bincount(rank[y == 1], minlength=7).tolist(),
)
print(
    "predicted-disgust rows by true class:",
    np.bincount(y[yp == 1], minlength=7).tolist(),
)
print(f"disgust rows where disgust in top-2: {((rank[y == 1]) < 2).mean():.3f}")

print("\n== confidence/calibration ==")
conf = proba.max(axis=1)
print(f"mean confidence={conf.mean():.3f} accuracy={(yp == y).mean():.3f}")
for lo in range(0, 10):
    m = (conf * 10).astype(int) == lo
    if m.sum():
        print(
            f"conf {lo / 10:.1f}-{(lo + 1) / 10:.1f}: n={m.sum():5d} acc={(yp[m] == y[m]).mean():.3f}"
        )
