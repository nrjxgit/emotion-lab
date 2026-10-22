"""Brightness-quartile slice (lighting proxy; FER2013 has no lighting labels)."""

import sys
import numpy as np

sys.path.insert(0, "training")
from fer_images import load_split

z = np.load("eval_results/fer_test_proba.npz")
proba, y = z["proba"], z["y"]
X, _, _ = load_split("eval_data/fer2013/test")
assert X.shape[0] == len(y)
bright = X.mean(axis=(1, 2))
yp = proba.argmax(axis=1)
qs = np.quantile(bright, [0, 0.25, 0.5, 0.75, 1.0])
print("brightness quartile boundaries:", np.round(qs, 1).tolist())
for q in range(4):
    m = (bright >= qs[q]) & (bright <= qs[q + 1])
    print(
        f"Q{q} (n={m.sum()}): acc={(yp[m] == y[m]).mean():.3f} meanconf={proba[m].max(axis=1).mean():.3f}"
    )
