import sys, time
import numpy as np

sys.path.insert(0, "backend/app/face")
sys.path.insert(0, "training")
from detector import detect_faces
from fer_images import load_split
from fer_model import load_fer, preprocess, predict_fer

X, _, _ = load_split("eval_data/fer2013/test")
w = load_fer()
rng = np.random.default_rng(0)
idx = rng.choice(len(X), 20, replace=False)
det_t, fer_t = [], []
for i in idx:
    img = np.stack([X[i]] * 3, -1)
    big = np.stack([np.repeat(np.repeat(X[i], 4, 0), 4, 1)] * 3, -1)
    t0 = time.time()
    faces = detect_faces(big)
    det_t.append((time.time() - t0) * 1000)
    if faces:
        b = faces[0]
        H, W, _ = big.shape
        x0, y0 = max(0, b["x"]), max(0, b["y"])
        x1, y1 = min(W, b["x"] + b["w"]), min(H, b["y"] + b["h"])
        if x1 <= x0 or y1 <= y0:
            continue
        crop = big[y0:y1, x0:x1]
        t0 = time.time()
        predict_fer(preprocess(crop), w)
        fer_t.append((time.time() - t0) * 1000)
print(
    f"detect: n={len(det_t)} p50={np.median(det_t):.1f}ms p95={np.percentile(det_t, 95):.1f}ms"
)
print(
    f"fer: n={len(fer_t)} p50={np.median(fer_t):.1f}ms p95={np.percentile(fer_t, 95):.1f}ms"
)
print(f"faces found: {len(fer_t)}/20")
