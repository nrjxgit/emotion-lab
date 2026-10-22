import sys, json
import numpy as np

sys.path.insert(0, "backend/app/face")
from expression import ExpressionPredictor

d = np.load("eval_results/compare_subset.npz")
X, y = d["X"], d["y"]
out = {}
for profile in ("legacy", "corrected"):
    p = ExpressionPredictor(profile=profile)
    proba = np.array(
        [list(p.predict(np.stack([im] * 3, -1), [])["probs"].values()) for im in X]
    )
    # 8-class probs incl unknown; map to 7 by dropping unknown then argmax over FER order
    order = [
        "happiness",
        "sadness",
        "anger",
        "surprise",
        "fear",
        "disgust",
        "neutral",
        "unknown",
    ]
    fer = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    back = {
        "happiness": "happy",
        "sadness": "sad",
        "anger": "angry",
        "surprise": "surprise",
        "fear": "fear",
        "disgust": "disgust",
        "neutral": "neutral",
    }
    pred = np.array(
        [
            ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"].index(
                back[max(order[:7], key=lambda c: proba[i][order.index(c)])]
            )
            for i in range(len(X))
        ]
    )
    out[profile] = {"acc": float((pred == y).mean())}
    out[profile]["pred"] = pred.tolist()
a = np.array(out["legacy"]["pred"])
b = np.array(out["corrected"]["pred"])
out["agreement"] = float((a == b).mean())
out["n"] = len(X)
json.dump(out, open("eval_results/v0.6/old_vs_new.json", "w"), indent=1)
print(
    "legacy acc:",
    round(out["legacy"]["acc"], 4),
    "corrected acc:",
    round(out["corrected"]["acc"], 4),
    "agreement:",
    round(out["agreement"], 4),
)
