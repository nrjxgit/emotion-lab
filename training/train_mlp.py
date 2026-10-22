"""Real MLP training on FER2013 pixels (when CSV supplied) + labeled synthetic smoke for CI."""

import argparse
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib


def train(X: np.ndarray, y: np.ndarray, out: str) -> dict:
    n = X.shape[0]
    Xs = X.reshape(n, -1).astype(np.float32) / 255.0
    clf = make_pipeline(
        StandardScaler(with_mean=False),
        MLPClassifier(hidden_layer_sizes=(128,), max_iter=30, random_state=0),
    )
    clf.fit(Xs, y)
    acc = float(clf.score(Xs, y))
    joblib.dump(clf, out)
    return {"train_acc": acc, "n": n, "synthetic": False}


def run_synthetic(out: str, n: int = 60) -> dict:
    rng = np.random.default_rng(0)
    X = (rng.random((n, 48, 48)) * 255).astype(np.uint8)
    y = (X.mean(axis=(1, 2)) > 127).astype(int)
    r = train(X, y, out)
    r["synthetic"] = True
    r["warning"] = (
        "synthetic blobs only: tests pipeline mechanics, NOT emotion recognition"
    )
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="")
    ap.add_argument("--out", default="models/mlp_fer2013.joblib")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--max-n", type=int, default=5000)
    a = ap.parse_args()
    if a.synthetic or not a.csv:
        from fer_csv import load_fer_csv  # noqa (import check)

        if not a.csv:
            print(run_synthetic(a.out))
            return
    from fer_csv import load_fer_csv

    X, y = load_fer_csv(a.csv, usage="Training", max_n=a.max_n)
    print(train(X, y, a.out))


if __name__ == "__main__":
    main()
