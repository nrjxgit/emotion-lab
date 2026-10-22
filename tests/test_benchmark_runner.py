"""Tests for emolab.benchmark.runner: identical samples, identical crops,
identical metrics for every adapter."""

import sys
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emolab.inference import FaceEmotionAdapter, ModelResult, ModelMetadata
from emolab.benchmark.runner import run_benchmark


class FixedAdapter(FaceEmotionAdapter):
    """Returns fixed per-class probabilities for every input."""

    name = "fixed-a"
    version = "1.0"
    input_shape = (48, 48, 3)
    native_labels = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
    source = "test"
    licence = "MIT-test"
    calibrated = False

    def __init__(self, probs):
        self._probs = probs
        self.seen = []

    def predict_proba(self, face_rgb: np.ndarray) -> ModelResult:
        self.seen.append(face_rgb.tobytes())
        return ModelResult(
            probabilities={
                c: float(v) for c, v in zip(self.native_labels, self._probs)
            },
            top1=self.native_labels[int(np.argmax(self._probs))],
            confidence=float(max(self._probs)),
            uncertainty=float(1.0 - max(self._probs)),
            latency_ms=1.0,
            metadata=ModelMetadata(
                model_id=self.name,
                version=self.version,
                input_shape=self.input_shape,
                class_mapping={i: l for i, l in enumerate(self.native_labels)},
                calibrated=self.calibrated,
                supported_classes=set(self.native_labels),
                provenance={"source": self.source, "licence": self.licence},
            ),
        )


class AbstainingAdapter(FaceEmotionAdapter):
    """Abstains on every third sample."""

    name = "abstainer"
    version = "1.0"
    input_shape = (48, 48, 3)
    native_labels = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
    source = "test"
    licence = "MIT-test"
    calibrated = False

    def predict_proba(self, face_rgb: np.ndarray):
        if int(np.sum(face_rgb)) % 3 == 0:
            return None
        return ModelResult(
            probabilities={"angry": 1.0},
            top1="angry",
            confidence=1.0,
            uncertainty=0.0,
            latency_ms=1.0,
            metadata=ModelMetadata(
                model_id=self.name,
                version=self.version,
                input_shape=self.input_shape,
                class_mapping={i: l for i, l in enumerate(self.native_labels)},
                calibrated=self.calibrated,
                supported_classes=set(self.native_labels),
                provenance={"source": self.source, "licence": self.licence},
            ),
        )


def test_runner_returns_full_metrics():
    rng = np.random.default_rng(0)
    X = [np.zeros((48, 48, 3), dtype=np.uint8) for _ in range(8)]
    y = [0, 0, 1, 1, 2, 2, 0, 1]
    a = FixedAdapter([0.8, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0])
    r = run_benchmark([a], X, y, dataset="toy")["fixed-a"]
    assert set(r) >= {
        "dataset",
        "n",
        "n_detected",
        "coverage",
        "accuracy",
        "macro_f1",
        "per_class_precision",
        "per_class_recall",
        "confusion",
        "ece",
        "abstention_coverage",
        "selective_accuracy",
        "latency_ms",
    }
    assert r["dataset"] == "toy"
    assert r["n"] == 8
    assert r["n_detected"] == 8
    assert r["coverage"] == 1.0
    # angry(0) predicted for all; correct on indices 0,1,6 => 3/8 = 0.375
    assert abs(r["accuracy"] - 0.375) < 1e-9
    assert abs(r["ece"] - 0.425) < 1e-9  # |0.375 accuracy - 0.8 confidence|


def test_abstention_excluded_from_metrics():
    # sum of image bytes: 0,1,2,3,4,5 -> abstain when sum % 3 == 0 (indices 0,3)
    X = [np.zeros((48, 48, 3), dtype=np.uint8) for _ in range(6)]
    for i, img in enumerate(X):
        img[0, 0, 0] = i
    y = [0, 1, 0, 1, 0, 1]
    a = AbstainingAdapter()
    r = run_benchmark([a], X, y, dataset="toy")["abstainer"]
    assert r["n"] == 6
    assert r["n_detected"] == 4
    assert abs(r["coverage"] - 4 / 6) < 1e-9
    assert abs(r["abstention_coverage"] - 2 / 6) < 1e-9
    # only indices 2 and 4 (y=0) predicted correctly among detected => 0.5
    assert abs(r["accuracy"] - 0.5) < 1e-9


def test_identical_inputs_for_all_adapters():
    seen = []

    def crop_fn(img):
        seen.append(img.tobytes())
        return img

    rng = np.random.default_rng(2)
    X = [rng.integers(0, 256, (48, 48, 3), dtype=np.uint8) for _ in range(4)]
    y = [0, 1, 2, 0]
    a1 = FixedAdapter([1.0, 0, 0, 0, 0, 0, 0])
    a2 = FixedAdapter([0, 1.0, 0, 0, 0, 0, 0])
    a2.name = "fixed-b"
    run_benchmark([a1, a2], X, y, dataset="toy", crop_fn=crop_fn)
    img_bytes = [x.tobytes() for x in X]
    assert seen == img_bytes
    assert a1.seen == a2.seen == img_bytes
    assert len(seen) == 4  # common preprocessing runs once per sample
