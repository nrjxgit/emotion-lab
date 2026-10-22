"""emolab.inference adapter contract tests.

These tests define the contract that every facial-expression model adapter
must satisfy, regardless of where its weights came from. No external
repositories or weights are required: adapters are exercised with synthetic
face crops so the suite runs offline.
"""

import sys
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emolab.inference import (
    FaceEmotionAdapter,
    ModelResult,
    ModelMetadata,
    register_adapter,
    get_adapter,
    list_adapters,
)

FER7 = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class DummyAdapter(FaceEmotionAdapter):
    """Adapter returning a fixed distribution over its own 3-class label set."""

    name = "dummy-test"
    version = "0.1-test"
    input_shape = (48, 48, 3)
    native_labels = ["neutral", "happy", "sad"]
    source = "test-double"
    licence = "MIT-test"
    calibrated = False

    def predict_proba(self, face_rgb: np.ndarray) -> ModelResult:
        face_rgb = np.asarray(face_rgb)
        if face_rgb.ndim != 3 or face_rgb.shape[:2] != self.input_shape[:2]:
            raise ValueError(f"expected {self.input_shape[:2]} input")
        return ModelResult(
            probabilities={"neutral": 0.2, "happy": 0.5, "sad": 0.3},
            top1="happy",
            confidence=0.5,
            uncertainty=0.5,
            latency_ms=12.0,
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


def test_adapter_contract_returns_typed_result():
    a = DummyAdapter()
    img = np.zeros((48, 48, 3), dtype=np.uint8)
    r = a.predict_proba(img)
    assert isinstance(r, ModelResult)
    assert set(r.probabilities) == {"neutral", "happy", "sad"}
    assert abs(sum(r.probabilities.values()) - 1.0) < 1e-9
    assert r.top1 == "happy"
    assert r.confidence == 0.5
    assert r.uncertainty == 0.5
    assert r.latency_ms == 12.0
    assert r.metadata.model_id == "dummy-test"
    assert r.metadata.input_shape == (48, 48, 3)
    assert r.metadata.calibrated is False
    assert r.metadata.supported_classes == {"neutral", "happy", "sad"}
    assert r.metadata.provenance["licence"] == "MIT-test"


def test_predict_proba_rejects_wrong_input_shape():
    a = DummyAdapter()
    bad = np.zeros((32, 32, 3), dtype=np.uint8)
    try:
        a.predict_proba(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for wrong input shape")


def test_registry_round_trip():
    register_adapter("dummy-test", DummyAdapter)
    a = get_adapter("dummy-test")
    assert isinstance(a, DummyAdapter)
    assert "dummy-test" in list_adapters()


def test_unknown_adapter_rejected():
    try:
        get_adapter("no-such-adapter")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown adapter")


def test_unsupported_classes_not_remapped():
    """An adapter reporting probabilities outside supported_classes violates
    the contract: supported_classes is authoritative."""
    a = DummyAdapter()
    img = np.zeros((48, 48, 3), dtype=np.uint8)
    r = a.predict_proba(img)
    assert set(r.probabilities) <= set(r.metadata.supported_classes)


def test_metadata_uncalibrated_by_default():
    a = DummyAdapter()
    img = np.zeros((48, 48, 3), dtype=np.uint8)
    r = a.predict_proba(img)
    assert r.metadata.calibrated is False
    assert "source" in r.metadata.provenance
    assert "licence" in r.metadata.provenance


def test_model_metadata_required_fields():
    m = ModelMetadata(
        model_id="x",
        version="1",
        input_shape=(48, 48, 1),
        class_mapping={0: "a", 1: "b"},
        calibrated=True,
        supported_classes={"a", "b"},
        provenance={"source": "s"},
    )
    assert m.model_id == "x"
    assert m.calibrated is True
