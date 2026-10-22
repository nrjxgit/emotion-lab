"""Tests for OpenCode model adapters wrapping the existing backend adapters.

mini-Xception runs against the genuine local weights
(models/fer_mini_xception_h5.hdf5). ViT is exercised through a stub forward
function so the test does not depend on network access; the real ViTAdapter
is exercised end-to-end elsewhere.
"""

import sys
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emolab.inference import FaceEmotionAdapter, ModelMetadata, ModelResult
from emolab.adapters.mini_xception import MiniXceptionAdapter
from emolab.adapters.vit import ViTAdapter

FER7 = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def test_mini_xception_adapter_contract():
    a = MiniXceptionAdapter()
    assert a.name == "mini-xception"
    assert a.input_shape == (64, 64, 1)
    assert a.native_labels == tuple(FER7)
    assert a.calibrated is False

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    r = a.predict_proba(img)
    assert isinstance(r, ModelResult)
    assert set(r.probabilities) == set(FER7)
    assert abs(sum(r.probabilities.values()) - 1.0) < 1e-9
    assert r.top1 in FER7
    assert 0.0 <= r.confidence <= 1.0
    assert abs(r.confidence - r.probabilities[r.top1]) < 1e-9
    assert abs(r.confidence + r.uncertainty - 1.0) < 1e-9
    assert r.latency_ms >= 0.0
    m = r.metadata
    assert m.model_id == "mini-xception"
    assert m.version == "1.0"
    assert m.input_shape == (64, 64, 1)
    assert m.calibrated is False
    assert m.supported_classes == set(FER7)
    assert m.provenance["licence"].lower() == "mit"
    assert r.metadata.provenance["licence"].lower() == "mit"


def test_mini_xception_deterministic_on_identical_input():
    a = MiniXceptionAdapter()
    img = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
    r1 = a.predict_proba(img)
    r2 = a.predict_proba(img)
    assert r1.probabilities == r2.probabilities


def test_vit_adapter_contract_with_stub_forward():
    fake_labels = ["sad", "disgust", "angry", "neutral", "fear", "surprise", "happy"]
    a = ViTAdapter(
        id2label=fake_labels,
        forward_fn=lambda x: np.array([0, 0, 1, 0, 0, 0, 0], float),
    )
    assert a.name == "vit"
    assert a.revision == "747cf16692eea925b54b7b543cf436848128b68d"
    assert a.input_shape == (224, 224, 3)
    assert a.calibrated is False

    img = np.zeros((224, 224, 3), dtype=np.uint8)
    r = a.predict_proba(img)
    m = r.metadata
    assert m.model_id == "vit"
    assert m.input_shape == (224, 224, 3)
    assert m.calibrated is False
    assert m.provenance["revision"] == "747cf16692eea925b54b7b543cf436848128b68d"


def test_adapters_in_global_registry():
    from emolab.inference import get_adapter, list_adapters

    assert "mini-xception" in list_adapters()
    assert "vit" in list_adapters()
    assert isinstance(get_adapter("mini-xception"), MiniXceptionAdapter)
    assert isinstance(get_adapter("vit"), ViTAdapter)
