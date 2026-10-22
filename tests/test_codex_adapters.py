"""Tests for Codex body-language adapter and face+body fusion stub."""

import sys
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emolab.adapters.codex_body import CodexBodyAdapter
from emolab.adapters.fusion_stub import fused_side_by_side

FER7 = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def test_codex_body_adapter_contract():
    a = CodexBodyAdapter()
    assert a.name == "codex-body"
    assert a.input_shape == (224, 224, 3)
    assert a.native_labels == tuple(FER7)
    assert a.calibrated is False
    # stub raises NotImplementedError until weights/licence verified
    try:
        a.predict_proba(np.zeros((224, 224, 3), dtype=np.uint8))
    except NotImplementedError:
        pass
    else:
        raise AssertionError("expected NotImplementedError")


def test_fusion_stub_side_by_side():
    face = {
        "probs": {
            "angry": 0.8,
            "disgust": 0.0,
            "fear": 0.0,
            "happy": 0.1,
            "sad": 0.0,
            "surprise": 0.0,
            "neutral": 0.1,
        }
    }
    body = {
        "probs": {
            "angry": 0.2,
            "disgust": 0.0,
            "fear": 0.1,
            "happy": 0.1,
            "sad": 0.0,
            "surprise": 0.0,
            "neutral": 0.6,
        }
    }
    r = fused_side_by_side(face, body)
    assert r["mode"] == "side-by-side"
    assert r["face"] == face
    assert r["body"] == body
    # no conflation of modalities
    assert r["face"]["probs"]["angry"] == 0.8
    assert r["body"]["probs"]["neutral"] == 0.6


def test_fusion_stub_handles_missing_modality():
    face = {
        "probs": {
            "angry": 0.8,
            "disgust": 0.0,
            "fear": 0.0,
            "happy": 0.1,
            "sad": 0.0,
            "surprise": 0.0,
            "neutral": 0.1,
        }
    }
    r = fused_side_by_side(face, None)
    assert r["mode"] == "face-only"
    assert r["face"] == face
    assert r["body"] is None
