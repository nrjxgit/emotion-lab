import numpy as np


def test_minixception_adapter_contract():
    import sys

    sys.path.insert(0, "backend/app/face")
    from adapters import get_adapter, FER7

    a = get_adapter("mini-xception")
    assert a.name == "mini-xception" and list(a.native_labels) == FER7
    p = a.predict_proba(np.full((80, 80, 3), 128, np.uint8))
    assert p.shape == (7,) and abs(p.sum() - 1.0) < 1e-5


def test_vit_head_mapping_with_stub():
    import sys

    sys.path.insert(0, "backend/app/face")
    from adapters import ViTAdapter

    fake_labels = ["sad", "disgust", "angry", "neutral", "fear", "surprise", "happy"]
    a = ViTAdapter(
        id2label=fake_labels,
        forward_fn=lambda x: np.array([0, 0, 1, 0, 0, 0, 0], float),
    )
    p = a.predict_proba(np.zeros((64, 64, 3), np.uint8))
    assert int(np.argmax(p)) == 0  # angry in FER7


def test_hsemotion_stub_raises_informative():
    import sys

    sys.path.insert(0, "backend/app/face")
    from adapters import get_adapter
    import pytest

    with pytest.raises(NotImplementedError, match="separate"):
        get_adapter("hsemotion").predict_proba(np.zeros((8, 8, 3), np.uint8))


def test_unknown_adapter_rejected():
    import sys

    sys.path.insert(0, "backend/app/face")
    from adapters import get_adapter
    import pytest

    with pytest.raises(ValueError):
        get_adapter("nope")


def test_vit_pins_exact_revision_without_loading_weights():
    from backend.app.face.adapters import ViTAdapter

    a = ViTAdapter()
    assert a.revision == "747cf16692eea925b54b7b543cf436848128b68d"
