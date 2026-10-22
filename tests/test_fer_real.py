import sys, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "app" / "face"))

FER7 = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def test_weights_load():
    from fer_model import load_fer

    w = load_fer()
    for layer in [
        "conv2d_1",
        "separable_conv2d_1",
        "conv2d_7",
        "batch_normalization_1",
    ]:
        assert layer in w, f"missing {layer}"
    assert w["conv2d_1"]["kernel"].shape == (3, 3, 1, 8)
    assert w["conv2d_7"]["kernel"].shape == (3, 3, 128, 7)


def test_probs_sum_and_input_sensitive():
    from fer_model import load_fer, preprocess, predict_fer

    w = load_fer()
    dark = np.zeros((80, 80, 3), np.uint8)
    grad = np.tile(np.linspace(0, 255, 80, dtype=np.uint8)[None, :, None], (80, 1, 3))
    p1 = predict_fer(preprocess(dark), w)
    p2 = predict_fer(preprocess(grad), w)
    assert abs(p1.sum() - 1.0) < 1e-5 and abs(p2.sum() - 1.0) < 1e-5
    assert np.max(np.abs(p1 - p2)) > 0.01, "outputs identical -> weights not active"


def test_zeroed_weights_uniform():
    from fer_model import preprocess, predict_fer, load_fer
    import copy

    w = load_fer()
    w0 = {k: {kk: np.zeros_like(vv) for kk, vv in v.items()} for k, v in w.items()}
    grad = np.tile(np.linspace(0, 255, 80, dtype=np.uint8)[None, :, None], (80, 1, 3))
    p0 = predict_fer(preprocess(grad), w0)
    assert np.max(p0) - np.min(p0) < 0.01, "zeroed weights should be uniform"


def test_preprocess_shape_range():
    from fer_model import preprocess

    img = (np.random.rand(100, 90, 3) * 255).astype(np.uint8)
    x = preprocess(img)
    assert x.shape == (64, 64, 1) and x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0


def test_tf_crosscheck_when_available():
    import importlib.util

    if importlib.util.find_spec("tensorflow") is None:
        import pytest

        pytest.skip("tensorflow not installed")
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    from fer_model import load_fer, preprocess, predict_fer

    tfm = tf.keras.models.load_model(
        str(ROOT / "models" / "fer_mini_xception_h5.hdf5"), compile=False
    )
    w = load_fer()
    rng = np.random.default_rng(7)
    worst = 0.0
    for img in [
        np.zeros((80, 80, 3), np.uint8),
        np.full((80, 80, 3), 128, np.uint8),
        (rng.random((80, 80, 3)) * 255).astype(np.uint8),
    ]:
        x = preprocess(img)
        d = float(
            np.max(
                np.abs(
                    predict_fer(x, w) - np.asarray(tfm.predict(x[None], verbose=0)[0])
                )
            )
        )
        worst = max(worst, d)
    assert worst < 1e-5, worst
