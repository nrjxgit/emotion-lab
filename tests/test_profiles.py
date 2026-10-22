import sys, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "app" / "face"))


def test_preprocess_v2_range_matches_upstream():
    from fer_model import preprocess

    black = np.zeros((48, 48, 3), np.uint8)
    white = np.full((48, 48, 3), 255, np.uint8)
    assert preprocess(black).max() == 0.0  # legacy default unchanged
    assert preprocess(black, v2=True).min() == -1.0
    assert preprocess(white, v2=True).max() == 1.0


def test_corrected_is_default_and_legacy_reproduces():
    from expression import ExpressionPredictor

    rng = np.random.default_rng(3)
    crop = (rng.random((90, 90, 3)) * 255).astype(np.uint8)
    new = ExpressionPredictor()
    assert new.profile == "corrected"
    leg = ExpressionPredictor(profile="legacy")
    assert leg.profile == "legacy"
    assert new.predict(crop, [])["source"] == ExpressionPredictor.TRAINED_SOURCE
    # legacy path byte-identical to manual legacy computation
    from fer_model import load_fer, preprocess, predict_fer, FER7
    from expression import FERMAP

    w = load_fer()
    p7 = [float(v) for v in predict_fer(preprocess(crop), w)]
    probs = {
        c: 0.0
        for c in [
            "happiness",
            "sadness",
            "anger",
            "surprise",
            "fear",
            "disgust",
            "neutral",
            "unknown",
        ]
    }
    for name, p in zip(FER7, p7):
        probs[FERMAP[name]] += p
    mx = max(probs.values())
    probs["unknown"] = max(0.0, (1.0 - mx) * 0.25)
    s = sum(probs.values())
    expected = {k: v / s for k, v in probs.items()}
    got = leg.predict(crop, [])["probs"]
    assert max(abs(got[k] - expected[k]) for k in expected) == 0.0
    # corrected and legacy differ on non-uniform input, agree on structure
    a = new.predict(crop, [])["probs"]
    b = leg.predict(crop, [])["probs"]
    assert set(a) == set(b) and abs(sum(a.values()) - 1.0) < 1e-9
    assert max(abs(a[k] - b[k]) for k in a) > 1e-6


def test_margin_crop_clamps_and_expands():
    sys.path.insert(0, str(ROOT / "backend" / "app"))
    from face.crop import crop_face

    img = np.ones((100, 100, 3), np.uint8)
    c = crop_face(img, {"x": 80, "y": 80, "w": 40, "h": 40}, margin=0.25)
    assert c.shape[0] <= 100 and c.shape[1] <= 100 and c.size > 0
    c0 = crop_face(img, {"x": 10, "y": 10, "w": 20, "h": 20}, margin=0.0)
    assert c0.shape == (20, 20, 3)
