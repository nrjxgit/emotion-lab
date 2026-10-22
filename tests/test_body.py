import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "app" / "body"))


def test_static_low_energy_missing():
    from movement import movement_features, predict_body_emotion

    track = [[(0.5, 0.5)] * 5 for _ in range(30)]
    f = movement_features(track)
    assert f["energy"] < 0.1
    r = predict_body_emotion(f)
    assert r["uncertainty"] > 0.4
    assert r["type"] == "motion-only"
    assert predict_body_emotion(movement_features([]))["visible"] is False


def test_oscillating_high_energy():
    import math
    from movement import movement_features

    track = [[(0.5 + 0.1 * math.sin(i / 2.0), 0.5)] * 5 for i in range(30)]
    f = movement_features(track)
    assert f["energy"] > 0.02
