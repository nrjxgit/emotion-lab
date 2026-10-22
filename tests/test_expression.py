import sys, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "app" / "face"))


def test_trained_source_loaded():
    from expression import ExpressionPredictor

    p = ExpressionPredictor()
    assert p.source == ExpressionPredictor.TRAINED_SOURCE, p._status


def test_trained_probs_sum():
    from expression import ExpressionPredictor

    p = ExpressionPredictor()
    gray = np.full((64, 64, 3), 128, dtype=np.uint8)
    r = p.predict(gray, [])
    assert abs(sum(r["probs"].values()) - 1.0) < 1e-5
    assert r["source"] == ExpressionPredictor.TRAINED_SOURCE


def test_observable_analysis_labeled():
    from expression import observable_feature_analysis

    aus = [{"code": "AU12", "intensity": 0.95}, {"code": "AU06", "intensity": 0.9}]
    r = observable_feature_analysis(aus)
    assert r["probs"]["happiness"] == max(r["probs"].values())
    assert r["source"] == "observable-feature-analysis"
