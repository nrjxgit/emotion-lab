import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))


def test_shapes_match_model_config():
    from numerical_audit import audit_shapes

    r = audit_shapes()
    assert r["mismatches"] == [], r["mismatches"]
    assert r["layers_checked"] >= 8
    assert r["bn_layers"] == 14


def test_determinism_and_softmax():
    from numerical_audit import audit_determinism

    r = audit_determinism()
    assert r["max_abs_diff"] == 0.0
    assert abs(r["prob_sum"] - 1.0) < 1e-9


def test_bn_eps_documented():
    from numerical_audit import BN_EPS

    assert BN_EPS == 0.001
