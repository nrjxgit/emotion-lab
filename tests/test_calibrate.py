import numpy as np


def test_temperature_lowersece_or_equal():
    import sys

    sys.path.insert(0, "training")
    from calibrate_temp import fit_temperature, apply_temperature, ece_from_proba

    rng = np.random.default_rng(0)
    logits = rng.normal(size=(400, 3)) * 2.0
    y = logits.argmax(1)
    flip = rng.random(400) < 0.2  # 20% label noise -> finite optimal T
    y = np.where(flip, (y + 1) % 3, y)
    T = fit_temperature(logits, y)
    assert 0.05 < T < 20.0
    from calibrate_temp import _softmax

    def nll(P):
        return float(-np.log(P[np.arange(len(y)), y] + 1e-12).mean())

    assert nll(apply_temperature(logits, T)) <= nll(_softmax(logits)) + 1e-9
    assert ece_from_proba(apply_temperature(logits, T), y) < 0.5


def test_abstention_curve_monotone_coverage():
    import sys

    sys.path.insert(0, "training")
    from calibrate_temp import abstention_curve

    rng = np.random.default_rng(1)
    P = rng.dirichlet([1, 1, 1], 200)
    y = rng.integers(0, 3, 200)
    curve = abstention_curve(P, y, thresholds=[0.0, 0.5, 0.9])
    covs = [c["coverage"] for c in curve]
    assert covs == sorted(covs, reverse=True) and curve[0]["coverage"] == 1.0
