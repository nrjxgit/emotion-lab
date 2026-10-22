"""Manufactured numerical fixtures only; seed505; no model or data payloads."""
from copy import deepcopy
import builtins
import math
import numpy as np
import pytest
from training import synthetic_methods as methods


@pytest.fixture(autouse=True)
def no_network_or_models(monkeypatch):
    import socket
    def denied(*a, **k): raise AssertionError("network forbidden")
    monkeypatch.setattr(socket.socket, "connect", denied)
    original = builtins.__import__
    def guarded(name, *a, **k):
        if name.startswith(("torch", "tensorflow", "transformers", "backend", "emolab.adapters")):
            raise AssertionError("model import forbidden")
        return original(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", guarded)


def test_hand_metrics_and_zero_selection_are_not_invented_zero_error():
    result = methods.metrics([[.75, .25]], [0])
    assert result["nll"] == pytest.approx(-np.log(.75))
    assert result["brier"] == pytest.approx(.125)
    assert result["ece10"] == pytest.approx(.25) and result["error"] == 0
    assert methods.metrics(np.empty((0, 7)), [])["error"] is None
    for key in ("nll", "brier", "ece10"): assert methods.metrics(np.empty((0, 7)), [])[key] is None


@pytest.mark.parametrize("temperature", [.25, .5, 1., 2., 4., np.float64(2.), np.float32(.5)])
def test_softmax_shift_positive_temperature_argmax_and_first_tie(temperature):
    z = np.array([[2., 2., -1.], [-2., 0., 3.]])
    before = z.copy(); p = methods.probabilities(z, temperature)
    assert np.allclose(p, methods.probabilities(z + 37, temperature))
    assert p.argmax(1).tolist() == [0, 2] and np.array_equal(z, before)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "bad"])
def test_nonfinite_or_nonnumeric_logits_rejected(value):
    with pytest.raises(ValueError): methods.probabilities([[value, 0]])


@pytest.mark.parametrize("temperature", [0, -1, float("inf"), float("nan"), True, np.bool_(True), "1"])
def test_invalid_temperature_rejected(temperature):
    with pytest.raises(ValueError): methods.probabilities([[1, 0]], temperature)


@pytest.mark.parametrize("bad", [[], [[1]], [[1]*8], [[1, 0]]*10001])
def test_bounds_and_shapes(bad):
    with pytest.raises(ValueError): methods.probabilities(bad)


@pytest.mark.parametrize("p,y", [([[.5, .4]], [0]), ([[1.1, -.1]], [0]), ([[1, 0]], [True]),
                                  ([[1, 0]], [2]), ([[1, 0]], [0.]), ([[1, 0]], [])])
def test_metric_validation(p, y):
    with pytest.raises(ValueError): methods.metrics(p, y)


def test_known_overconfident_wrong_half_prefers_max_grid_temperature():
    selected = methods.select_temperatures([[4, 0], [4, 0]], [0, 1])
    values = {row["method"]: row["temperature"] for row in selected["rows"]}
    assert values["identity_T1"] == 1
    assert values["validation_nll_grid"] == 4 and values["validation_brier_grid"] == 4
    assert values["legacy_validation_nll_200_grid"] == 5
    tied = methods.select_temperatures([[0, 0], [0, 0]], [0, 1])
    assert [r["temperature"] for r in tied["rows"]][1:3] == [.25, .25]


def test_split_before_outcomes_disjoint_roles_determinism_and_assessment_isolation(monkeypatch):
    assigned = methods.assign_roles(); sets = [set(ids) for ids in assigned.values()]
    assert all(len(ids) == 20 for ids in sets) and len(set.union(*sets)) == 60
    assert all(not left & right for i, left in enumerate(sets) for right in sets[i+1:])
    cohort = methods.generate_cohort(); validation = cohort["roles"]["validation"]
    seen = []; original = methods.calibration.fit_temperature
    def spy(z, y):
        assert np.array_equal(z, validation["logits"]) and np.array_equal(y, validation["labels"])
        seen.append(True); return original(z, y)
    monkeypatch.setattr(methods.calibration, "fit_temperature", spy)
    monkeypatch.setattr(methods, "generate_cohort", lambda: deepcopy(cohort))
    before = methods.run_experiment()
    cohort["roles"]["assessment"]["labels"] = (cohort["roles"]["assessment"]["labels"] + 1) % 7
    # Fit labels are unused by design; assessment mutation and fitter spy are the meaningful isolation checks.
    cohort["roles"]["fit"]["labels"] = (cohort["roles"]["fit"]["labels"] + 2) % 7
    after = methods.run_experiment()
    assert before["selection"] == after["selection"] and len(seen) == 2
    assert before["assessment"] != after["assessment"]


def test_report_labels_predeclared_methods_and_heldout_rows():
    report = methods.run_experiment()
    assert report == methods.run_experiment()
    assert report["total_rows"] == 1440 and report["bootstrap_replicates"] == 0
    assert [r["method"] for r in report["assessment"]["rows"]] == list(methods.METHODS)
    assert all(r["metrics"]["n"] == 480 for r in report["assessment"]["rows"])
    def check(value):
        if isinstance(value, dict):
            assert value["synthetic_only"] is True and value["exposure"] == methods.MARKER and value["seed"] == 505
            for child in value.values(): check(child)
        elif isinstance(value, list):
            for child in value: check(child)
    check(report)
    assert "cannot establish real model accuracy" in report["limitation"] and "no winner is deployed" in report["limitation"]


def test_discriminating_interior_nll_and_brier_optima_from_binary_hand_oracle():
    # Eight modest correct margins and two wrong margins; no random fixture search.
    margins = [1.] * 8 + [-.2, -3.]
    logits = [[margin, 0.] for margin in margins]; labels = [0] * 10
    # Independent scalar sigmoid/log oracle, not calibration/metrics helpers.
    nll = []; brier = []
    for t in (.25, .5, 1., 2., 4.):
        correct = [1 / (1 + math.exp(-margin / t)) for margin in margins]
        nll.append(-sum(math.log(p) for p in correct) / 10)
        brier.append(sum(2 * (1-p)**2 for p in correct) / 10)
    assert nll.index(min(nll)) == 3  # T=2, interior
    assert brier.index(min(brier)) == 1  # T=.5, different interior optimum
    selected = {r["method"]: r["temperature"] for r in methods.select_temperatures(logits, labels)["rows"]}
    assert selected["validation_nll_grid"] == 2.
    assert selected["validation_brier_grid"] == .5


def test_ece_lower_exclusive_upper_inclusive_boundary_and_report_statement():
    # At confidence .6: correct row lies exactly on the boundary, wrong row .65.
    # Existing (lower,upper] separates them -> .5*.4 + .5*.65 = .525.
    # Left-closed/right-open would group them -> abs(.5 - .625) = .125.
    result = methods.metrics([[.6, .4], [.65, .35]], [0, 1])
    assert result["ece10"] == pytest.approx(.525)
    report = methods.run_experiment()
    assert "(lower, upper]" in report["ece10_convention"]
    assert "differs from the DESIGN protocol" in report["ece10_convention"]
    assert methods.ECE10_CONVENTION in report["limitation"]
