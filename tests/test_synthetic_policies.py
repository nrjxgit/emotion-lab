"""Known-answer manufactured stream policies; no real camera, data or models."""
from copy import deepcopy
import builtins
import numpy as np
import pytest
from training import synthetic_policies as policies
from training import synthetic_methods as methods
from training.synthetic_methods import calibration


@pytest.fixture(autouse=True)
def no_models_or_network(monkeypatch):
    import socket
    def denied(*a, **k): raise AssertionError("network forbidden")
    monkeypatch.setattr(socket.socket, "connect", denied)
    original = builtins.__import__
    def guarded(name, *a, **k):
        if name.startswith(("torch", "tensorflow", "transformers", "backend", "emolab.adapters")):
            raise AssertionError("model import forbidden")
        return original(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", guarded)


def event(p, truth="angry", **kwargs):
    vector = None if p is None else [p, 0, 0, 1-p, 0, 0, 0]
    return policies._event(vector, truth, **kwargs)


def predictions(result): return [r["prediction"] for r in result["rows"]]


def test_known_stable_truth_ema_removes_flicker_and_error():
    events = [event(p) for p in [.9, .4, .6, .4]]
    raw = policies.simulate(events); ema = policies.simulate(events, smoothing="ema_alpha_0.5")
    assert predictions(raw) == ["angry", "happy", "angry", "happy"]
    assert predictions(ema) == ["angry"] * 4
    assert raw["metrics"]["flicker_count"] == 3 and ema["metrics"]["flicker_count"] == 0
    assert raw["metrics"]["selective_error"] == .5 and ema["metrics"]["selective_error"] == 0


def test_known_step_change_has_raw_zero_lag_and_ema_one_frame_lag():
    events = [event(.9)]*2 + [event(.1, "happy")]*3
    raw = policies.simulate(events); ema = policies.simulate(events, smoothing="ema_alpha_0.5")
    assert raw["metrics"]["mean_reached_transition_lag"] == 0
    assert ema["metrics"]["mean_reached_transition_lag"] == 1
    assert ema["metrics"]["transition_count"] == 1 and ema["metrics"]["flicker_count"] == 0


@pytest.mark.parametrize("reset", ["missing", "identity", "run", "stop", "support"])
def test_reset_drops_ema_memory_and_prevents_cross_identity_decisions(reset):
    first = event(.99); final = event(.1, "happy"); events = [first]
    if reset == "missing": events.append(event(None))
    elif reset == "stop": events.append(event(None, active=False))
    elif reset == "identity": final["identity"] = "SYNTHETIC-new-actor"
    elif reset == "run": final["run"] = "SYNTHETIC-new-run"
    else: final["supported_classes"] = ["angry", "happy"]
    events.append(final); before = deepcopy(events)
    result = policies.simulate(events, smoothing="ema_alpha_0.5")
    assert result["rows"][-1]["confidence"] == .9 and predictions(result)[-1] == "happy"
    assert result["metrics"]["transition_count"] == 0 and events == before


def test_causal_prefix_is_unchanged_by_future_values():
    events = [event(.9), event(.4), event(.6)]
    prefix = policies.simulate(events, smoothing="ema_alpha_0.5")
    extended = policies.simulate(events + [event(.01, "happy")]*5, smoothing="ema_alpha_0.5")
    assert prefix["rows"] == extended["rows"][:3]


def test_unsupported_classes_are_unavailable_not_supported_zero():
    result = policies.simulate([event(.8, support=["angry", "happy"])])
    scores = {r["label"]: r for r in result["rows"][0]["scores"]["rows"]}
    assert scores["surprise"]["value"] is None and scores["surprise"]["support"] == "unsupported"
    assert scores["angry"]["value"] == .8 and scores["happy"]["value"] == pytest.approx(.2)


def test_known_distribution_threshold_09_has_zero_error_at_half_coverage():
    events = policies.generate_distribution()
    for smoothing in policies.SMOOTHING:
        for threshold in policies.THRESHOLDS:
            result = policies.simulate(events, smoothing=smoothing, threshold=threshold)["metrics"]
            assert result["active_denominator"] == 400
            if threshold == .9:
                assert result["coverage"] == .5 and result["selective_error"] == 0 and result["accepted"] == 200
            else: assert result["coverage"] == 1 and result["selective_error"] == .5


def test_zero_coverage_is_null_and_exposes_legacy_empty_subset_semantics():
    events = [event(.6), event(None), event(None, active=False)]
    result = policies.simulate(events, threshold=.9)["metrics"]
    assert result["active_denominator"] == 2 and result["accepted"] == 0
    assert result["coverage"] == 0 and result["selective_error"] is None
    assert result["missing_face"] == 1 and result["stopped"] == 1
    assert calibration.abstention_curve([[.6, .4]], [0], thresholds=(.9,))[0]["accuracy"] == 0
    stopped = policies.simulate([event(None, active=False)])["metrics"]
    assert stopped["coverage"] is None and stopped["selective_error"] is None


@pytest.mark.parametrize("threshold", [.5, .7, .9])
def test_threshold_is_inclusive_at_exact_score(threshold):
    assert policies.simulate([event(threshold)], threshold=threshold)["metrics"]["accepted"] == 1


def test_unreached_transition_lag_is_null_not_zero():
    events = [event(.9), event(.9, "happy"), event(None, "happy")]
    result = policies.simulate(events)
    assert result["transitions"]["rows"][0]["lag_frames"] is None
    assert result["metrics"]["unresolved_transition_count"] == 1
    assert result["metrics"]["mean_reached_transition_lag"] is None


@pytest.mark.parametrize("kind", ["empty", "too_many", "extra", "nan", "negative", "mass", "bool", "truth", "id", "stopped_scores", "support", "unsupported_mass", "width", "threshold", "smoothing"])
def test_validation_rejects_undeclared_or_invalid_streams(kind, monkeypatch):
    events = [event(.9)]; kwargs = {}
    if kind == "empty": events = []
    elif kind == "too_many": events *= 10001
    elif kind == "extra": events[0]["extra"] = 1
    elif kind == "nan":
        events[0]["probabilities"] = [float("nan"), 0, 0, 0, 0, 0, 0]
        # Isolate finite validation: NaN would otherwise also fail the later mass check.
        monkeypatch.setattr(methods.np, "allclose", lambda *a, **k: True)
    elif kind == "negative": events[0]["probabilities"] = [-.1, .6, .5, 0, 0, 0, 0]  # mass1; only lower bound invalid
    elif kind == "mass": events[0]["probabilities"][0] = .5
    elif kind == "bool": events[0]["probabilities"] = [True, 0, 0, 0, 0, 0, 0]  # mass1; numeric coercion otherwise succeeds
    elif kind == "truth": events[0]["truth"] = "unknown"
    elif kind == "id": events[0]["identity"] = "real"
    elif kind == "stopped_scores": events[0]["active"] = False
    elif kind == "support":
        events[0]["supported_classes"] = ["angry", "angry"]
        events[0]["probabilities"] = [1., 0, 0, 0, 0, 0, 0]  # no unsupported positive mass
    elif kind == "unsupported_mass": events[0]["supported_classes"] = ["angry"]
    elif kind == "width": events[0]["probabilities"] = [.9, .1]
    elif kind == "threshold": kwargs["threshold"] = .95
    else: kwargs["smoothing"] = "future-aware"
    expected = {
        "empty": "bounded explicit event list", "too_many": "bounded explicit event list",
        "extra": "exact synthetic event structure", "nan": "finite bounded matrix",
        "negative": "unit probabilities", "mass": "unit probabilities", "bool": "real numeric matrix",
        "truth": "active boolean and named synthetic truth", "id": "bounded manufactured run/identity",
        "stopped_scores": "stopped event cannot carry scores", "support": "unique named supported classes",
        "unsupported_mass": "FER7 width and zero mass", "width": "FER7 width and zero mass",
        "threshold": "undeclared decision threshold", "smoothing": "undeclared smoothing policy",
    }
    with pytest.raises(ValueError, match=expected[kind]): policies.simulate(events, **kwargs)


def test_report_predeclared_grid_labels_determinism_and_limitations():
    report = policies.run_experiment(); assert report == policies.run_experiment()
    assert report["total_rows"] == 466 and report["bootstrap_replicates"] == 0
    assert report["policies"]["thresholds"] == ["off", .5, .7, .9]
    assert all(len(t["rows"]) == 8 for t in report["comparison"]["rows"])
    def check(value):
        if isinstance(value, dict):
            assert value["synthetic_only"] is True and value["exposure"] == policies.MARKER and value["seed"] == 505
            for child in value.values(): check(child)
        elif isinstance(value, list):
            for child in value: check(child)
    check(report)
    assert "cannot show real accuracy" in report["limitation"] and "no production evidence threshold" in report["limitation"]


def test_flicker_suppression_while_never_settled_and_after_settling_ends():
    events = [event(.9)] + [event(p, "fear") for p in [.9, .1, .9, .1]]
    result = policies.simulate(events)
    assert predictions(result) == ["angry", "angry", "happy", "angry", "happy"]
    assert result["metrics"]["flicker_count"] == 0  # all four frames are settling, never correct fear
    assert result["metrics"]["settling_never_ended_count"] == 1
    assert result["metrics"]["unresolved_transition_count"] == 1
    # After an accepted correct fear, ordinary stable-truth changes count again.
    correct = policies._event([0, 0, 1, 0, 0, 0, 0], "fear")
    reached = policies.simulate(events + [correct, event(.9, "fear")])
    assert reached["metrics"]["settling_never_ended_count"] == 0
    assert reached["metrics"]["flicker_count"] == 1


def test_abstention_breaks_flicker_adjacency_without_resetting_ema():
    events = [event(.9), event(.5), event(.1)]
    raw = policies.simulate(events, threshold=.7)
    assert predictions(raw) == ["angry", None, "happy"]
    assert raw["metrics"]["flicker_count"] == 0
    assert policies.simulate(events)["metrics"]["flicker_count"] == 1
    # .9 -> EMA .7 (rejected by .9) -> .4; a reset-on-abstention bug would yield .1.
    ema = policies.simulate(events, smoothing="ema_alpha_0.5", threshold=.9)
    assert ema["rows"][-1]["confidence"] == pytest.approx(.6)


def test_report_explicitly_carries_stream_and_by_construction_limits():
    report = policies.run_experiment()
    tables = {table["scenario"]: table for table in report["comparison"]["rows"]}
    assert "Flicker is not counted while settling" in tables["stream"]["limitation"]
    assert "Abstention breaks adjacency" in tables["stream"]["limitation"]
    assert "Do not compare flicker across thresholds" in report["limitation"]
    assert tables["distribution"]["limitation"] == "holds by construction (confidence tied to correctness)"
    assert "holds by construction (confidence tied to correctness)" in report["limitation"]
