"""Synthetic stream generator for HB-1 (EL-006 S1). Manufactured numbers only."""
import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from socialcue import state as hb1  # noqa: E402
from socialcue import synthetic as syn  # noqa: E402


def test_scenario_list_is_the_seven_designed_scenarios():
    assert set(syn.SCENARIOS) == {"calm", "nervous-like", "unsure-like", "needs-help-like",
                                  "insufficient", "multi-face", "dropout"}
    assert set(syn.EXPECTED) == set(syn.SCENARIOS)
    assert "SYNTHETIC" in syn.LABEL


@pytest.mark.parametrize("name", syn.SCENARIOS)
def test_each_scenario_yields_its_intended_state(name):
    result = hb1.analyze(syn.generate(name))
    want = syn.EXPECTED[name]
    assert {s["name"] for s in result["states"]} == want["states"], result
    assert result["insufficient"]["flag"] is want["insufficient"]
    for reason in want.get("reasons", ()):
        assert reason in result["insufficient"]["reasons"]
    if want["insufficient"]:
        assert result["states"] == []


@pytest.mark.parametrize("name", syn.SCENARIOS)
@pytest.mark.parametrize("seed", [1, 7, 42, 505, 2026])
def test_intended_state_is_robust_across_seeds(name, seed):
    result = hb1.analyze(syn.generate(name, seed=seed))
    assert {s["name"] for s in result["states"]} == syn.EXPECTED[name]["states"]
    assert result["insufficient"]["flag"] is syn.EXPECTED[name]["insufficient"]


def test_calm_is_the_default_no_cue_case():
    r = hb1.analyze(syn.generate("calm"))
    assert r["states"] == [] and r["insufficient"] == {"flag": False, "reasons": []}


@pytest.mark.parametrize("name", syn.SCENARIOS)
def test_records_are_well_formed_manufactured_frames(name):
    recs = syn.generate(name)
    times = [r.t for r in recs]
    assert times == sorted(times) and len(set(times)) == len(times)
    assert math.isclose(times[1] - times[0], 1 / 8.0, rel_tol=1e-6)
    for r in recs:
        assert isinstance(r, hb1.FrameRecord)
        if r.box is None:
            assert r.landmarks is None and r.expression is None and r.face_count == 0
            continue
        assert len(r.landmarks) == 5 and all(len(p) == 2 for p in r.landmarks)
        assert len(r.box) == 4
        assert set(r.expression) == set(hb1.FER7)
        assert math.isclose(sum(r.expression.values()), 1.0, abs_tol=1e-9)
        assert all(p >= 0 for p in r.expression.values())
        assert 0.0 <= r.quality <= 1.0 and 0.0 <= r.arousal <= 1.0


def test_generation_is_deterministic_per_seed_and_varies_across_seeds():
    assert syn.generate("nervous-like", seed=3) == syn.generate("nervous-like", seed=3)
    assert syn.generate("nervous-like", seed=3) != syn.generate("nervous-like", seed=4)


def test_strength_zero_removes_the_cue():
    for name in ("nervous-like", "unsure-like", "needs-help-like"):
        assert hb1.analyze(syn.generate(name, strength=0.0))["states"] == []


def test_noise_is_independent_of_strength():
    a = syn.generate("nervous-like", strength=0.2)
    b = syn.generate("nervous-like", strength=0.8)
    assert [r.t for r in a] == [r.t for r in b]
    # same unit noise, only scaled: centred mouth offsets are proportional
    ax = [r.landmarks[3][0] - 300.0 for r in a]
    bx = [r.landmarks[3][0] - 300.0 for r in b]
    ratio = (0.005 + (0.10 - 0.005) * 0.8) / (0.005 + (0.10 - 0.005) * 0.2)
    assert all(math.isclose(y, x * ratio, abs_tol=1e-6) for x, y in zip(ax, bx))


def test_dropout_has_a_late_gap_and_multi_face_has_two_faces():
    d = syn.generate("dropout")
    assert all(r.face_count == 1 for r in d if r.t < 5.0)
    assert sum(1 for r in d if r.t >= 5.0 and r.face_count == 0) >= 8
    m = syn.generate("multi-face")
    assert {r.face_count for r in m} == {1, 2}


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError):
        syn.generate("angry-customer")
