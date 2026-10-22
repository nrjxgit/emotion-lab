"""HB-1 hardening and discriminating tests from the independent review (EL-006 S1b). Synthetic only."""
import itertools
import json
import math
import pathlib
import random
import re
import sys
from dataclasses import replace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from socialcue import state as hb1  # noqa: E402
from socialcue import synthetic as syn  # noqa: E402

PINNED_CONFIG_SHA256 = "4940a4cbbb1df3c2af27407e78546e47812c5134b8fcc2aec0daad9e2874b6b3"
NAN, INF = float("nan"), float("inf")
BANNED = re.compile(r"\b(feels?|feeling|is nervous|is unsure|needs help now|detected emotion|reads?|diagnos\w*|truth|confirmed)\b", re.I)


def names(result):
    return [s["name"] for s in result["states"]]


def assert_clean(result):
    json.dumps(result, allow_nan=False)  # no NaN/inf anywhere
    assert math.isfinite(result["window"]["fps"])
    assert set(result) == {"states", "insufficient", "window", "heuristic", "proxy_eval_ref"}


NERV = syn.generate("nervous-like")
CALM = syn.generate("calm")
UNSURE = syn.generate("unsure-like")


def test_config_hash_unchanged_by_hardening():
    assert hb1.CONFIG_SHA256 == PINNED_CONFIG_SHA256


# --- malformed input never raises (F1-F4) -----------------------------------

def _each(recs, **kw):
    return [replace(r, **kw) for r in recs]


MALFORMED = {
    "face_count_none": lambda g: _each(g, face_count=None),
    "face_count_str": lambda g: _each(g, face_count="2"),
    "face_count_nan": lambda g: _each(g, face_count=NAN),
    "face_count_bool": lambda g: _each(g, face_count=True),
    "landmarks_ints": lambda g: _each(g, landmarks=(1, 2, 3, 4, 5)),
    "landmarks_triples": lambda g: _each(g, landmarks=((1, 2, 3),) * 5),
    "landmarks_none_points": lambda g: _each(g, landmarks=(None,) * 5),
    "landmarks_strings": lambda g: _each(g, landmarks=("ab",) * 5),
    "landmarks_short": lambda g: _each(g, landmarks=((1, 2),) * 4),
    "landmarks_nan": lambda g: _each(g, landmarks=((NAN, 0.0),) * 5),
    "box_garbage": lambda g: _each(g, box=("a", "b", "c", "d")),
    "expression_list": lambda g: _each(g, expression=[1, 2]),
    "expression_str": lambda g: _each(g, expression="x"),
    "expression_none_values": lambda g: _each(g, expression={k: None for k in hb1.FER7}),
    "expression_str_values": lambda g: _each(g, expression={k: "a" for k in hb1.FER7}),
    "t_none_all": lambda g: _each(g, t=None),
    "t_str_all": lambda g: _each(g, t="a"),
    "t_nan_all": lambda g: _each(g, t=NAN),
    "t_inf_all": lambda g: _each(g, t=INF),
    "quality_none": lambda g: _each(g, quality=None),
    "arousal_str": lambda g: _each(g, arousal="hi"),
    "none_items": lambda g: [None] * 5 + g[:3],
    "dict_items": lambda g: [{"t": 1.0}, "x", 7, 3.5] + g[:3],
    "empty": lambda g: [],
}


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_malformed_stream_never_raises(case):
    out = hb1.analyze(MALFORMED[case](list(NERV)))
    assert_clean(out)
    assert out["states"] == [] and out["insufficient"]["flag"] is True
    feats = hb1.extract_features(MALFORMED[case](list(NERV)))
    assert math.isfinite(feats["fps"]) and "dwell" in feats


@pytest.mark.parametrize("bad", [None, 5, 3.5, object(), "abc"])
def test_non_sequence_input_is_empty_not_a_crash(bad):
    out = hb1.analyze(bad)
    assert_clean(out)
    assert out["states"] == [] and out["insufficient"]["flag"]
    assert "no_face" in out["insufficient"]["reasons"]
    assert hb1.extract_features(bad)["usable_frames"] == 0


def test_generator_input_works():
    assert names(hb1.analyze(r for r in NERV)) == ["nervous-like"]


@pytest.mark.parametrize("bad_t", [NAN, INF, -INF, None, "a", True])
def test_bad_timestamp_appended_to_a_good_stream_keeps_the_good_state(bad_t):
    stream = list(NERV) + [replace(NERV[-1], t=bad_t)]
    out = hb1.analyze(stream)
    assert_clean(out)
    assert names(out) == ["nervous-like"]
    assert out == hb1.analyze(NERV)


def test_bad_records_in_the_middle_are_ignored():
    stream = list(NERV)
    stream.insert(10, replace(NERV[10], t=NAN))
    stream.insert(12, None)
    stream.insert(14, {"t": 1.0})
    stream.insert(16, replace(NERV[16], face_count=None))
    assert names(hb1.analyze(stream)) == ["nervous-like"]


def test_a_few_bad_frames_only_lower_coverage():
    stream = [replace(r, landmarks=(None,) * 5) if i % 5 == 0 else r for i, r in enumerate(NERV)]
    out = hb1.analyze(stream)
    assert_clean(out) and names(out) == ["nervous-like"] and out["window"]["usable_frames"] < 24


def test_seeded_field_fuzz_never_raises_and_stays_json_clean():
    rng = random.Random(505)
    junk = [None, NAN, INF, -INF, "x", [], {}, (), 1e308, -1e308, 0, -1, True, object]
    fields = ["t", "expression", "quality", "arousal", "valence", "box", "landmarks", "face_count"]
    stream = list(UNSURE)
    for _ in range(300):
        mutated = list(stream)
        for _ in range(rng.randint(1, 12)):
            i = rng.randrange(len(mutated))
            f = rng.choice(fields)
            v = rng.choice(junk)
            mutated[i] = replace(mutated[i], **{f: v() if v is object else v})
        out = hb1.analyze(mutated)
        assert_clean(out)
        for st in out["states"]:
            assert 0.0 < st["confidence"] <= 0.8 and 0.0 < st["score"] <= 1.0


def test_empty_config_is_not_silently_replaced_but_none_means_frozen():
    with pytest.raises(KeyError):
        hb1.analyze(NERV, {})
    assert hb1.analyze(NERV, None) == hb1.analyze(NERV)


# --- clamping (F6) -------------------------------------------------------------

def test_quality_and_arousal_are_clamped():
    base = syn.generate("nervous-like", strength=0.55)
    ref = hb1.analyze(base)
    assert names(ref) == ["nervous-like"] and ref["states"][0]["confidence"] < 0.8
    assert hb1.analyze(_each(base, quality=50.0)) == hb1.analyze(_each(base, quality=1.0))
    assert hb1.analyze(_each(base, arousal=9.0)) == hb1.analyze(_each(base, arousal=1.0))
    assert hb1.analyze(_each(base, arousal=-4.0)) == hb1.analyze(_each(base, arousal=0.0))
    assert hb1.analyze(_each(base, quality=-5.0))["insufficient"]["reasons"] == ["low_quality"]
    inflated = hb1.analyze(_each(base, quality=50.0))["states"][0]["confidence"]
    assert inflated <= 0.8 and inflated == hb1.analyze(_each(base, quality=1.0))["states"][0]["confidence"]


def test_out_of_range_probabilities_are_clamped():
    wild = {k: (5.0 if k == "fear" else -3.0) for k in hb1.FER7}
    clamped = {k: (1.0 if k == "fear" else 0.0) for k in hb1.FER7}
    assert hb1.analyze(_each(NERV, expression=wild)) == hb1.analyze(_each(NERV, expression=clamped))


# --- single-condition negative controls: nervous-like ----------------------------

def _mix(a, b, **override):
    """Frames of ``a`` with the head geometry (box, landmarks) of ``b``."""
    return [replace(x, box=y.box, landmarks=y.landmarks, **override) for x, y in zip(a, b)]


def test_nervous_needs_high_negative_arousal_probability():
    calm_expr = _each(NERV, expression=CALM[0].expression)
    out = hb1.analyze(calm_expr)  # jitter and arousal stay high, p_neg is low
    assert names(out) == []
    assert hb1.extract_features(calm_expr)["head_jitter"] > 0.04 and hb1.extract_features(calm_expr)["arousal"] > 0.5


def test_nervous_fires_on_jitter_alone_or_arousal_alone_but_not_neither():
    full = hb1.analyze(NERV)
    assert names(full) == ["nervous-like"]
    still_head = _mix(NERV, CALM)  # high p_neg + high arousal, calm head: arousal arm only
    assert hb1.extract_features(still_head)["head_jitter"] < 0.04
    assert names(hb1.analyze(still_head)) == ["nervous-like"]
    low_arousal = [replace(r, arousal=0.1) for r in NERV]  # jitter arm only
    assert hb1.extract_features(low_arousal)["arousal"] < 0.5
    assert names(hb1.analyze(low_arousal)) == ["nervous-like"]
    neither = _mix([replace(r, arousal=0.1) for r in NERV], CALM)
    assert names(hb1.analyze(neither)) == []  # an OR, not "drop either term": both gone => nothing


def test_nervous_arms_have_the_documented_margins():
    base = hb1.extract_features(NERV)
    cfg = hb1.HB1_CONFIG
    m = lambda **kw: hb1._base_margins(dict(base, **kw), cfg)["nervous-like"][0]  # noqa: E731
    assert m(p_neg_arousal=0.9, head_jitter=0.0, arousal=0.0) == 0.0
    assert m(p_neg_arousal=0.9, head_jitter=0.2, arousal=0.0) == 1.0
    assert m(p_neg_arousal=0.9, head_jitter=0.0, arousal=0.9) == 1.0
    assert m(p_neg_arousal=0.3, head_jitter=0.2, arousal=0.9) == 0.0  # strict threshold
    assert m(p_neg_arousal=0.45, head_jitter=0.2, arousal=0.9) == pytest.approx(0.5)


# --- single-condition negative controls: unsure-like ----------------------------

def _flicker(top, labels, n=len(UNSURE)):
    out = []
    for i in range(n):
        p = {k: (1 - top) / 6 for k in hb1.FER7}
        p[labels[i % len(labels)]] = top
        out.append(p)
    return out


def _with_expr(recs, exprs, **kw):
    return [replace(r, expression=e, **kw) for r, e in zip(recs, exprs)]


def test_unsure_control_stream_fires_all_terms():
    f = hb1.extract_features(UNSURE)
    assert names(hb1.analyze(UNSURE)) == ["unsure-like"]
    assert f["hesitation"] > 0.25 and 0.15 < f["gaze_away"] < 0.6 and f["arousal"] < 0.35 and f["mean_top_prob"] < 0.5


def test_unsure_needs_a_neutral_or_low_confidence_expression():
    confident_flicker = _with_expr(UNSURE, _flicker(0.8, ("angry", "happy", "disgust")))
    f = hb1.extract_features(confident_flicker)
    assert f["hesitation"] > 0.25 and f["p_neutral"] < 0.4 and f["mean_top_prob"] >= 0.5
    assert names(hb1.analyze(confident_flicker)) == []


def test_unsure_needs_hesitation():
    steady = _with_expr(UNSURE, _flicker(0.35, ("neutral",)))
    f = hb1.extract_features(steady)
    assert f["hesitation"] < 0.25 and f["mean_top_prob"] < 0.5
    assert names(hb1.analyze(steady)) == []


def test_unsure_needs_moderate_head_turning_in_both_directions():
    still = _mix(UNSURE, CALM)
    assert hb1.extract_features(still)["gaze_away"] < 0.15
    assert names(hb1.analyze(still)) == []
    # always turned away: above the band's upper edge (0.60)
    always = [replace(r, landmarks=(r.landmarks[0], r.landmarks[1], (r.landmarks[2][0] + 0.4 * 60.0, r.landmarks[2][1]),
                                    r.landmarks[3], r.landmarks[4])) for r in _mix(UNSURE, CALM)]
    assert hb1.extract_features(always)["gaze_away"] > 0.6
    assert names(hb1.analyze(always)) == []


def test_unsure_needs_low_arousal():
    aroused = _each(UNSURE, arousal=0.9)
    assert hb1.extract_features(aroused)["hesitation"] > 0.25
    assert names(hb1.analyze(aroused)) == []


def test_unsure_arm_margins_match_the_rules():
    base = hb1.extract_features(UNSURE)
    cfg = hb1.HB1_CONFIG
    best = dict(base, p_neutral=0.7, mean_top_prob=0.9, hesitation=0.9, gaze_away=0.35, arousal=0.0)
    m = lambda **kw: hb1._base_margins(dict(best, **kw), cfg)["unsure-like"][0]  # noqa: E731
    assert m() == pytest.approx(1.0)
    assert m(p_neutral=0.2) == 0.0 and m(mean_top_prob=0.3, p_neutral=0.2) > 0.0  # OR of neutral / low confidence
    assert m(hesitation=0.25) == 0.0 and m(gaze_away=0.15) == 0.0 and m(gaze_away=0.6) == 0.0
    assert m(arousal=0.35) == 0.0
    assert m(gaze_away=0.3) == pytest.approx(1.0) and m(gaze_away=0.2) == pytest.approx(0.5)


# --- monotone in each driver ----------------------------------------------------

def _grid(lo, hi, n=11):
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def _margins(feature, values, best, state):
    cfg = hb1.HB1_CONFIG
    return [hb1._base_margins(dict(best, **{feature: v}), cfg)[state][0] for v in values]


NERV_BEST = {"p_neg_arousal": 0.9, "head_jitter": 0.0, "arousal": 0.0}


@pytest.mark.parametrize("feature,best", [
    ("p_neg_arousal", {"head_jitter": 0.3, "arousal": 0.9}),
    ("head_jitter", {"p_neg_arousal": 0.9, "arousal": 0.0}),
    ("arousal", {"p_neg_arousal": 0.9, "head_jitter": 0.0}),
])
def test_nervous_margin_is_nondecreasing_in_each_driver(feature, best):
    hi = 0.12 if feature == "head_jitter" else 1.0  # jitter saturates at 0.10
    ms = _margins(feature, _grid(0.0, hi), dict(hb1.extract_features(NERV), **best), "nervous-like")
    assert all(b >= a for a, b in zip(ms, ms[1:])), ms
    assert ms[0] == 0.0 and ms[-1] == 1.0 and len(set(ms)) >= 3


UNSURE_BEST = {"p_neutral": 0.8, "mean_top_prob": 0.95, "hesitation": 1.0, "gaze_away": 0.35, "arousal": 0.0}


@pytest.mark.parametrize("feature,increasing", [("p_neutral", True), ("hesitation", True),
                                                ("mean_top_prob", False), ("arousal", False)])
def test_unsure_margin_is_monotone_in_each_driver(feature, increasing):
    best = dict(hb1.extract_features(UNSURE), **UNSURE_BEST)
    if feature == "mean_top_prob":
        best["p_neutral"] = 0.0  # make low confidence the only way into the neutral term
    ms = _margins(feature, _grid(0.0, 1.0), best, "unsure-like")
    ms = ms if increasing else ms[::-1]
    assert all(b >= a for a, b in zip(ms, ms[1:])), ms
    assert ms[0] == 0.0 and ms[-1] > 0.0 and len(set(ms)) >= 3


def test_unsure_gaze_band_rises_then_falls():
    best = dict(hb1.extract_features(UNSURE), **UNSURE_BEST)
    ms = _margins("gaze_away", _grid(0.0, 1.0, 21), best, "unsure-like")
    peak = ms.index(max(ms))
    assert ms[0] == 0.0 and ms[-1] == 0.0 and max(ms) == 1.0
    assert all(b >= a for a, b in zip(ms[:peak], ms[1:peak + 1]))
    assert all(b <= a for a, b in zip(ms[peak:], ms[peak + 1:]))


def test_confidence_scales_with_each_multiplier():
    base = syn.generate("nervous-like", strength=0.55)
    full = hb1.analyze(base)["states"][0]["confidence"]
    assert hb1.analyze(_each(base, quality=0.5))["states"][0]["confidence"] < full
    thinned = [r if i % 4 else replace(r, box=None, landmarks=None, expression=None, face_count=0)
               for i, r in enumerate(base)]
    out = hb1.analyze(thinned)
    assert out["states"] and out["states"][0]["confidence"] < full


# --- needs-help sustained fraction, directly ------------------------------------

@pytest.mark.parametrize("pattern,expected_score", [
    ((False, False, False, False), None),
    ((True, False, False, False), None),   # 0.25 < 0.5
    ((True, True, False, False), 0.5),
    ((False, False, True, True), 0.5),     # sustained earlier, not now: still a temporal escalation
    ((True, False, True, False), 0.5),
    ((True, True, True, False), 0.75),
    ((True, True, True, True), 1.0),
])
def test_needs_help_requires_half_of_the_evaluation_points(monkeypatch, pattern, expected_score):
    recs = syn.generate("needs-help-like")  # dwell 13.9 s, 3 gaze episodes: only the sustain term varies
    calls = iter(pattern)
    monkeypatch.setattr(hb1, "_fires_base", lambda f, cfg: next(calls))
    out = hb1.analyze(recs)
    help_states = [s for s in out["states"] if s["name"] == "needs-help-like"]
    if expected_score is None:
        assert help_states == []
    else:
        assert len(help_states) == 1 and help_states[0]["score"] == pytest.approx(expected_score)
        assert f"sustained_fraction={sum(pattern) / 4:.3f}" in help_states[0]["evidence"]


def test_needs_help_sustain_uses_four_points_one_and_a_half_seconds_apart(monkeypatch):
    seen = []
    monkeypatch.setattr(hb1, "_fires_base", lambda f, cfg: seen.append(f["n_records"]) or True)
    hb1.analyze(syn.generate("needs-help-like"))
    assert len(seen) == hb1.HB1_CONFIG["needs_help_like"]["sustain_points"] == 4


# --- wording over real outputs --------------------------------------------------

def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(k)
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


def test_wording_over_real_outputs_of_every_scenario():
    seen_states, n_strings = set(), 0
    for name, strength in itertools.product(syn.SCENARIOS, (0.4, 1.0)):
        recs = syn.generate(name, strength=strength)
        for k in range(1, len(recs) + 1, 6):
            out = hb1.analyze(recs[:k])
            assert_clean(out)
            for text in _strings(out):
                n_strings += 1
                assert not BANNED.search(text), (name, k, text)
            for st in out["states"]:
                seen_states.add(st["name"])
                assert st["name"].endswith("-like")
                assert all(re.fullmatch(r"[\w\[\]\- ]+=[-\d.]+", e) for e in st["evidence"])
                assert any(e.startswith("gaze_away[head-orientation proxy]") for e in st["evidence"]) or st["name"] != "unsure-like"
    assert seen_states == set(hb1.STATE_NAMES)  # the scan really covered every state
    assert n_strings > 500


def test_wording_constants():
    assert not BANNED.search(hb1.WORDING)
    assert "signs consistent with" in hb1.WORDING and "unvalidated" in hb1.WORDING
    assert "proxy" in hb1.FEATURE_LABELS["gaze_away"].lower() and "not gaze" in hb1.FEATURE_LABELS["gaze_away"]
    assert "PROXY" in hb1.extract_features.__doc__
    assert "WORDING" not in hb1.analyze(NERV) and hb1.HEURISTIC in hb1.analyze(NERV).values()
