"""HB-1 heuristic state module (EL-006 S1). Synthetic streams only; no media, weights or network."""
import copy
import hashlib
import json
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from socialcue import state as hb1  # noqa: E402
from socialcue import synthetic as syn  # noqa: E402

PINNED_CONFIG_SHA256 = "4940a4cbbb1df3c2af27407e78546e47812c5134b8fcc2aec0daad9e2874b6b3"
CONFIG_PATH = ROOT / "socialcue" / "hb1_config.json"
BANNED = re.compile(r"\b(feels?|feeling|is nervous|is unsure|detected emotion|reads?|diagnos\w*|truth|confirmed)\b", re.I)


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


def _names(result):
    return {s["name"] for s in result["states"]}


def _blank(records, n_usable, tail=24):
    """In the last ``tail`` ticks keep ``n_usable`` evenly spread frames (ends included); the rest become no-face ticks."""
    head, win = list(records[:-tail]), list(records[-tail:])
    n = len(win)
    keep = {round(i * (n - 1) / (n_usable - 1)) for i in range(n_usable)} if n_usable > 1 else ({0} if n_usable else set())
    blanked = [r if i in keep else hb1.FrameRecord(r.t, None, r.quality, 0.0, 0.0, None, None, 0)
               for i, r in enumerate(win)]
    return head + blanked


# --- frozen config -----------------------------------------------------------

def test_config_hash_is_pinned():
    assert hb1.CONFIG_SHA256 == PINNED_CONFIG_SHA256
    assert hb1.config_sha256(hb1.HB1_CONFIG) == PINNED_CONFIG_SHA256


def test_json_file_matches_embedded_config_and_hash():
    on_disk = json.loads(CONFIG_PATH.read_text(encoding="ascii"))
    assert on_disk == hb1.HB1_CONFIG
    assert hb1.config_sha256(on_disk) == PINNED_CONFIG_SHA256
    canonical = json.dumps(on_disk, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert hashlib.sha256(canonical.encode("ascii")).hexdigest() == hb1.CONFIG_SHA256


def test_config_records_every_section_with_a_comment():
    cfg = hb1.HB1_CONFIG
    assert cfg["comment"]
    for section in ("window", "gates", "confidence", "features", "nervous_like", "unsure_like", "needs_help_like"):
        assert cfg[section]["comment"], section
    assert cfg["window"]["seconds"] == 3.0 and cfg["window"]["min_seconds"] == 2.0
    assert cfg["window"]["min_usable_fps"] == 5.0
    assert cfg["gates"]["min_coverage"] == 0.6 and cfg["gates"]["min_quality"] == 0.3
    assert cfg["confidence"]["cap"] == 0.8


def test_changing_a_threshold_changes_the_hash():
    cfg = copy.deepcopy(hb1.HB1_CONFIG)
    cfg["gates"]["min_coverage"] = 0.5
    assert hb1.config_sha256(cfg) != PINNED_CONFIG_SHA256


# --- purity ------------------------------------------------------------------

def test_import_has_no_side_effects_under_poisoned_open_and_socket():
    code = (
        "import sys, json, hashlib, math, copy, random, dataclasses, typing, builtins, io, os, socket\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "before = set(sys.modules)\n"
        "def boom(*a, **k):\n"
        "    raise RuntimeError('side effect at import')\n"
        "builtins.open = boom; io.open = boom; os.open = boom\n"
        "os.listdir = boom; os.scandir = boom; os.makedirs = boom; os.mkdir = boom\n"
        "socket.socket = boom; socket.create_connection = boom; socket.getaddrinfo = boom\n"
        "import socialcue, socialcue.state, socialcue.synthetic\n"
        "new = set(sys.modules) - before\n"
        "bad = [m for m in new if m.split('.')[0] in ('numpy','cv2','torch','onnxruntime','requests','urllib3','http','subprocess','tempfile','socket')]\n"
        "assert not bad, bad\n"
        "r = socialcue.state.analyze(socialcue.synthetic.generate('calm'))\n"
        "assert r['heuristic'] == 'HB-1 hypothesis, unvalidated'\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "OK"


# --- output shape and wording ------------------------------------------------

def test_output_shape_is_exactly_section_4():
    r = hb1.analyze(syn.generate("needs-help-like"))
    assert set(r) == {"states", "insufficient", "window", "heuristic", "proxy_eval_ref"}
    assert set(r["insufficient"]) == {"flag", "reasons"}
    assert set(r["window"]) == {"seconds", "usable_frames", "fps"}
    assert r["heuristic"] == "HB-1 hypothesis, unvalidated"
    assert r["proxy_eval_ref"] is None
    assert r["window"]["seconds"] == 3.0
    for s in r["states"]:
        assert set(s) == {"name", "score", "confidence", "evidence"}
        assert s["name"] in hb1.STATE_NAMES
        assert all(re.fullmatch(r"[\w\[\]\- ]+=[-\d.]+", e) for e in s["evidence"]), s["evidence"]


def test_gaze_is_named_as_a_head_orientation_proxy():
    r = hb1.analyze(syn.generate("unsure-like"))
    gaze = [e for s in r["states"] for e in s["evidence"] if e.startswith("gaze_away")]
    assert gaze and all("head-orientation proxy" in e for e in gaze)
    assert "PROXY" in hb1.HB1_CONFIG["features"]["comment"]


# --- insufficient evidence ---------------------------------------------------

def test_no_state_below_coverage_threshold():
    base = syn.generate("nervous-like")
    assert _names(hb1.analyze(base)) == {"nervous-like"}
    for n_usable in (0, 3, 8, 12, 14):  # 14 / 24 = 0.583 < 0.6
        sub = _blank(base, n_usable)
        assert hb1.extract_features(sub)["coverage"] < 0.6
        r = hb1.analyze(sub)
        assert r["states"] == []
        assert r["insufficient"]["flag"] is True
        assert "low_coverage" in r["insufficient"]["reasons"] or "no_face" in r["insufficient"]["reasons"]


def test_coverage_just_above_gate_is_analysed():
    base = syn.generate("nervous-like")
    sub = _blank(base, 15)  # 15 / 24 = 0.625 >= 0.6 and 15 / 2.875 s = 5.2 usable fps
    f = hb1.extract_features(sub)
    assert 0.6 <= f["coverage"] < 0.65 and f["fps"] >= 5.0
    r = hb1.analyze(sub)
    assert r["insufficient"]["flag"] is False
    assert _names(r) == {"nervous-like"}


def test_low_quality_multiple_faces_no_face_and_filling():
    low = hb1.analyze(syn.generate("insufficient"))
    assert low["insufficient"]["flag"] and "low_quality" in low["insufficient"]["reasons"] and low["states"] == []

    multi = hb1.analyze(syn.generate("multi-face"))
    assert multi["insufficient"]["flag"] and "multiple_faces" in multi["insufficient"]["reasons"] and multi["states"] == []

    empty = hb1.analyze([])
    assert empty["insufficient"]["flag"] and "no_face" in empty["insufficient"]["reasons"] and empty["states"] == []

    early = hb1.analyze([r for r in syn.generate("nervous-like") if r.t <= 1.5])
    assert early["insufficient"]["flag"] and "window_filling" in early["insufficient"]["reasons"] and early["states"] == []


def test_quality_gate_boundary():
    ok = syn.build(8.0, quality=0.31)
    bad = syn.build(8.0, quality=0.25, nerv=lambda t: 1.0)
    assert "low_quality" not in hb1.analyze(ok)["insufficient"]["reasons"]
    r = hb1.analyze(bad)
    assert r["states"] == [] and "low_quality" in r["insufficient"]["reasons"]


def test_dropout_stream_recovers_when_the_face_returns():
    recs = syn.build(14.0, drop=lambda t, u: 5.0 <= t < 8.0)
    assert hb1.analyze([r for r in recs if r.t <= 7.5])["insufficient"]["flag"] is True
    assert hb1.analyze(recs)["insufficient"]["flag"] is False


# --- confidence --------------------------------------------------------------

STRENGTHS = [i / 8 for i in range(9)]


@pytest.mark.parametrize("scenario,name", [("nervous-like", "nervous-like"), ("unsure-like", "unsure-like")])
def test_confidence_is_monotone_in_feature_strength(scenario, name):
    conf = []
    for s in STRENGTHS:
        states = {x["name"]: x for x in hb1.analyze(syn.generate(scenario, strength=s))["states"]}
        conf.append(states[name]["confidence"] if name in states else 0.0)
    assert all(b >= a for a, b in zip(conf, conf[1:])), conf
    assert conf[0] == 0.0 and conf[-1] > 0.0
    assert len(set(conf)) >= 3, conf


def test_confidence_cap_is_08():
    r = hb1.analyze(syn.generate("nervous-like", strength=1.0))
    s = r["states"][0]
    f = hb1.extract_features(syn.generate("nervous-like", strength=1.0))
    assert s["score"] == 1.0
    assert s["score"] * f["coverage"] * f["quality"] > 0.8  # uncapped product exceeds the cap
    assert s["confidence"] == 0.8
    for name in syn.SCENARIOS:
        for strength in STRENGTHS:
            for st in hb1.analyze(syn.generate(name, strength=strength))["states"]:
                assert 0.0 < st["confidence"] <= 0.8
                assert 0.0 < st["score"] <= 1.0


def test_confidence_scales_with_coverage_and_quality():
    full = hb1.analyze(syn.generate("nervous-like", strength=0.75))["states"][0]["confidence"]
    low_q = hb1.analyze(syn.build(8.0, nerv=lambda t: 0.75, quality=0.5))["states"][0]["confidence"]
    assert low_q < full


# --- needs-help-like is a temporal escalation --------------------------------

def test_needs_help_is_never_a_single_frame_output():
    recs = syn.generate("needs-help-like")
    for k in (1, 2, 8, 16, 24):
        assert "needs-help-like" not in _names(hb1.analyze(recs[:k]))
    early = hb1.analyze([r for r in recs if r.t <= 3.0])
    assert early["states"] == []


def test_needs_help_escalates_after_dwell_with_sustained_cue():
    recs = syn.generate("needs-help-like")
    seen = [(r.t, _names(hb1.analyze([x for x in recs if x.t <= r.t]))) for r in recs[::4]]
    first = next(t for t, names in seen if "needs-help-like" in names)
    assert first > 5.0  # dwell gate; the cue itself only starts at 4 s
    assert "needs-help-like" in seen[-1][1] and "unsure-like" in seen[-1][1]


def test_needs_help_requires_dwell():
    recs = syn.build(14.0, uns=lambda t: 1.0, drop=lambda t, u: t < 9.0)
    r = hb1.analyze(recs)
    assert "unsure-like" in _names(r)
    assert "needs-help-like" not in _names(r)


def test_needs_help_requires_looking_around_episodes():
    recs = syn.build(14.0, nerv=lambda t: 1.0 if t >= 4.0 else 0.0)
    names = _names(hb1.analyze(recs))
    assert names == {"nervous-like"}


# --- determinism -------------------------------------------------------------

def test_analyze_is_deterministic_and_order_independent_and_pure():
    recs = syn.generate("needs-help-like")
    snapshot = list(recs)
    a = hb1.analyze(recs)
    assert a == hb1.analyze(recs)
    assert a == hb1.analyze(list(reversed(recs)))
    assert recs == snapshot
    assert json.dumps(a, sort_keys=True) == json.dumps(hb1.analyze(syn.generate("needs-help-like")), sort_keys=True)


def test_custom_config_changes_behaviour_without_touching_the_frozen_one():
    cfg = copy.deepcopy(hb1.HB1_CONFIG)
    cfg["confidence"]["cap"] = 0.5
    r = hb1.analyze(syn.generate("nervous-like"), cfg)
    assert r["states"][0]["confidence"] == 0.5
    assert hb1.analyze(syn.generate("nervous-like"))["states"][0]["confidence"] == 0.8
    assert hb1.config_sha256(hb1.HB1_CONFIG) == PINNED_CONFIG_SHA256


# S21: manufactured review reproductions, no payloads.
def _s21_record(t):
    return hb1.FrameRecord(t, {"fear": 1.0}, .9, .9, 0., (0., 0., 30., 30.),
                           ((0., 0.), (20., 0.), (10., 10.), (2., 15.), (18., 15.)), 1)


def test_s21_trailing_absence_does_not_extend_dwell_or_escalate():
    from dataclasses import replace
    rows = []
    for i in range(49):
        r = _s21_record(i / 8)
        if i >= 40:
            r = replace(r, expression=None, box=None, landmarks=None, face_count=0)
        elif i in (27, 28, 33, 34):
            r = replace(r, landmarks=(r.landmarks[0], r.landmarks[1], (16., 10.), *r.landmarks[3:]))
        rows.append(r)
    assert hb1.extract_features(rows)["dwell"] == 0
    assert "needs-help-like" not in _names(hb1.analyze(rows))
    # Even an allowed trailing gap must not count as observed presence.
    assert hb1.extract_features(rows[:44])["dwell"] == pytest.approx(4.875)


def test_s21_duplicate_instants_cannot_manufacture_temporal_support():
    from dataclasses import replace
    rows = [replace(_s21_record(0), expression=None)] + [_s21_record(2.)] * 12 + [_s21_record(2.1)] * 12
    result = hb1.analyze(rows)
    assert result["insufficient"]["flag"] is True
    assert result["states"] == []
    assert result["window"]["usable_frames"] == 2


def test_s21_distinct_burst_cannot_fill_empty_expected_time_bins():
    from dataclasses import replace
    rows = [replace(_s21_record(0), expression=None)] + [_s21_record(2 + i / 1000) for i in range(24)]
    result = hb1.analyze(rows)
    assert result["insufficient"]["flag"] is True
    assert result["states"] == []


def test_s21_unusable_duplicate_remains_missing_in_either_order():
    from dataclasses import replace
    usable = _s21_record(2.)
    absent = replace(usable, expression=None, box=None, landmarks=None, face_count=0)
    for pair in ([usable, absent], [absent, usable]):
        assert hb1.extract_features(pair)["usable_frames"] == 0


def test_s23_trailing_absent_phase_shift_preserves_gate_outcome():
    from dataclasses import replace
    rows = [replace(_s21_record(0), expression=None)]
    rows += [_s21_record(k * .25 + offset) for k in range(1, 12) for offset in (-.001, .001)]
    outputs = [hb1.analyze(rows + [replace(_s21_record(end), expression=None)]) for end in (3., 3.001)]
    assert outputs[0]["insufficient"] == outputs[1]["insufficient"]
    assert [s["name"] for s in outputs[0]["states"]] == [s["name"] for s in outputs[1]["states"]]


def test_s23_dated_snapshot_and_stalled_dependency_limit_are_documented():
    status = (ROOT / "docs/STATUS.md").read_text(encoding="utf-8")
    docs = (ROOT / "docs/SOCIAL_CUES.md").read_text(encoding="utf-8")
    assert "S23b dated count snapshot" in status
    assert "no request timeout: a never-settling dependency can stall resume" in docs
