"""Stage-2 body-cue features on manufactured skeletons (EL-006 S6b). No pose model, media, camera or dataset."""
import copy
import importlib.util
import json
import math
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from socialcue import body as B  # noqa: E402
from socialcue import state as hb1  # noqa: E402
from socialcue import synthetic as syn  # noqa: E402
from socialcue import synthetic_skeleton as S  # noqa: E402

PINNED_BODY_CONFIG_SHA256 = "c2105b50c32245c8edfacfc3bda2c5020bb6575e92b289eb19ed07d1e79087d9"
NOISE = 0.004


def run(seq, **kw):
    kw.setdefault("consent", True)
    return B.extract_body_cues(seq.t, seq.lm, seq.W, seq.H, **kw)


def present(r):
    return sorted(c["cue"] for c in r["cues"] if c["present"])


def cue(r, name):
    return next(c for c in r["cues"] if c["cue"] == name)


def feats(name, **kw):
    r = run(S.build(name, **kw))
    assert r["status"] == "ok", (name, r["reason"])
    return r["features"]


# --- frozen config -------------------------------------------------------------------------------

def test_body_config_hash_is_pinned_and_hypothesis_only():
    assert B.BODY_CONFIG_SHA256 == PINNED_BODY_CONFIG_SHA256 == B.config_sha256(B.BODY_CONFIG)
    cfg = B.BODY_CONFIG
    assert "NEVER tuned" in cfg["comment"]
    assert cfg["confidence"]["cue_cap"] == 0.6 < cfg["confidence"]["state_cap"] == 0.8
    assert cfg["fidget"]["thr_sw_per_s"] == 0.30 and cfg["self_touch"]["radius_sw"] == 0.45
    changed = copy.deepcopy(cfg)
    changed["fidget"]["thr_sw_per_s"] = 0.31
    assert B.config_sha256(changed) != PINNED_BODY_CONFIG_SHA256


def test_generators_are_the_eleven_named_synthetic_skeletons():
    assert len(S.NAMES) == 11 and set(S.NAMES) == {"calm", "fidget", "self_touch", "freeze", "lean_in", "arms_crossed",
                                                   "occluded", "gesture", "tilt", "desk_crop", "swap"}
    for name in S.NAMES:
        seq = S.build(name)
        assert "SYNTHETIC" in seq.label and len(seq.t) == 30 and all(len(f) == 33 for f in seq.lm)
        assert (seq.W, seq.H) == (640, 480) and seq.t[1] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        S.build("walking")


# --- exact fixture values (analytic expectations confirmed on the first run) ----------------------------

def test_calm_has_no_cue_except_stillness():
    r = run(S.f_calm())
    assert present(r) == ["stillness"]
    f = r["features"]
    for k in ("fidget_energy", "hand_range", "self_touch_frac", "gesture_rate", "lean_scale", "body_motion", "arm_cross_frac", "arm_closure", "head_tilt_deg", "head_lat_sw"):
        assert f[k] == pytest.approx(0.0, abs=1e-9), k
    assert f["head_height_sw"] == pytest.approx(0.55, abs=1e-9)
    assert r["window"]["frames"] == 30 and r["window"]["fps"] == pytest.approx(10.0)
    assert r["window"]["pose_quality"] == pytest.approx(23 / 33, abs=1e-3)


def test_fidget_exact():
    r = run(S.f_fidget())
    f = r["features"]
    # S60: lagged (0.25 s) speed definition and fixture amplitude 0.08 (was 0.06 / 0.5706 / 0.1141 with consecutive-frame speeds)
    assert f["fidget_energy"] == pytest.approx(0.3403, abs=1e-3) and f["hand_range"] == pytest.approx(0.1522, abs=5e-4)
    assert "fidget" in present(r) and f["self_touch_frac"] == 0 and f["gesture_rate"] == 0
    c = cue(r, "fidget")
    assert 0 < c["confidence"] <= 0.6 and c["unit"] == "SW/s"


def test_self_touch_exact():
    r = run(S.f_self_touch())
    f = r["features"]
    assert f["self_touch_frac"] == pytest.approx(0.40) and f["self_touch_run_s"] == pytest.approx(1.1)
    assert "self_touch" in present(r)
    assert f["fidget_energy"] == 0 and f["gesture_rate"] == 0 and f["n_strokes"] == 0  # rise/lower end in contact frames


def test_freeze_exact_and_negative_controls():
    r = run(S.f_freeze())
    f = r["features"]
    assert f["body_motion_a"] == pytest.approx(0.3656, abs=5e-4) and f["body_motion_b"] == pytest.approx(0.0, abs=1e-9)
    assert f["body_motion"] == pytest.approx(0.1765, abs=1e-3)
    assert present(r) == ["freeze_onset"]  # whole-window stillness absent
    assert f["fidget_energy"] == 0 and f["self_touch_frac"] == 0  # the body frame removes translation
    for name in ("calm", "lean_in"):
        g = feats(name)
        assert g["body_motion_a"] <= 0.07 and "freeze_onset" not in present(run(S.build(name)))


def test_lean_in_exact():
    r = run(S.f_lean_in())
    f = r["features"]
    assert f["lean_scale"] == pytest.approx(math.log(1.2), abs=1e-6)
    assert "lean_in" in present(r) and "lean_back" not in present(r)
    assert 0.10 <= f["fidget_energy"] < 0.30 and "fidget" not in present(r)  # design said ~0.148; measured 0.156
    assert f["body_motion"] == pytest.approx(0.054, abs=2e-3)
    assert f["head_tilt_deg"] == 0 and f["gesture_rate"] == 0 and f["self_touch_frac"] == 0


def test_lean_back_is_the_mirror_image():
    r = run(S.build("lean_in", final_scale=1 / 1.2))
    assert present(r).count("lean_back") == 1 and "lean_in" not in present(r)
    assert r["features"]["lean_scale"] == pytest.approx(-math.log(1.2), abs=1e-6)


def test_arms_crossed_exact_and_visibility_variants():
    r = run(S.f_arms_crossed())
    f = r["features"]
    assert f["arm_cross_frac"] == 1.0 and f["arm_closure"] == pytest.approx(1 - math.sqrt(0.17), abs=1e-6)
    assert "arms_crossed" in present(r) and f["fidget_energy"] == 0 and f["self_touch_frac"] == 0
    low = run(S.build("arms_crossed", wrist_vis=0.35))
    assert "arms_crossed" in present(low) and "low_vis=true" in cue(low, "arms_crossed")["evidence"]
    gone = run(S.build("arms_crossed", wrist_vis=0.2))
    c = cue(gone, "arms_crossed")
    assert c["status"] == "none" and c["reason"].startswith("hands_not_visible")


def test_gesture_exact():
    r = run(S.f_gesture())
    f = r["features"]
    assert f["n_strokes"] == 5 and f["gesture_rate"] == pytest.approx(5 / 2.9, abs=1e-6)
    assert f["hand_range"] == pytest.approx(0.856, abs=2e-3) and f["fidget_energy"] == pytest.approx(1.712, abs=0.01)
    assert "fidget" not in present(r)  # hand range exceeds 0.5 SW
    assert any("band=high" in e for e in cue(r, "gesture_rate")["evidence"]) and cue(r, "gesture_rate")["present"] is False


def test_tilt_exact():
    r = run(S.f_tilt())
    f = r["features"]
    assert f["head_tilt_deg"] == pytest.approx(20.0, abs=1e-6) and f["head_lat_sw"] == pytest.approx(0.1368, abs=5e-4)
    assert f["head_height_sw"] == pytest.approx(0.5259, abs=5e-4)
    assert "head_tilt" in present(r) and "head_offset" not in present(r) and "head_low" not in present(r)


# --- noisy bands (measured; the noise generator is not numpy's) --------------------------------------------

@pytest.mark.parametrize("name,checks", [
    ("calm", dict(fidget_energy=(0.0, 0.08), body_motion=(0.05, 0.10), lean_scale=(-0.02, 0.02), head_tilt_deg=(-1, 1), arm_closure=(0, 0.03))),
    ("fidget", dict(fidget_energy=(0.32, 0.42), hand_range=(0.13, 0.19))),
    ("freeze", dict(body_motion_a=(0.33, 0.43), body_motion_b=(0.05, 0.10))),
    ("lean_in", dict(lean_scale=(0.16, 0.20), fidget_energy=(0.10, 0.20))),
    ("arms_crossed", dict(arm_closure=(0.55, 0.62), arm_cross_frac=(1.0, 1.0))),
    ("gesture", dict(gesture_rate=(1.5, 1.9), n_strokes=(5, 5))),
    ("tilt", dict(head_tilt_deg=(19.0, 21.0))),
    ("self_touch", dict(self_touch_frac=(0.39, 0.41), self_touch_run_s=(1.09, 1.11), fidget_energy=(0.0, 0.10), gesture_rate=(0, 0))),
])
def test_noisy_fixtures_stay_in_band(name, checks):
    f = feats(name, noise_sw=NOISE)
    for key, (lo, hi) in checks.items():
        assert lo <= f[key] <= hi, (name, key, f[key])


def test_noisy_present_decisions_match_the_exact_ones():
    expect = {"calm": ["stillness"], "fidget": ["fidget"], "self_touch": ["self_touch"], "freeze": ["freeze_onset"],
              "lean_in": ["lean_in", "stillness"], "arms_crossed": ["arms_crossed", "stillness"], "tilt": ["head_tilt", "stillness"]}
    for name, want in expect.items():
        got = present(run(S.build(name, noise_sw=NOISE)))
        assert sorted(got) == sorted(want), (name, got)
    assert present(run(S.build("gesture", noise_sw=NOISE))) == []


# --- fps, occlusion, partial body, identity ----------------------------------------------------------------------

def test_fps_dependence_of_fidget():
    r20 = run(S.build("fidget", fs=20, n=60))
    f20 = r20["features"]["fidget_energy"]
    # Manufactured 10/20 fps fixture comparison only: fixed-lag energy still has
    # comb-filter and wider rate-dependence limits; this proves no real-motion invariance.
    f10 = run(S.build("fidget"))["features"]["fidget_energy"]
    assert f20 == pytest.approx(0.3762, abs=2e-3) and 0.85 < f20 / f10 < 1.2
    assert "fidget" in present(r20)
    half = run(S.every_other(S.build("fidget")))  # 5 fps
    assert half["status"] == "ok" and cue(half, "fidget")["status"] == "none" and cue(half, "fidget")["reason"] == "fps<8"
    low = run(S.build("calm", fs=4, n=12))  # 4 fps: enough frames and span, but below the 5 fps floor
    assert low["status"] == "none" and low["reason"] == "fps<5"


def test_occluded_hand_cues_are_none_not_garbage():
    r = run(S.f_occluded())
    assert r["status"] == "ok"
    for name in ("fidget", "self_touch", "arms_crossed", "hands_closed", "gesture_rate"):
        c = cue(r, name)
        assert c["status"] == "none" and c["reason"] == "occluded: coverage 0.53", (name, c["reason"])
        assert c["present"] is False and c["value"] is None and c["confidence"] == 0.0
    f = r["features"]
    assert "self_touch_frac" not in f and f["lean_scale"] == 0 and f["head_tilt_deg"] == 0 and f["body_motion"] == pytest.approx(0, abs=1e-9)
    assert present(r) == ["stillness"]
    # a reader that ignored visibility would see the hand on the nose
    ignoring = S.build("occluded")
    ignoring.lm = [[[x, y, 0.99] for x, y, _ in f_] for f_ in ignoring.lm]
    assert "self_touch" in present(run(ignoring))


def test_desk_crop_hides_hand_cues_without_claiming_calm():
    r = run(S.f_desk_crop())
    for name in ("fidget", "self_touch", "arms_crossed", "hands_closed", "gesture_rate"):
        c = cue(r, name)
        assert c["status"] == "none" and c["reason"] == "hands_not_visible" and not c["present"]
    assert r["window"]["pose_quality"] == pytest.approx(15 / 33, abs=1e-3)
    assert cue(r, "head_tilt")["status"] == "ok" and r["features"]["lean_scale"] == 0
    # hidden hands must not support a negative claim
    hb = hb1.analyze(syn.generate("nervous-like", strength=0.55))
    out = B.attach_body(hb, r)
    assert out["states"][0]["body_support"]["delta"] >= 0.0
    # confidences are scaled by pose quality: same cue, lower quality, lower confidence
    full = run(S.f_lean_in())
    dc = S.build("lean_in")
    for frame in dc.lm:
        for i in range(15, 23):
            frame[i][2] = 0.1
    assert 0 < cue(run(dc), "lean_in")["confidence"] < cue(full, "lean_in")["confidence"]


def test_identity_swap_is_rejected_whole():
    r = run(S.f_swap())
    assert r["status"] == "none" and r["reason"] == "identity_discontinuity" and r["cues"] == []


def test_window_gating_reasons():
    seq = S.build("calm")
    assert run(S.SkeletonSeq("x", seq.t[:9], seq.lm[:9], 640, 480))["reason"] == "window_filling"
    assert run(S.SkeletonSeq("x", [t * 0.05 for t in range(30)], seq.lm, 640, 480))["reason"] == "window_filling"  # 1.45 s span
    small = S.scale_about_shoulders(seq, 0.25)  # shoulder width 40 px < 48
    assert run(small)["reason"] == "no_shoulders"
    nobody = S.build("calm")
    for fr in nobody.lm:
        for p in fr:
            p[2] = 0.0
    assert run(nobody)["reason"] == "no_pose"
    noshoulder = S.build("calm")
    for fr in noshoulder.lm:
        fr[11][2] = fr[12][2] = 0.1
    assert run(noshoulder)["reason"] == "no_shoulders"


def test_consent_and_bystander_guards():
    seq = S.build("fidget")
    assert B.extract_body_cues(seq.t, seq.lm, seq.W, seq.H)["reason"] == "consent_off"  # fail closed by default
    assert run(seq, consent=False)["reason"] == "consent_off" and run(seq, consent="yes")["reason"] == "consent_off"
    assert run(seq, face_count=2)["reason"] == "multiple_people"
    assert run(seq, face_box=(230, 52, 120, 120))["status"] == "ok"   # centre 30 px from the pose nose, width 120
    assert run(seq, face_box=(190, 52, 120, 120))["reason"] == "body_face_mismatch"  # 70 px > 0.5 * 120
    assert run(seq, face_box=(0, 0, 0, 0))["reason"] == "body_face_mismatch"


def test_invalid_pixels_and_out_of_range_points_are_not_trusted():
    seq = S.build("self_touch")
    for fr in seq.lm:
        for i in (16, 20, 22):
            fr[i][0] = 2.0  # every hand point the cue reads is far outside [-0.10, 1.10]: invalid even with high visibility
    r = run(seq)
    assert "self_touch" not in present(r)


# --- invariances and properties -----------------------------------------------------------------------------------------

def _compare(a, b, tol=1e-9):
    assert a["status"] == b["status"] == "ok"
    assert a["features"].keys() == b["features"].keys()
    for k in a["features"]:
        assert a["features"][k] == pytest.approx(b["features"][k], abs=tol), k
    assert present(a) == present(b)


@pytest.mark.parametrize("name", ["fidget", "self_touch", "freeze", "lean_in", "arms_crossed", "gesture", "tilt"])
def test_translation_and_scale_invariance(name):
    base = run(S.build(name))
    _compare(base, run(S.translate(S.build(name), 37.0, -21.0)))
    _compare(base, run(S.scale_about_shoulders(S.build(name), 0.5)))
    _compare(base, run(S.scale_about_shoulders(S.build(name), 2.0)))


@pytest.mark.parametrize("name", ["fidget", "self_touch", "arms_crossed"])
def test_left_right_mirror_gives_the_same_values(name):
    _compare(run(S.build(name)), run(S.mirror(S.build(name))))


def test_aspect_trap_canvas_size_does_not_change_features():
    base = run(S.f_self_touch())
    _compare(base, run(S.resize_canvas(S.f_self_touch(), 1280, 480)))
    _compare(base, run(S.resize_canvas(S.f_self_touch(), 640, 960)))


def test_determinism():
    a = S.build("fidget", noise_sw=NOISE)
    assert a.lm == S.build("fidget", noise_sw=NOISE).lm and a.lm != S.build("fidget", noise_sw=NOISE, seed=7).lm
    assert json.dumps(run(a), sort_keys=True) == json.dumps(run(S.build("fidget", noise_sw=NOISE)), sort_keys=True)
    snapshot = copy.deepcopy((a.t, a.lm))
    run(a)
    assert (a.t, a.lm) == snapshot  # inputs are not mutated


def test_monotone_in_amplitude_and_final_scale():
    fe = [feats("fidget", amplitude=a)["fidget_energy"] for a in (0.02, 0.04, 0.08, 0.12)]
    assert all(y > x for x, y in zip(fe, fe[1:])), fe
    ls = [feats("lean_in", final_scale=s)["lean_scale"] for s in (1.05, 1.1, 1.2, 1.3)]
    assert all(y > x for x, y in zip(ls, ls[1:])), ls
    assert feats("lean_in", final_scale=1.05)["lean_scale"] < 0.12 <= feats("lean_in", final_scale=1.2)["lean_scale"]


def test_movement_py_consistency_on_the_freeze_track():
    spec = importlib.util.spec_from_file_location("movement_module", ROOT / "backend" / "app" / "body" / "movement.py")
    movement = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(movement)
    seq = S.f_freeze()
    ids = (0, 11, 12, 13, 14, 15, 16)
    o_ref = [sorted(((f[11][c] * (640, 480)[c] + f[12][c] * (640, 480)[c]) / 2 for f in seq.lm))[len(seq.lm) // 2] for c in (0, 1)]
    track = [[((f[i][0] * 640 - o_ref[0]) / 160.0, (f[i][1] * 480 - o_ref[1]) / 160.0) for i in ids] for f in seq.lm[:15]]
    feats_ = movement.movement_features(track)
    r = run(seq)
    assert feats_["energy"] * 10.0 == pytest.approx(r["features"]["body_motion_a"], abs=1e-9)
    assert "posture" not in json.dumps(r)  # the movement.py posture label is never surfaced


# --- robustness ----------------------------------------------------------------------------------------------------------------

NAN, INF = float("nan"), float("inf")


def _bad_inputs():
    seq = S.build("fidget")
    good = lambda: copy.deepcopy(seq.lm)  # noqa: E731
    out = [("none_t", None, good()), ("none_lm", seq.t, None), ("empty", [], []), ("short_frames", seq.t, [f[:10] for f in good()]),
           ("short_points", seq.t, [[p[:2] for p in f] for f in good()]), ("scalar_frames", seq.t, [5] * 30), ("none_frames", seq.t, [None] * 30),
           ("str_points", seq.t, [["abc"] * 33 for _ in range(30)]), ("t_nan", [NAN] * 30, good()), ("t_inf", [INF] * 30, good()),
           ("t_str", ["a"] * 30, good()), ("length_mismatch", seq.t[:20], good()), ("dict_frames", seq.t, [{} for _ in range(30)])]
    nan_lm = good()
    for f in nan_lm:
        for p in f:
            p[0] = NAN
    out.append(("nan_x", seq.t, nan_lm))
    inf_lm = good()
    for f in inf_lm:
        for p in f:
            p[2] = INF
    out.append(("inf_vis", seq.t, inf_lm))
    return out


@pytest.mark.parametrize("label,t,lm", _bad_inputs(), ids=[b[0] for b in _bad_inputs()])
def test_malformed_input_never_raises(label, t, lm):
    r = B.extract_body_cues(t, lm, 640, 480, consent=True)
    assert r["status"] in {"ok", "none"}
    json.dumps(r, allow_nan=False)
    if r["status"] == "ok":
        assert all(c["status"] == "none" or math.isfinite(c["value"] if c["value"] is not None else 0.0) for c in r["cues"])


@pytest.mark.parametrize("w,h", [(0, 480), (640, -1), (NAN, 480), (None, 480), ("a", "b"), (INF, INF)])
def test_bad_image_size_is_invalid_input(w, h):
    seq = S.build("calm")
    assert B.extract_body_cues(seq.t, seq.lm, w, h, consent=True)["reason"] == "invalid_input"


@pytest.mark.parametrize("fc", [None, "2", NAN, True, [1]])
def test_bad_face_count_is_invalid_input(fc):
    seq = S.build("calm")
    assert B.extract_body_cues(seq.t, seq.lm, 640, 480, consent=True, face_count=fc)["reason"] == "invalid_input"


def test_partial_frames_only_lower_coverage():
    seq = S.build("fidget")
    for i in (3, 9, 15):
        seq.lm[i] = None
    r = run(seq)
    assert r["status"] == "ok" and "fidget" in present(r) and cue(r, "fidget")["coverage"] < 1.0


# --- evidence adapter: items and capped support only -----------------------------------------------------------------------------

def test_evidence_items_carry_the_hypothesis_wording_and_the_cap():
    for name in ("fidget", "self_touch", "lean_in", "arms_crossed", "freeze", "tilt"):
        items = B.body_evidence(run(S.build(name)))
        assert items, name
        for it in items:
            assert it["source"] == "stage2-body" and "proxy/hypothesis, unvalidated" in it["wording"] and 0 < it["confidence"] <= 0.6
            assert set(it) == {"source", "cue", "value", "unit", "confidence", "wording", "evidence", "bystander_checked"}
            assert not re.search(r"\b(nervous|anxious|afraid|fear|stress|needs help|feels?)\b", json.dumps(it), re.I)
    assert B.body_evidence(run(S.f_swap())) == [] and B.body_evidence(None) == [] and B.body_evidence({"status": "ok"}) == []
    assert "stage2" not in B.BODY_WORDING and "unvalidated" in B.BODY_WORDING and "unvalidated" in B.HEURISTIC


def test_cue_text_never_claims_a_state():
    for name in S.NAMES:
        r = run(S.build(name))
        text = json.dumps(r)
        assert not re.search(r"\b(nervous|anxious|afraid|stressed|deceit|lying|needs help)\b", text, re.I), name
        assert "states" not in r


def _nervous(strength=0.55):
    return hb1.analyze(syn.generate("nervous-like", strength=strength))


def test_support_delta_is_bounded_and_only_boosts_present_cues():
    hb = _nervous(0.55)
    base = hb["states"][0]["confidence"]
    out = B.attach_body(hb, run(S.f_fidget()))
    st = out["states"][0]
    assert st["body_support"]["items"] == ["fidget"] and 0 < st["body_support"]["delta"] <= 0.10
    assert st["confidence"] == pytest.approx(min(0.8, base + st["body_support"]["delta"]), abs=1e-4)
    assert "proxy/hypothesis, unvalidated" in st["body_support"]["note"]
    assert hb["states"][0]["confidence"] == base  # input is not mutated
    both = B.support_delta("nervous-like", run(S.build("self_touch")))
    assert both["items"] == ["self_touch"]
    assert B.support_delta("needs-help-like", run(S.f_lean_in()))["items"] == ["lean_in"]
    assert B.support_delta("unsure-like", run(S.f_tilt()))["items"] == ["head_tilt"]
    for name in ("arms_crossed", "gesture", "freeze"):
        assert B.support_delta("unsure-like", run(S.build(name)))["delta"] == 0.0  # mapped to no state


def test_weak_negative_only_for_nervous_with_visible_still_hands():
    hb = _nervous(0.55)
    base = hb["states"][0]["confidence"]
    calm = B.attach_body(hb, run(S.f_calm(), face_box=(230, 52, 120, 120)))  # bystander check ran: the full weak negative
    assert calm["states"][0]["body_support"]["delta"] == -0.05 and calm["states"][0]["confidence"] == pytest.approx(base - 0.05, abs=1e-4)
    for body in (run(S.f_desk_crop()), run(S.f_occluded()), run(S.f_swap()), run(S.f_fidget())):
        assert B.attach_body(hb, body)["states"][0]["body_support"]["delta"] >= 0.0
    unsure = hb1.analyze(syn.generate("unsure-like", strength=0.7))
    assert B.attach_body(unsure, run(S.f_calm()))["states"][0]["body_support"]["delta"] == 0.0  # negative applies to nervous-like only


def test_body_unavailable_means_no_bonus_and_no_penalty():
    hb = _nervous(0.55)
    for body in (run(S.f_swap()), {"status": "none", "reason": "consent_off"}, None, "junk"):
        out = B.attach_body(hb, body)
        sup = out["states"][0]["body_support"]
        assert sup["delta"] == 0.0 and sup["items"] == [] and sup["note"].startswith("body: not available (")
        assert out["states"][0]["confidence"] == hb["states"][0]["confidence"]


def test_body_cues_never_create_a_state_or_rescue_insufficient_evidence():
    insufficient = hb1.analyze(syn.generate("insufficient"))
    assert insufficient["insufficient"]["flag"] and insufficient["states"] == []
    out = B.attach_body(insufficient, run(S.f_fidget()))
    assert out["states"] == [] and out["insufficient"]["flag"] is True
    assert any(c["present"] for c in out["body"]["cues"])  # shown as cues only
    calm = hb1.analyze(syn.generate("calm"))
    assert B.attach_body(calm, run(S.f_fidget()))["states"] == []
    forced = copy.deepcopy(_nervous(1.0))
    forced["insufficient"] = {"flag": True, "reasons": ["multiple_faces"]}
    assert B.attach_body(forced, run(S.f_fidget()))["states"] == []


def test_final_confidence_never_exceeds_the_hb1_cap():
    hb = _nervous(1.0)
    assert hb["states"][0]["confidence"] == 0.8
    boosted = B.attach_body(hb, run(S.f_fidget()))
    assert boosted["states"][0]["confidence"] == 0.8
    for st in B.attach_body(hb1.analyze(syn.generate("needs-help-like")), run(S.f_lean_in()))["states"]:
        assert st["confidence"] <= 0.8
    for body in (run(S.build(n, noise_sw=NOISE)) for n in S.NAMES):
        for st in B.attach_body(_nervous(0.9), body)["states"]:
            assert -0.05 <= st["body_support"]["delta"] <= 0.10 and st["confidence"] <= 0.8


def test_body_block_schema():
    out = B.attach_body(_nervous(), run(S.f_fidget()))
    blk = out["body"]
    assert set(blk) == {"heuristic", "status", "reason", "window", "cues", "bystander_checked", "pinned"}
    assert blk["heuristic"] == "S2 hypothesis, unvalidated; descriptive movement cues" and blk["status"] == "ok" and blk["reason"] is None
    assert set(blk["window"]) == {"seconds", "frames", "fps", "pose_quality"}
    for c in blk["cues"]:
        assert set(c) == {"cue", "value", "unit", "present", "confidence", "coverage", "valid_frames", "evidence", "status", "reason"}
        assert 0.0 <= c["confidence"] <= 0.6
    assert B.attach_body(_nervous(), run(S.f_swap()))["body"]["reason"] == "identity_discontinuity"


# --- purity: no IO, no pose model ----------------------------------------------------------------------------------------------------------

def test_modules_cannot_import_or_call_a_pose_model():
    for path in (ROOT / "socialcue" / "body.py", ROOT / "socialcue" / "synthetic_skeleton.py"):
        src = path.read_text(encoding="utf-8")
        code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith(("#", '"""')))
        for needle in ("import mediapipe", "from mediapipe", "import cv2", "import numpy", "from numpy", "import torch", "import onnx",
                       "estimate_pose", "backend.app", "solutions.pose", "open(", "pathlib", "socket", "requests", "urllib", "subprocess",
                       "os.environ", "getenv", "__import__", "importlib", "eval(", "exec("):
            assert needle not in code, (path.name, needle)


def test_import_has_no_side_effects_and_loads_no_pose_runtime():
    code = (
        "import sys, json, hashlib, math, copy, random, dataclasses, typing, builtins, io, os, socket\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "before = set(sys.modules)\n"
        "def boom(*a, **k):\n"
        "    raise RuntimeError('side effect at import')\n"
        "builtins.open = boom; io.open = boom; os.open = boom\n"
        "os.listdir = boom; os.scandir = boom; os.makedirs = boom; os.mkdir = boom\n"
        "socket.socket = boom; socket.create_connection = boom; socket.getaddrinfo = boom\n"
        "import socialcue.body, socialcue.synthetic_skeleton\n"
        "new = set(sys.modules) - before\n"
        "bad = [m for m in new if m.split('.')[0] in ('numpy','cv2','torch','onnxruntime','mediapipe','requests','urllib3','http','subprocess','tempfile','socket')]\n"
        "assert not bad, bad\n"
        "from socialcue import body, synthetic_skeleton as S\n"
        "seq = S.build('fidget')\n"
        "r = body.extract_body_cues(seq.t, seq.lm, seq.W, seq.H, consent=True)\n"
        "assert r['status'] == 'ok' and 'mediapipe' not in sys.modules\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "OK"


def test_extraction_touches_no_files(monkeypatch, tmp_path):
    import builtins
    writes = []
    real = builtins.open

    def spy(file, mode="r", *a, **k):
        if any(c in str(mode) for c in "wax+"):
            writes.append(file)
        return real(file, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", spy)
    for name in S.NAMES:
        run(S.build(name, noise_sw=NOISE))
    assert writes == [] and list(tmp_path.iterdir()) == []


# --- review S10 hardening (EL-006 S10b) ------------------------------------------------------------------------------

def _hand_override(seq, qpos, frames, vis=0.99):
    """Put all six hand points the self-touch cue reads at one body-frame point (shoulder-width units) for ``frames``."""
    x, y = 0.5 + 0.25 * qpos[0], 5 / 12 + qpos[1] / 3
    for k in frames:
        for i in (15, 16, 19, 20, 21, 22):
            seq.lm[k][i] = [x, y, vis]
    return seq


def test_identity_swap_cannot_hide_behind_invisible_shoulders():
    for hidden in ([14], [15], [14, 15], [13, 14, 15], [15, 16, 17]):
        seq = S.build("swap")
        for k in hidden:
            seq.lm[k][11][2] = seq.lm[k][12][2] = 0.1
        r = run(seq)
        assert r["status"] == "none" and r["reason"] == "identity_discontinuity" and r["cues"] == [], hidden


def test_a_gap_in_the_shoulders_without_a_switch_is_still_analysed():
    seq = S.build("lean_in")
    for k in (10, 11):
        seq.lm[k][11][2] = seq.lm[k][12][2] = 0.1
    r = run(seq)
    assert r["status"] == "ok" and "lean_in" in present(r)
    calm = S.build("calm")
    calm.lm[7][11][2] = 0.1
    assert run(calm)["status"] == "ok"


def test_bystander_checked_is_false_without_a_face_box_and_true_with_one():
    seq = S.build("fidget")
    assert run(seq)["bystander_checked"] is False
    assert run(seq, face_box=(230, 52, 120, 120))["bystander_checked"] is True
    mismatch = run(seq, face_box=(190, 52, 120, 120))
    assert mismatch["reason"] == "body_face_mismatch" and mismatch["bystander_checked"] is True
    for early in (run(seq, consent=False), run(seq, face_count=2), run(S.f_swap(), face_box=(230, 52, 120, 120))):
        assert early["bystander_checked"] is False
    shifted = S.build("calm")
    for fr in shifted.lm:
        for p in fr:
            p[0] += 0.3
    assert run(shifted)["status"] == "ok" and run(shifted)["bystander_checked"] is False  # fails open, but says so
    assert run(shifted, face_box=(230, 52, 120, 120))["reason"] == "body_face_mismatch"


def test_unchecked_evidence_is_flagged_and_capped_lower():
    big = S.build("fidget", amplitude=0.2)  # strong motion, so the checked confidence is above the 0.3 unchecked cap
    body_unchecked = run(big)
    body_checked = run(big, face_box=(230, 52, 120, 120))
    pick = lambda items: next(i for i in items if i["cue"] == "fidget")  # noqa: E731
    u, c = pick(B.body_evidence(body_unchecked)), pick(B.body_evidence(body_checked))
    assert u["bystander_checked"] is False and c["bystander_checked"] is True
    assert u["confidence"] == B.UNCHECKED_CUE_CAP < c["confidence"] <= 0.6
    assert any("bystander: not checked" in e for e in u["evidence"]) and not any("bystander" in e for e in c["evidence"])
    hb = hb1.analyze(syn.generate("nervous-like", strength=0.55))
    du = B.attach_body(hb, body_unchecked)["states"][0]["body_support"]
    dc = B.attach_body(hb, body_checked)["states"][0]["body_support"]
    assert 0 < du["delta"] < dc["delta"] and "bystander not checked" in du["note"]
    assert B.attach_body(hb, body_checked)["body"]["bystander_checked"] is True
    assert "CALLER CONTRACT" in B.__doc__ and "face_box" in B.__doc__


@pytest.mark.parametrize("hb_in,body_in", [
    (None, "ok"), ({"states": [{}]}, "ok"), ({"states": [{"name": "nervous-like", "confidence": "x"}]}, "ok"),
    ("ok", {"status": "ok", "cues": [1]}), ("ok", {"status": "ok", "cues": None}), ("ok", {"status": "ok", "cues": [{"cue": "fidget"}], "features": 5}),
    ({"states": "nope"}, "ok"), ({"states": [None]}, "ok"), ([], "ok"), ("junk", "ok"), ({"states": [{"name": ["x"], "confidence": 0.5}]}, "ok"),
])
def test_attach_body_and_support_delta_never_raise(hb_in, body_in):
    good_body, good_hb = run(S.f_fidget()), hb1.analyze(syn.generate("nervous-like", strength=0.55))
    hb = good_hb if hb_in == "ok" else hb_in
    body = good_body if body_in == "ok" else body_in
    snapshot = copy.deepcopy(hb)
    out = B.attach_body(hb, body)
    assert hb == snapshot  # input untouched
    assert out["body"]["status"] in {"ok", "none"}
    if hb_in != "ok" or body_in != "ok":
        assert out["body"]["reason"] == "invalid_input" and out["body"]["status"] == "none"
        assert all("body_support" not in st for st in out.get("states", []) if isinstance(st, dict))  # unmodified copy
        if isinstance(snapshot, dict):
            assert {k: v for k, v in out.items() if k != "body"} == snapshot
    json.dumps(out, allow_nan=False)
    for state in ("nervous-like", "unsure-like", "needs-help-like", None, ["x"], 5):
        d = B.support_delta(state, body)
        assert set(d) == {"delta", "items", "note"} and -0.05 <= d["delta"] <= 0.10


def test_nan_and_hostile_confidences_cannot_inflate_support():
    body = run(S.f_fidget())
    for c in body["cues"]:
        if c["cue"] == "fidget":
            c["confidence"] = float("nan")
    d = B.support_delta("nervous-like", body)
    assert d["delta"] == 0.0 and B.body_evidence(body)[0]["confidence"] == 0.0
    for c in body["cues"]:
        if c["cue"] == "fidget":
            c["confidence"] = 99.0
    assert B.body_evidence(body)[0]["confidence"] <= B.UNCHECKED_CUE_CAP and B.support_delta("nervous-like", body)["delta"] <= 0.10
    hb = {"states": [{"name": "nervous-like", "confidence": float("nan"), "score": 0.5, "evidence": []}], "insufficient": {"flag": False, "reasons": []}}
    out = B.attach_body(hb, run(S.f_fidget()))
    assert out["body"]["reason"] == "invalid_input"


def test_a_config_that_differs_from_the_pinned_hash_is_refused_unless_explicitly_allowed():
    seq = S.build("fidget")
    loose = copy.deepcopy(B.BODY_CONFIG)
    loose["confidence"]["cue_cap"] = 5.0
    refused = run(seq, config=loose)
    assert refused["status"] == "none" and refused["reason"] == "config_not_pinned" and refused["pinned"] is False and refused["cues"] == []
    allowed = run(seq, config=loose, allow_unpinned=True)
    assert allowed["status"] == "ok" and allowed["pinned"] is False
    assert max(c["confidence"] for c in allowed["cues"]) <= B.MODULE_CUE_CAP == 0.6 and allowed["cues"]
    same = run(seq, config=copy.deepcopy(B.BODY_CONFIG))  # equal content hashes equal: pinned
    assert same["status"] == "ok" and same["pinned"] is True
    assert run(seq)["pinned"] is True
    for junk in ("nope", 5, [], {"x": object()}):
        assert run(seq, config=junk)["reason"] == "config_not_pinned"
        assert run(seq, config=junk, allow_unpinned=True)["status"] == "none"  # still never raises
    assert B.support_delta("nervous-like", run(seq), cfg=loose)["note"] == "body: not available (config_not_pinned)"
    assert B.attach_body(hb1.analyze(syn.generate("nervous-like", strength=0.55)), run(seq), cfg=loose)["states"][0]["body_support"]["delta"] == 0.0


def test_arm_cue_wording_distinguishes_never_visible_from_partly_occluded():
    never = cue(run(S.build("arms_crossed", wrist_vis=0.2)), "arms_crossed")
    assert never["status"] == "none" and never["reason"] == "hands_not_visible"
    partial = S.build("arms_crossed")
    for k in range(5, 20):
        partial.lm[k][15][2] = partial.lm[k][16][2] = 0.2
    c = cue(run(partial), "arms_crossed")
    assert c["status"] == "none" and c["reason"].startswith("occluded: coverage")
    assert "hands_not_visible" in B.__doc__ and "occluded: coverage" in B.__doc__


# --- previously unpinned cues and thresholds ----------------------------------------------------------------------------

def test_head_offset_and_head_low_become_present_on_variants():
    off = run(S.build("tilt", degrees=0.0, head_dx=0.40))
    assert "head_offset" in present(off) and off["features"]["head_lat_sw"] == pytest.approx(0.40, abs=1e-6)
    assert "head_tilt" not in present(off) and "head_low" not in present(off)
    low = run(S.build("tilt", degrees=0.0, head_dy=0.30))
    assert "head_low" in present(low) and low["features"]["head_height_sw"] == pytest.approx(0.25, abs=1e-6)
    assert "head_offset" not in present(low)
    assert "head_offset" in present(run(S.build("tilt", degrees=0.0, head_dx=-0.40)))
    assert "head_tilt" in present(run(S.build("tilt", degrees=-20.0)))
    assert "head_tilt" not in present(run(S.build("tilt", degrees=10.0)))


@pytest.mark.parametrize("kw,cue_name,expected", [
    (dict(degrees=14.0), "head_tilt", False), (dict(degrees=16.0), "head_tilt", True),
    (dict(degrees=0.0, head_dx=0.28), "head_offset", False), (dict(degrees=0.0, head_dx=0.32), "head_offset", True),
    (dict(degrees=0.0, head_dy=0.18), "head_low", False), (dict(degrees=0.0, head_dy=0.22), "head_low", True),
])
def test_head_thresholds_flip_between_the_two_sides_of_each_threshold(kw, cue_name, expected):
    assert (cue_name in present(run(S.build("tilt", **kw)))) is expected


def test_hands_closed_becomes_present_when_clasped_but_not_crossed():
    r = run(S.build("arms_crossed", clasped=True))
    f = r["features"]
    assert f["arm_cross_frac"] == 0.0 and f["arm_closure"] == pytest.approx(0.8, abs=1e-6)
    assert "hands_closed" in present(r) and "arms_crossed" not in present(r)
    assert "hands_closed" not in present(run(S.f_arms_crossed())) and "hands_closed" not in present(run(S.f_calm()))
    assert "weak" in cue(r, "hands_closed")["evidence"]


@pytest.mark.parametrize("point,contact", [
    ((0.44, -0.55), True), ((0.46, -0.55), False),    # radius 0.45 SW around the nose
    ((0.0, -0.16), True), ((0.0, -0.14), False),      # y_max -0.15 SW: close enough but not above the shoulder line
])
def test_self_touch_radius_and_y_max_boundaries(point, contact):
    seq = _hand_override(S.build("calm"), point, range(8, 23))
    r = run(seq)
    assert r["status"] == "ok"
    assert (r["features"]["self_touch_frac"] > 0) is contact, (point, r["features"].get("self_touch_frac"))
    assert ("self_touch" in present(r)) is contact


@pytest.mark.parametrize("n_frames,expected", [(8, False), (9, True)])
def test_self_touch_fraction_threshold_between_8_and_9_contact_frames_of_30(n_frames, expected):
    seq = _hand_override(S.build("calm"), (0.0, -0.40), range(5, 5 + n_frames))
    r = run(seq)
    assert r["features"]["self_touch_frac"] == pytest.approx(n_frames / 30)
    assert ("self_touch" in present(r)) is expected


def test_self_touch_needs_a_contiguous_run_of_half_a_second():
    alternating = _hand_override(S.build("calm"), (0.0, -0.40), range(0, 18, 2))  # 9 contact frames, never adjacent
    r = run(alternating)
    assert r["features"]["self_touch_frac"] == pytest.approx(0.30) and r["features"]["self_touch_run_s"] < 0.5
    assert "self_touch" not in present(r)
    four = run(_hand_override(S.build("calm"), (0.0, -0.40), range(4, 8)))  # contiguous but only 0.4 s
    assert "self_touch" not in present(four)


@pytest.mark.parametrize("amp,expected", [(0.066, False), (0.075, True)])
def test_fidget_threshold_flips_between_the_two_amplitudes(amp, expected):
    r = run(S.build("fidget", amplitude=amp))
    assert abs(r["features"]["fidget_energy"] - 0.30) < 0.03
    assert ("fidget" in present(r)) is expected


@pytest.mark.parametrize("final,expected", [(1.11, False), (1.13, True)])
def test_lean_threshold_flips_between_the_two_scales(final, expected):
    r = run(S.build("lean_in", final_scale=final))
    assert abs(r["features"]["lean_scale"] - 0.12) < 0.02
    assert ("lean_in" in present(r)) is expected


def test_cue_confidence_is_computed_from_module_outputs():
    """Margin/coverage/quality come from the module's own output, not from constants typed into the test."""
    for name, cue_name, value_key, thr in (("fidget", "fidget", "fidget_energy", 0.30), ("self_touch", "self_touch", "self_touch_frac", 0.30),
                                           ("lean_in", "lean_in", "lean_scale", 0.12)):
        r = run(S.build(name))
        c = cue(r, cue_name)
        margin = (r["features"][value_key] - thr) / thr
        qs = min(1.0, r["window"]["pose_quality"] / B.BODY_CONFIG["confidence"]["quality_ref"])
        expected = min(0.6, 0.6 * min(max(margin, 0.0), 1.0) * c["coverage"] * qs)
        assert c["confidence"] == pytest.approx(expected, abs=2e-4), name
        assert 0 < c["confidence"] <= 0.6
    weak = cue(run(S.build("fidget", amplitude=0.075)), "fidget")["confidence"]
    strong = cue(run(S.build("fidget", amplitude=0.12)), "fidget")["confidence"]
    assert 0 < weak < strong <= 0.6


def test_decisions_survive_the_noisy_generator_on_both_sides_of_the_thresholds():
    for amp, expected in ((0.05, False), (0.08, True)):
        assert ("fidget" in present(run(S.build("fidget", amplitude=amp, noise_sw=NOISE)))) is expected
    for point, contact in (((0.40, -0.55), True), ((0.52, -0.55), False)):
        assert ("self_touch" in present(run(_hand_override(S.build("calm", noise_sw=0.0), point, range(8, 23))))) is contact


# --- review S11 residuals (EL-006 S11b) ---------------------------------------------------------------------------------------

def _inflating_config():
    cfg = copy.deepcopy(B.BODY_CONFIG)
    cfg["confidence"].update(state_cap=5.0, delta_max=5.0, delta_per_support=50.0)
    return cfg


def test_an_unpinned_config_can_never_raise_the_state_cap_or_the_delta_bound():
    cfg = _inflating_config()
    assert B.config_sha256(cfg) != B.BODY_CONFIG_SHA256
    hb = hb1.analyze(syn.generate("nervous-like", strength=1.0))
    assert hb["states"][0]["confidence"] == 0.8
    body = run(S.f_fidget(), face_box=(230, 52, 120, 120))
    out = B.attach_body(hb, body, cfg, allow_unpinned=True)
    sup = out["states"][0]["body_support"]
    assert out["states"][0]["confidence"] == 0.8 and B.MODULE_STATE_CAP == 0.8  # old behaviour: 5.0
    assert 0 < sup["delta"] <= B.MODULE_DELTA_MAX == 0.10
    assert B.support_delta("nervous-like", body, cfg, allow_unpinned=True)["delta"] <= 0.10
    lower = hb1.analyze(syn.generate("nervous-like", strength=0.55))
    base = lower["states"][0]["confidence"]
    boosted = B.attach_body(lower, body, cfg, allow_unpinned=True)["states"][0]
    assert boosted["confidence"] == pytest.approx(min(0.8, base + 0.10), abs=1e-4) and boosted["confidence"] <= 0.8


def test_a_lower_state_cap_in_an_allowed_config_is_still_respected():
    cfg = copy.deepcopy(B.BODY_CONFIG)
    cfg["confidence"]["state_cap"] = 0.5
    hb = hb1.analyze(syn.generate("nervous-like", strength=1.0))
    out = B.attach_body(hb, run(S.f_fidget()), cfg, allow_unpinned=True)
    assert out["states"][0]["confidence"] == 0.5


def _poisoned(body):
    bad = copy.deepcopy(body)
    bad["window"] = object()
    bad["extra"] = object()
    for c in bad["cues"]:
        if c["cue"] == "fidget":
            c["value"] = float("nan")
            c["confidence"] = float("inf")
            c["evidence"] = [object(), "ok text", float("-inf")]
            c["coverage"] = {1: float("nan"), "k": (object(),)}
    return bad


def test_attach_body_output_is_strict_json_even_for_a_poisoned_body_block():
    hb = hb1.analyze(syn.generate("nervous-like", strength=0.55))
    good = run(S.f_fidget())
    bad = _poisoned(good)
    snapshot = repr(bad)
    out = B.attach_body(hb, bad)
    json.dumps(out, allow_nan=False)  # old behaviour: TypeError (object) / non-finite numbers
    assert repr(bad) == snapshot  # the caller's block is not mutated
    fid = next(c for c in out["body"]["cues"] if c["cue"] == "fidget")
    assert fid["value"] == 0.0 and fid["confidence"] == 0.0
    assert fid["evidence"] == [B.INVALID_MARKER, "ok text", 0.0]
    assert fid["coverage"] == {"1": 0.0, "k": [B.INVALID_MARKER]}
    assert out["body"]["window"] == B.INVALID_MARKER and "extra" not in out["body"]
    assert out["states"][0]["body_support"]["delta"] <= 0.10  # support from a hostile confidence stays bounded


def test_sanitiser_keeps_clean_blocks_identical_and_handles_hostile_shapes():
    hb = hb1.analyze(syn.generate("nervous-like", strength=0.55))
    for name in S.NAMES:
        body = run(S.build(name))
        blk = B.attach_body(hb, body)["body"]
        assert blk["cues"] == body["cues"] and blk["window"] == body["window"] and blk["reason"] == body["reason"] and blk["status"] == body["status"]
    deep = {"status": "ok", "cues": [{"cue": "x", "evidence": [[[[[[[[[["deep"]]]]]]]]]]}], "features": {}}
    out = B.attach_body(hb, deep)
    json.dumps(out, allow_nan=False)
    nested = {"status": "ok", "cues": [{"cue": "x"}], "features": {}, "reason": {"a": float("nan")}, "window": {"seconds": float("inf")}}
    blk = B.attach_body(hb, nested)["body"]
    assert blk["reason"] == {"a": 0.0} and blk["window"] == {"seconds": 0.0}


def test_non_ok_and_non_dict_bodies_still_pass_through_safely():
    hb = hb1.analyze(syn.generate("nervous-like", strength=0.55))
    weird = {"status": "none", "reason": object(), "window": float("nan"), "cues": []}
    out = B.attach_body(hb, weird)
    json.dumps(out, allow_nan=False)
    assert out["body"]["reason"] == B.INVALID_MARKER and out["body"]["window"] == 0.0
    for body in (None, "junk", 5):
        json.dumps(B.attach_body(hb, body), allow_nan=False)



# S21: steady contact in two observed bursts does not bridge missing time.
def test_s21_self_touch_does_not_bridge_long_gap():
    s = S.build("self_touch")
    ts = [i * .05 for i in range(5)] + [2.7 + i * .05 for i in range(5)]
    result = B.extract_body_cues(ts, [s.lm[15]] * 10, s.W, s.H, consent=True)
    assert result["status"] == "ok"
    assert result["features"]["self_touch_run_s"] == pytest.approx(.2)
    assert not cue(result, "self_touch")["present"]


@pytest.mark.parametrize("count,reason", [(0, "no_subject"), (-1, "invalid_input"), (.5, "invalid_input")])
def test_s21_body_requires_one_integer_face(count, reason):
    result = run(S.build("calm"), face_count=count)
    assert result["status"] == "none"
    assert result["reason"] == reason
    assert result["cues"] == []


def test_s21_self_touch_splits_nonpositive_steps():
    s = S.build("self_touch")
    times = [i * .1 for i in range(30) for _ in range(2)]
    result = B.extract_body_cues(times, [s.lm[15]] * len(times), s.W, s.H, consent=True)
    assert result["status"] == "ok"
    assert result["features"]["self_touch_run_s"] == pytest.approx(.1)
    assert not cue(result, "self_touch")["present"]


def test_s21_observed_contiguous_half_second_still_meets_frozen_touch_rule():
    s = S.build("self_touch")
    times = [i * .1 for i in range(30)]
    landmarks = [s.lm[15] if i < 6 or 20 <= i < 26 else s.lm[0] for i in range(30)]
    result = B.extract_body_cues(times, landmarks, s.W, s.H, consent=True)
    assert result["features"]["self_touch_run_s"] == pytest.approx(.5)
    assert cue(result, "self_touch")["present"]


# --- S60 (review S59 findings 1, 3, 4, 7, 8; manufactured data only) ---------------------------------------------------------------

def _calm_noise(fs, sigma=0.01, seed=505):
    return S.build("calm", fs=fs, n=int(3 * fs) + 1, noise_sw=sigma, seed=seed)


def test_s60_f1_noise_alone_gives_comparable_fidget_energy_at_every_frame_rate():
    energies = {fs: run(_calm_noise(fs))["features"]["fidget_energy"] for fs in (8, 10, 20, 30)}
    assert max(energies.values()) / min(energies.values()) < 1.6, energies  # before: 0.195 -> 0.778, a 4x spread
    for fs in (20, 30):
        r = run(_calm_noise(fs))
        assert "fidget" not in present(r), (fs, energies[fs])  # before: present at 20 and 30 fps, confidence up to the cap
        assert cue(r, "fidget")["confidence"] == 0.0
    for fs in (8, 10):
        assert "fidget" not in present(run(_calm_noise(fs)))


def test_s60_f1_more_seeds_and_larger_noise_do_not_scale_with_frame_rate():
    for seed in (1, 7, 505):
        e8 = run(_calm_noise(8, 0.02, seed))["features"]["fidget_energy"]
        e30 = run(_calm_noise(30, 0.02, seed))["features"]["fidget_energy"]
        assert 0.6 < e30 / e8 < 1.7, (seed, e8, e30)


def test_s60_f1_real_motion_still_reads_present_at_native_and_other_rates():
    for fs in (8, 10, 20, 30):
        r = run(S.build("fidget", fs=fs, n=3 * fs + 1))
        assert "fidget" in present(r), (fs, r["features"]["fidget_energy"])
    assert "fidget" not in present(run(S.build("calm")))


def test_s60_f1_lag_is_a_documented_module_constant_and_the_hash_is_unchanged():
    assert B.FIDGET_LAG_S == 0.25 and "FIDGET_LAG_S" in B.__doc__
    assert B.BODY_CONFIG_SHA256 == PINNED_BODY_CONFIG_SHA256
    assert "feature DEFINITION" in B.__doc__ or "feature definition" in B.__doc__


def test_s60_f3_documented_support_maxima_are_the_real_values():
    cues = []
    for name in ("fidget", "self_touch", "freeze_onset", "head_tilt", "lean_in"):
        cues.append({"cue": name, "status": "ok", "present": True, "confidence": 0.6, "coverage": 1.0, "evidence": [], "value": 1.0, "unit": ""})
    body = {"status": "ok", "reason": None, "window": {}, "cues": cues, "features": {}, "bystander_checked": True, "pinned": True}
    got = {s: B.support_delta(s, body)["delta"] for s in ("nervous-like", "unsure-like", "needs-help-like")}
    assert got == {"nervous-like": 0.0375, "unsure-like": 0.015, "needs-help-like": 0.015}
    assert B.MAX_POSITIVE_SUPPORT == got  # the module states what is reachable
    doc = " ".join(B.support_delta.__doc__.split())
    assert "0.0375" in doc and "0.015" in doc and "safety" in doc and "+0.10" in doc


def test_s60_f4_negative_delta_is_scaled_down_when_the_bystander_check_did_not_run():
    calm = run(S.f_calm())
    unchecked = B.support_delta("nervous-like", calm)
    checked = B.support_delta("nervous-like", run(S.f_calm(), face_box=(230, 52, 120, 120)))
    assert checked["delta"] == -0.05
    assert unchecked["delta"] == -0.025  # scaled by UNCHECKED_CUE_CAP / MODULE_CUE_CAP (before: the full -0.05)
    assert abs(unchecked["delta"]) < abs(checked["delta"]) and "bystander not checked" in unchecked["note"]
    hb = hb1.analyze(syn.generate("nervous-like", strength=0.55))
    assert B.attach_body(hb, calm)["states"][0]["body_support"]["delta"] == -0.025


def _elbows_hidden(seq, frames):
    for k in frames:
        seq.lm[k][13][2] = seq.lm[k][14][2] = 0.1
    return seq


def test_s60_f7_freeze_onset_needs_enough_used_pairs_in_each_half():
    full = run(S.f_freeze())
    assert "freeze_onset" in present(full)
    sparse = run(_elbows_hidden(S.f_freeze(), range(0, 11)))  # overall coverage ~0.62, first half keeps 3 of 14 pairs
    c = cue(sparse, "freeze_onset")
    assert c["status"] == "none" and c["present"] is False and "first half" in c["reason"], c
    assert "stillness" in [x["cue"] for x in sparse["cues"]]  # the whole-window cue is unaffected
    late = run(_elbows_hidden(S.f_freeze(), range(20, 30)))
    assert cue(late, "freeze_onset")["status"] == "none" and "last half" in cue(late, "freeze_onset")["reason"]
    mild = run(_elbows_hidden(S.f_freeze(), [3, 20]))  # a few dropped frames still leave both halves usable
    assert "freeze_onset" in present(mild)


def test_s60_f8_fidget_fps_floor_interaction_is_documented_and_behaves_as_documented():
    doc = " ".join(B.__doc__.split())
    assert "median frame interval above 0.125 s" in doc and "fps<8" in doc
    def native(dt):
        seq = S.build("fidget", n=30)
        seq.t = [k * dt for k in range(30)]
        return run(seq)
    assert cue(native(0.125), "fidget")["status"] == "ok"          # exactly 8 fps: available
    slower = cue(native(0.13), "fidget")                           # a request period just above 125 ms
    assert slower["status"] == "none" and slower["reason"] == "fps<8"
