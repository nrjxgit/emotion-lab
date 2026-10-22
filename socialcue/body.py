"""Stage-2 body-cue features F1-F7 on a MediaPipe-33 landmark sequence (EL-006 S6b). Research prototype.

PURE module: stdlib only, no IO at import, no filesystem, no network, no pose model. It never imports or
calls mediapipe/cv2/numpy and has no way to run a pose model: it only measures landmark sequences that a
caller hands in (tests use manufactured skeletons). The backend consumes it only behind socialcue=1; the UI has no body row.

The cues are descriptive movement measurements. They are hypotheses with no validation, they never create
an HB-1 state, and they cannot turn an INSUFFICIENT Stage-1 window into a state (see ``attach_body``).
Every threshold is a hypothesis frozen in BODY_CONFIG (canonical sha256 pinned in a test); none was tuned
on real data. Design: receipts/EL-006-socialcue/S6/stage2-design.md.

Self-touch duration is observed first-to-last contiguous contact time (no extra median frame).
Since S21 it needs >=0.5 s observed contact: at 8 FPS, four observations span only 0.375 s
(previously credited as 0.5 s). This changes sensitivity without changing thresholds; the BODY
config hash cannot detect code-level feature-definition changes. Stage 2 is wired behind the research flag; no pose model is installed or validated by this wiring.

HB-1 coverage (not this module's body pair-count coverage) credits distinct 125 ms bins;
HB-1 window.fps is a bin-credit rate, not measured usable FPS. S22 review synthetic
200-seed timing probes: uniform 90-160 ms mean coverage 0.995 -> 0.933 (minimum
0.917 -> 0.792), 60-190 ms mean 0.984 -> 0.889; insufficiency rates unchanged.
Coverage-scaled live confidence may run a few percent below perfect-timing synthetic
streams; this is a synthetic timing inference, not real-camera validation/calibration.

CALLER CONTRACT (including the flagged backend wiring):
- ``face_count`` must be a strict Python int (not bool): 1.0 and numpy integer scalars are
  ``invalid_input``; callers must validate then cast to int first. Only 1 admits body evidence;
  0 gives ``no_subject``, negative/malformed values ``invalid_input``, and ints >1 ``multiple_people``.
- Pass ``consent=True`` only after the separate body-data consent; the default fails closed (``consent_off``).
- Pass the real Stage-1 ``face_count`` and the Stage-1 ``face_box`` (px). Without ``face_box`` the bystander /
  identity face-match does NOT run: the output says ``bystander_checked: false``, ``body_evidence`` flags those
  items and caps their confidence at UNCHECKED_CUE_CAP, and ``support_delta`` uses the same cap. A caller that
  cannot supply a face box must not present body cues as belonging to the consenting person.
- A config other than BODY_CONFIG is refused (``config_not_pinned``) unless ``allow_unpinned=True``; such output
  carries ``pinned: false`` and every cue confidence is capped at MODULE_CUE_CAP.
- ``attach_body``/``support_delta`` never raise: malformed input returns the unmodified HB-1 copy plus an
  ``invalid_input`` body block (delta 0).

Fidget energy (feature DEFINITION changed in S60, review S59 finding 1, under an UNCHANGED BODY config hash and unchanged
thresholds): the F1 speed is measured over a fixed time lag FIDGET_LAG_S = 0.25 s (the earlier hand position is linearly
interpolated at t - 0.25 s between the two frames around it), not over consecutive frames, so independent per-frame
landmark jitter no longer produces a speed that grows with the frame rate. Values are not comparable with S6b-S59 numbers.
Gesture strokes (F7) keep consecutive-pair speeds. The BODY hash covers thresholds, not this code-level definition.

Limits of the lagged definition (review S61 findings A, B, D; synthetic numbers on the fidget fixture at amplitude 0.08, no real data): displacement over a fixed 0.25 s lag is a comb filter, so it is nulled for motion at multiples of 4 Hz and attenuated near them (20 fps, 4 Hz: energy 0.000, was 1.141; 10 fps, 3 Hz: 0.246, not present, was 0.571; 10 fps, 4 Hz: 0.130, was 0.923). The effective detection bar for real motion at the shipped rate is higher than before: at 2 Hz and 8 fps the energy is 0.32 against the unchanged 0.30 threshold (confidence 0.04), and the fixture amplitude had to be raised 0.06 -> 0.08 to stay above it. Rate-invariance holds for independent landmark noise only; synthetic 2 Hz motion still moves 0.32 -> 0.47 from 8 to 30 fps. body_motion, stillness and freeze_onset still depend on frame rate (calm noise sigma 0.004 body_motion 0.054 at 8 fps, 0.211 at 30 fps; stillness present at 8/10 fps and absent at 20/30 for pure noise), so offline replay above 8 fps is not comparable. No real-motion calibration exists, and the Stage-2 motion definition will be revisited when real pose data exists. Fixture numbers in older receipts (fidget_energy, stillness, body_motion, hand_range) are not comparable with S60 and later.

Fidget needs a median frame interval of at most 0.125 s (>= 8 fps, ``min_fps_fidget``). The shipped client period is 120 ms
minus request time, so any median frame interval above 0.125 s, which real inference latency can easily cause, makes the
fidget cue unavailable with reason ``fps<8``; the floor lives in the frozen BODY config and was not changed.

Body support is bounded by the frozen weights, not by the module clamp: the largest reachable positive confidence delta
(every supporting cue present at the cue cap, bystander checked) is MAX_POSITIVE_SUPPORT = nervous-like +0.0375,
unsure-like +0.015, needs-help-like +0.015. The +0.10 clamp is a safety rail that the frozen config cannot reach.

Wording note: when no frame has both wrists visible the arm cues report ``hands_not_visible``; when some do but
coverage is below 0.6 they report ``occluded: coverage X``. (The design said ``occluded`` for wrist vis 0.2; this
module distinguishes "never visible" from "partly occluded".)
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Optional

HEURISTIC = "S2 hypothesis, unvalidated; descriptive movement cues"
BODY_WORDING = "proxy/hypothesis, unvalidated: descriptive movement cue, not an inner state"

# MediaPipe-33 indices used here (standard BlazePose layout; not verified against a primary page).
NOSE, EAR_L, EAR_R = 0, 7, 8
SH_L, SH_R, EL_L, EL_R, WR_L, WR_R = 11, 12, 13, 14, 15, 16
INDEX_L, INDEX_R, THUMB_L, THUMB_R = 19, 20, 21, 22
N_LANDMARKS = 33
MODULE_CUE_CAP = 0.6  # frozen cue cap, enforced even when an unpinned config is explicitly allowed
UNCHECKED_CUE_CAP = 0.3  # cap for evidence whose bystander/face match was not run
MODULE_STATE_CAP = 0.8  # HB-1 state cap, enforced by attach_body whatever config is passed
MODULE_DELTA_MAX = 0.10  # body support bounds, enforced whatever config is passed
MODULE_DELTA_MIN = -0.05
FIDGET_LAG_S = 0.25  # fixed lag for the F1 speed (see the module docstring); a code-level definition, not a BODY config key

BODY_CONFIG = {
    "name": "S2-CFG",
    "version": 1,
    "comment": "Hypothesis thresholds from stage2-design.md section 1.4; NEVER tuned on real data; synthetic skeletons only.",
    "window": {"seconds": 3.0, "hop_seconds": 1.0, "min_frames": 10, "min_span_s": 2.0, "min_fps": 5.0,
               "min_fps_fidget": 8.0, "max_pair_dt_s": 0.5},
    "validity": {"v_min": 0.5, "v_min_arms": 0.3, "xy_min": -0.10, "xy_max": 1.10},
    "scale": {"s_min_px": 48.0, "s_valid_frac_min": 0.6, "disc_s_ratio_lo": 0.75, "disc_s_ratio_hi": 1.33, "disc_o_jump_sw": 1.0},
    "coverage": {"cue_min": 0.6, "pairs_min": 0.5, "min_pairs": 8},
    "fidget": {"thr_sw_per_s": 0.30, "hand_range_max_sw": 0.50},
    "self_touch": {"radius_sw": 0.45, "y_max_sw": -0.15, "frac_min": 0.30, "run_min_s": 0.5, "run_min_frames": 3},
    "gesture": {"stroke_v_sw_per_s": 1.0, "stroke_net_sw": 0.5, "stroke_min_pairs": 2, "band_low": 0.3, "band_high": 1.0},
    "lean": {"thr_ln": 0.12},
    "motion": {"m_act": 0.30, "m_still": 0.12},
    "arms": {"cross_x_margin_sw": 0.10, "cross_y_lo_sw": 0.2, "cross_y_hi_sw": 1.2, "cross_frac_min": 0.70, "closure_min": 0.5},
    "head": {"tilt_deg": 15.0, "lat_sw": 0.30, "low_sw": 0.35},
    "face_match": {"max_dist_over_box_width": 0.5},
    "confidence": {"cue_cap": 0.6, "quality_ref": 0.6, "state_cap": 0.8, "delta_max": 0.10, "delta_per_support": 0.05,
                   "delta_negative": -0.05, "negative_fidget_max": 0.15},
    "support": {
        "nervous-like": {"fidget": 0.5, "self_touch": 0.5, "freeze_onset": 0.25},
        "unsure-like": {"head_tilt": 0.25, "self_touch": 0.25},
        "needs-help-like": {"lean_in": 0.25, "self_touch": 0.25},
    },
}


def canonical_json(cfg) -> str:
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_sha256(cfg) -> str:
    return hashlib.sha256(canonical_json(cfg).encode("ascii")).hexdigest()


BODY_CONFIG_SHA256 = config_sha256(BODY_CONFIG)
MAX_POSITIVE_SUPPORT = {
    state: round(min(BODY_CONFIG["confidence"]["delta_max"], BODY_CONFIG["confidence"]["delta_per_support"]
                     * sum(weights.values()) * BODY_CONFIG["confidence"]["cue_cap"]), 4)
    for state, weights in BODY_CONFIG["support"].items()
}


# --- small numeric helpers ------------------------------------------------------------

def _fin(*v) -> bool:
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v)


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _pct(xs, q):
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def _runs(flags):
    """Maximal runs of True as (start, end) inclusive index pairs."""
    out, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


# --- frame preparation -----------------------------------------------------------------

class _Frame:
    __slots__ = ("t", "pts", "S", "O")

    def __init__(self, t, pts):
        self.t, self.pts, self.S, self.O = t, pts, None, None


def _parse_frame(f, W, H, cfg):
    """33 entries of (px, py, vis, in_range) or None; a malformed frame is all-None (never raises)."""
    out = [None] * N_LANDMARKS
    try:
        if len(f) != N_LANDMARKS:
            return out
        lo, hi = cfg["validity"]["xy_min"], cfg["validity"]["xy_max"]
        for i in range(N_LANDMARKS):
            x, y, v = f[i]
            if _fin(x, y, v):
                out[i] = (x * W, y * H, float(v), lo <= x <= hi and lo <= y <= hi)
    except Exception:
        return [None] * N_LANDMARKS
    return out


def _ok(fr, i, vmin):
    p = fr.pts[i]
    return p is not None and p[3] and p[2] >= vmin


class _Ctx:
    """Window context: frames, scale, reference origin, helpers for body/global-frame coordinates."""

    def __init__(self, frames, cfg):
        self.f, self.cfg = frames, cfg
        self.v = cfg["validity"]["v_min"]
        self.va = cfg["validity"]["v_min_arms"]
        self.n = len(frames)
        self.t = [fr.t for fr in frames]
        for fr in frames:
            if _ok(fr, SH_L, self.v) and _ok(fr, SH_R, self.v):
                a, b = fr.pts[SH_L], fr.pts[SH_R]
                fr.S = math.hypot(a[0] - b[0], a[1] - b[1])
                fr.O = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        sv = [fr.S for fr in frames if fr.S is not None]
        self.s_valid_frac = len(sv) / self.n if self.n else 0.0
        self.S_ref = _median(sv)
        ov = [fr.O for fr in frames if fr.O is not None]
        self.O_ref = (_median([o[0] for o in ov]), _median([o[1] for o in ov])) if ov else None

    def q(self, k, i, vmin=None):
        """Body-frame coordinate (shoulder-width units, origin = shoulder midpoint) or None."""
        fr = self.f[k]
        if fr.O is None or not _ok(fr, i, self.v if vmin is None else vmin):
            return None
        p = fr.pts[i]
        return ((p[0] - fr.O[0]) / self.S_ref, (p[1] - fr.O[1]) / self.S_ref)

    def g(self, k, i):
        fr = self.f[k]
        if not _ok(fr, i, self.v):
            return None
        p = fr.pts[i]
        return ((p[0] - self.O_ref[0]) / self.S_ref, (p[1] - self.O_ref[1]) / self.S_ref)

    def pair_ok(self, k):
        dt = self.t[k] - self.t[k - 1]
        return dt if 0 < dt <= self.cfg["window"]["max_pair_dt_s"] else None


def _fail(reason, window=None, checked=False):
    return {"heuristic": HEURISTIC, "status": "none", "reason": reason, "window": window, "cues": [], "features": {},
            "bystander_checked": checked, "pinned": True}


# --- cue construction ------------------------------------------------------------------------

def _cue(name, value, unit, present, margin, coverage, valid, quality_scale, cfg, evidence=(), reason=None, status="ok"):
    if status != "ok":
        return {"cue": name, "value": None, "unit": unit, "present": False, "confidence": 0.0, "coverage": round(coverage, 4),
                "valid_frames": valid, "evidence": [], "status": "none", "reason": reason}
    conf = 0.0
    if present:
        conf = cfg["confidence"]["cue_cap"] * _clamp(margin, 0.0, 1.0) * coverage * quality_scale
    return {"cue": name, "value": None if value is None else round(float(value), 6), "unit": unit, "present": bool(present),
            "confidence": round(min(cfg["confidence"]["cue_cap"], conf), 4), "coverage": round(coverage, 4),
            "valid_frames": valid, "evidence": list(evidence), "status": "ok", "reason": None}


def _unavailable(names, unit_map, reason, coverage, valid, cfg):
    return [_cue(n, None, unit_map.get(n, ""), False, 0.0, coverage, valid, 1.0, cfg, reason=reason, status="none") for n in names]


def _hand_reason(valid, cov, pair_cov, n_pairs, cfg, need_pairs=True):
    if valid == 0:
        return "hands_not_visible"
    c = cfg["coverage"]
    if cov < c["cue_min"]:
        return f"occluded: coverage {cov:.2f}"
    if need_pairs and pair_cov < c["pairs_min"]:
        return f"occluded: coverage {pair_cov:.2f}"
    if need_pairs and n_pairs < c["min_pairs"]:
        return "too_few_pairs"
    return None


# --- main extraction -----------------------------------------------------------------------------

def _is_pinned(cfg) -> bool:
    try:
        return config_sha256(cfg) == BODY_CONFIG_SHA256
    except Exception:
        return False


def extract_body_cues(t, lm, width, height, *, consent: bool = False, face_count: int = 1, face_box=None, config=None,
                      allow_unpinned: bool = False) -> dict:
    """Window features at the last frame time. Never raises; any failure is a status 'none' block.

    t: capture timestamps in seconds; lm: per frame 33 x (x_norm, y_norm, visibility); width/height: image size in px.
    face_box: (x, y, w, h) px of the Stage-1 face for the bystander/identity guard. If it is None the guard does not
    run and the output has ``bystander_checked: false`` (see the caller contract in the module docstring).
    config: must hash to BODY_CONFIG_SHA256 unless ``allow_unpinned=True`` (then ``pinned: false``, confidences capped).
    """
    cfg = BODY_CONFIG if config is None else config
    pinned = config is None or _is_pinned(cfg)
    if not pinned and allow_unpinned is not True:
        out = _fail("config_not_pinned")
        out["pinned"] = False
        return out
    try:
        out = _extract(t, lm, width, height, consent, face_count, face_box, cfg)
    except Exception:
        out = _fail("invalid_input")
    out["pinned"] = pinned
    if not pinned:
        for c in out.get("cues", []):
            c["confidence"] = min(MODULE_CUE_CAP, c["confidence"])
    return out


def _extract(t, lm, width, height, consent, face_count, face_box, cfg):
    if consent is not True:
        return _fail("consent_off")
    if not _fin(width, height) or width <= 0 or height <= 0:
        return _fail("invalid_input")
    if isinstance(face_count, bool) or not isinstance(face_count, int) or face_count < 0:
        return _fail("invalid_input")
    if face_count == 0:
        return _fail("no_subject")
    if face_count > 1:
        return _fail("multiple_people")
    if t is None or lm is None:
        return _fail("no_pose")
    frames = []
    for tk, f in zip(list(t), list(lm)):
        if _fin(tk):
            frames.append(_Frame(float(tk), _parse_frame(f, width, height, cfg)))
    frames.sort(key=lambda fr: fr.t)
    if not frames:
        return _fail("no_pose")
    wc = cfg["window"]
    t_end = frames[-1].t
    frames = [fr for fr in frames if t_end - wc["seconds"] < fr.t <= t_end]
    v = cfg["validity"]["v_min"]
    n = len(frames)
    if not any(any(_ok(fr, i, v) for i in range(N_LANDMARKS)) for fr in frames):
        return _fail("no_pose")
    dts = [frames[k].t - frames[k - 1].t for k in range(1, n) if frames[k].t - frames[k - 1].t > 0]
    span = frames[-1].t - frames[0].t
    fps = 1.0 / _median(dts) if dts else 0.0
    quality = _mean([sum(1 for i in range(N_LANDMARKS) if _ok(fr, i, v)) / N_LANDMARKS for fr in frames])
    window = {"seconds": wc["seconds"], "frames": n, "fps": round(fps, 3), "pose_quality": round(quality, 4)}
    if n < wc["min_frames"] or span < wc["min_span_s"] - 1e-9:
        return _fail("window_filling", window)
    if fps < wc["min_fps"]:
        return _fail("fps<5", window)
    ctx = _Ctx(frames, cfg)
    sc = cfg["scale"]
    if ctx.s_valid_frac < sc["s_valid_frac_min"] or ctx.S_ref is None or ctx.S_ref < sc["s_min_px"]:
        return _fail("no_shoulders", window)
    prev = None  # compare with the previous VALID frame so hidden shoulders cannot hide a person switch
    for k in range(n):
        b = frames[k]
        if b.S is None:
            continue
        if prev is not None:
            ratio = b.S / prev.S
            jump = math.hypot(b.O[0] - prev.O[0], b.O[1] - prev.O[1]) / ctx.S_ref
            if not (sc["disc_s_ratio_lo"] <= ratio <= sc["disc_s_ratio_hi"]) or jump > sc["disc_o_jump_sw"]:
                return _fail("identity_discontinuity", window)
        prev = b
    checked = face_box is not None
    if face_box is not None:
        try:
            bx, by, bw, bh = face_box
            noses = [fr.pts[NOSE] for fr in frames if _ok(fr, NOSE, v)]
            if not noses or not _fin(bx, by, bw, bh) or bw <= 0:
                return _fail("body_face_mismatch", window, True)
            nx, ny = _median([p[0] for p in noses]), _median([p[1] for p in noses])
            if math.hypot(nx - (bx + bw / 2.0), ny - (by + bh / 2.0)) > cfg["face_match"]["max_dist_over_box_width"] * bw:
                return _fail("body_face_mismatch", window, True)
        except Exception:
            return _fail("body_face_mismatch", window, True)

    qs = min(1.0, quality / cfg["confidence"]["quality_ref"])
    features, cues = {}, []

    # --- F1 fidget energy + hand range, F7 gesture strokes (share the pair speeds) --------------
    hand_pts = {"L": (WR_L, INDEX_L), "R": (WR_R, INDEX_R)}
    pairs = []  # (k, s_k, {side: speed})
    for k in range(1, n):
        dt = ctx.pair_ok(k)
        if dt is None:
            continue
        sides = {}
        for side, ids in hand_pts.items():
            sp = []
            for i in ids:
                a, b = ctx.q(k - 1, i), ctx.q(k, i)
                if a is not None and b is not None:
                    sp.append(math.hypot(b[0] - a[0], b[1] - a[1]) / dt)
            if sp:
                sides[side] = _mean(sp)
        if sides:
            pairs.append((k, max(sides.values()), sides))
    lag_pairs = []  # (k, s_k, {side: speed}) over the fixed lag FIDGET_LAG_S, interpolating the earlier position
    j1 = 0
    for k in range(1, n):
        target = ctx.t[k] - FIDGET_LAG_S
        while j1 + 1 < k and ctx.t[j1 + 1] <= target:
            j1 += 1
        j2 = j1 + 1
        if ctx.t[j1] > target or j2 >= k:
            continue
        gap = ctx.t[j2] - ctx.t[j1]
        if not (0 < gap <= wc["max_pair_dt_s"]):
            continue
        w = (target - ctx.t[j1]) / gap
        sides = {}
        for side, ids in hand_pts.items():
            sp = []
            for i in ids:
                a, b, c = ctx.q(j1, i), ctx.q(j2, i), ctx.q(k, i)
                if a is not None and b is not None and c is not None:
                    sp.append(math.hypot(c[0] - (a[0] + w * (b[0] - a[0])), c[1] - (a[1] + w * (b[1] - a[1]))) / FIDGET_LAG_S)
            if sp:
                sides[side] = _mean(sp)
        if sides:
            lag_pairs.append((k, max(sides.values()), sides))
    hand_ids = (WR_L, WR_R, INDEX_L, INDEX_R)
    valid_hand = sum(1 for k in range(n) if frames[k].O is not None and any(ctx.q(k, i) is not None for i in hand_ids))
    cov_h = valid_hand / n
    pair_cov = len(pairs) / (n - 1)
    hreason = _hand_reason(valid_hand, cov_h, pair_cov, len(pairs), cfg)
    f1_units = {"fidget": "SW/s", "gesture_rate": "strokes/s"}
    touch_flags = [False] * n  # filled by F2 below, used by F7
    fc = cfg["fidget"]

    # --- F2 self-touch --------------------------------------------------------------------------
    st = cfg["self_touch"]
    touch_ids = (WR_L, WR_R, INDEX_L, INDEX_R, THUMB_L, THUMB_R)
    denom, contact = 0, []
    for k in range(n):
        anchor = ctx.q(k, NOSE)
        if anchor is None:
            alt = [p for p in (ctx.q(k, i) for i in (2, 5, 9, 10)) if p is not None]
            anchor = (_mean([p[0] for p in alt]), _mean([p[1] for p in alt])) if alt else None
        best = None
        if anchor is not None:
            for i in touch_ids:
                p = ctx.q(k, i)
                if p is not None:
                    d = math.hypot(p[0] - anchor[0], p[1] - anchor[1])
                    if best is None or d < best[0]:
                        best = (d, p[1])
        if best is None:
            contact.append(None)
            continue
        denom += 1
        contact.append(best[0] <= st["radius_sw"] and best[1] <= st["y_max_sw"])
    cov_t = denom / n
    treason = _hand_reason(denom, cov_t, 1.0, 99, cfg, need_pairs=False)
    touch_flags = [bool(c) for c in contact]

    if hreason is None and fps < wc["min_fps_fidget"]:
        f1_reason = "fps<8"
    elif hreason is None and len(lag_pairs) < cfg["coverage"]["min_pairs"]:
        f1_reason = "too_few_pairs"
    else:
        f1_reason = hreason
    if f1_reason is None:
        energy = _median([p[1] for p in lag_pairs])
        side_speed = {s: (_median([p[2][s] for p in lag_pairs if s in p[2]]), _mean([p[2][s] for p in lag_pairs if s in p[2]]))
                      for s in ("L", "R") if any(s in p[2] for p in lag_pairs)}
        fast = max(side_speed, key=lambda s: side_speed[s])  # median speed, ties broken by mean speed (side-symmetric)
        wr = WR_L if fast == "L" else WR_R
        wq = [ctx.q(k, wr) for k in range(n)]
        wq = [p for p in wq if p is not None]
        rng = math.hypot(_pct([p[0] for p in wq], 0.95) - _pct([p[0] for p in wq], 0.05),
                         _pct([p[1] for p in wq], 0.95) - _pct([p[1] for p in wq], 0.05)) if wq else 0.0
        features["fidget_energy"], features["hand_range"] = energy, rng
        present = energy >= fc["thr_sw_per_s"] and rng <= fc["hand_range_max_sw"]
        cues.append(_cue("fidget", energy, "SW/s", present, (energy - fc["thr_sw_per_s"]) / fc["thr_sw_per_s"], cov_h, valid_hand, qs, cfg,
                         [f"fidget_energy={energy:.3f} SW/s", f"hand_range={rng:.3f} SW"]))
    else:
        cues += _unavailable(["fidget"], f1_units, f1_reason, cov_h, valid_hand, cfg)

    if treason is None:
        nvalid = [c for c in contact if c is not None]
        frac = sum(1 for c in nvalid if c) / len(nvalid)
        # A contact run needs observed adjacent times, never a missing interval.
        runs = []
        start = None
        for k, touching in enumerate(touch_flags):
            if start is not None and (not touching or ctx.pair_ok(k) is None):
                runs.append((start, k - 1))
                start = None
            if touching and start is None:
                start = k
        if start is not None:
            runs.append((start, n - 1))
        longest_s, longest_f = 0.0, 0
        for a, b in runs:
            duration = frames[b].t - frames[a].t
            if (duration, b - a + 1) > (longest_s, longest_f):
                longest_s, longest_f = duration, b - a + 1
        features["self_touch_frac"], features["self_touch_run_s"] = frac, longest_s
        present = frac >= st["frac_min"] and longest_s >= st["run_min_s"] - 1e-9 and longest_f >= st["run_min_frames"]
        cues.append(_cue("self_touch", frac, "fraction", present, (frac - st["frac_min"]) / st["frac_min"], cov_t, denom, qs, cfg,
                         [f"self_touch_frac={frac:.3f}", f"longest_run_s={longest_s:.2f}", "2D cannot separate touching from holding in front"]))
    else:
        cues += _unavailable(["self_touch"], {"self_touch": "fraction"}, treason, cov_t, denom, cfg)

    # --- F3 lean --------------------------------------------------------------------------------------
    third = n // 3
    early = [fr.S for fr in frames[:third] if fr.S is not None]
    late = [fr.S for fr in frames[n - third:] if fr.S is not None]
    lt = cfg["lean"]["thr_ln"]
    if third >= 1 and early and late:
        lean = math.log(_median(late) / _median(early))
        features["lean_scale"] = lean
        cov_s = ctx.s_valid_frac
        cues.append(_cue("lean_in", lean, "ln(scale)", lean >= lt, (lean - lt) / lt, cov_s, len([1 for fr in frames if fr.S]), qs, cfg,
                         [f"lean_scale={lean:.3f}", "apparent shoulder width: also chair, camera, zoom or torso yaw"]))
        cues.append(_cue("lean_back", lean, "ln(scale)", lean <= -lt, (-lean - lt) / lt, cov_s, len([1 for fr in frames if fr.S]), qs, cfg,
                         [f"lean_scale={lean:.3f}"]))
    else:
        cues += _unavailable(["lean_in", "lean_back"], {}, "no_shoulders", ctx.s_valid_frac, 0, cfg)

    # --- F4 body motion / stillness / freeze onset ---------------------------------------------------------
    mc = cfg["motion"]
    body_ids = (NOSE, SH_L, SH_R, EL_L, EL_R, WR_L, WR_R)

    min_pts = 5  # nose, both shoulders and both elbows are required; the wrists join only while visible

    def motion(lo, hi):
        """Total displacement / total time of the mean point displacement (dt-weighted M of the design)."""
        disp, tot, used = 0.0, 0.0, 0
        for k in range(lo + 1, hi + 1):
            dt = ctx.pair_ok(k)
            if dt is None:
                continue
            steps = []
            for i in body_ids:
                a, b = ctx.g(k - 1, i), ctx.g(k, i)
                if a is not None and b is not None:
                    steps.append(math.hypot(b[0] - a[0], b[1] - a[1]))
            if len(steps) < min_pts or any(ctx.g(k - 1, i) is None or ctx.g(k, i) is None for i in (NOSE, SH_L, SH_R, EL_L, EL_R)):
                continue
            disp += _mean(steps)
            tot += dt
            used += 1
        return (disp / tot if tot > 0 else None), used

    m_all, used_all = motion(0, n - 1)
    half = n // 2
    m_a, used_a = motion(0, half - 1)
    m_b, used_b = motion(n - half, n - 1)
    cov_m = used_all / (n - 1)
    if m_all is not None and cov_m >= cfg["coverage"]["cue_min"]:
        features["body_motion"] = m_all
        cues.append(_cue("stillness", m_all, "SW/s", m_all <= mc["m_still"], (mc["m_still"] - m_all) / mc["m_still"], cov_m, used_all, qs, cfg,
                         [f"body_motion={m_all:.3f} SW/s", "a still window cannot tell calm from frozen"]))
        half_pairs = max(half - 1, 1)
        short = None
        if used_a / half_pairs < cfg["coverage"]["cue_min"]:
            short = ("first", used_a / half_pairs)
        elif used_b / half_pairs < cfg["coverage"]["cue_min"]:
            short = ("last", used_b / half_pairs)
        if short is not None:  # each half needs the same coverage as the whole window before an onset can be claimed
            cues += _unavailable(["freeze_onset"], {}, f"occluded: coverage {short[1]:.2f} ({short[0]} half)", cov_m, used_all, cfg)
        elif m_a is not None and m_b is not None:
            features["body_motion_a"], features["body_motion_b"] = m_a, m_b
            onset = m_a >= mc["m_act"] and m_b <= mc["m_still"]
            margin = min((m_a - mc["m_act"]) / mc["m_act"], (mc["m_still"] - m_b) / mc["m_still"])
            cues.append(_cue("freeze_onset", m_a - m_b, "SW/s drop", onset, margin, cov_m, used_all, qs, cfg,
                             [f"M_first_half={m_a:.3f}", f"M_last_half={m_b:.3f}", "calm is the default explanation"]))
        else:
            cues += _unavailable(["freeze_onset"], {}, "too_few_pairs", cov_m, used_all, cfg)
    else:
        cues += _unavailable(["stillness", "freeze_onset"], {}, "occluded: coverage %.2f" % cov_m if used_all else "no_shoulders", cov_m, used_all, cfg)

    # --- F5 arm closure -----------------------------------------------------------------------------------------
    ac = cfg["arms"]
    crossed, closures, valid_arm, low_vis = 0, [], 0, False
    for k in range(n):
        a, b = ctx.q(k, WR_L, ctx.va), ctx.q(k, WR_R, ctx.va)
        if a is None or b is None:
            continue
        valid_arm += 1
        low_vis = low_vis or frames[k].pts[WR_L][2] < ctx.v or frames[k].pts[WR_R][2] < ctx.v
        closures.append(_clamp(1.0 - math.hypot(a[0] - b[0], a[1] - b[1]), 0.0, 1.0))
        yband = lambda y: ac["cross_y_lo_sw"] <= y <= ac["cross_y_hi_sw"]  # noqa: E731
        if a[0] <= -ac["cross_x_margin_sw"] and b[0] >= ac["cross_x_margin_sw"] and yband(a[1]) and yband(b[1]):
            crossed += 1
    cov_a = valid_arm / n
    arm_units = {"arms_crossed": "fraction", "hands_closed": "closure"}
    if valid_arm and cov_a >= cfg["coverage"]["cue_min"]:
        frac, closure = crossed / valid_arm, _median(closures)
        features["arm_cross_frac"], features["arm_closure"] = frac, closure
        flag = ["low_vis=true"] if low_vis else []
        cues.append(_cue("arms_crossed", frac, "fraction", frac >= ac["cross_frac_min"], (frac - ac["cross_frac_min"]) / ac["cross_frac_min"], cov_a, valid_arm, qs, cfg,
                         [f"arm_cross_frac={frac:.3f}", f"arm_closure={closure:.3f}"] + flag))
        cues.append(_cue("hands_closed", closure, "closure", closure >= ac["closure_min"] and frac < ac["cross_frac_min"], (closure - ac["closure_min"]) / ac["closure_min"],
                         cov_a, valid_arm, qs, cfg, [f"arm_closure={closure:.3f}", "weak"] + flag))
    else:
        reason = "hands_not_visible" if valid_arm == 0 else f"occluded: coverage {cov_a:.2f}"
        cues += _unavailable(["arms_crossed", "hands_closed"], arm_units, reason, cov_a, valid_arm, cfg)

    # --- F6 head-shoulder alignment ---------------------------------------------------------------------------------
    hc = cfg["head"]
    tilts, lats, heights = [], [], []
    for k in range(n):
        fr = frames[k]
        if all(_ok(fr, i, v) for i in (EAR_L, EAR_R, SH_L, SH_R)):
            th = math.degrees(math.atan2(fr.pts[EAR_L][1] - fr.pts[EAR_R][1], fr.pts[EAR_L][0] - fr.pts[EAR_R][0]))
            ts = math.degrees(math.atan2(fr.pts[SH_L][1] - fr.pts[SH_R][1], fr.pts[SH_L][0] - fr.pts[SH_R][0]))
            d = (th - ts + 180.0) % 360.0 - 180.0
            tilts.append(d)
        nq = ctx.q(k, NOSE)
        if nq is not None:
            lats.append(nq[0])
            heights.append(-nq[1])
    cov_ht, cov_hl = len(tilts) / n, len(lats) / n
    if cov_ht >= cfg["coverage"]["cue_min"]:
        tilt = _median(tilts)
        features["head_tilt_deg"] = tilt
        cues.append(_cue("head_tilt", tilt, "deg", abs(tilt) >= hc["tilt_deg"], (abs(tilt) - hc["tilt_deg"]) / hc["tilt_deg"], cov_ht, len(tilts), qs, cfg,
                         [f"head_tilt_deg={tilt:.2f}", "also listening, stiff neck or camera angle"]))
    else:
        cues += _unavailable(["head_tilt"], {"head_tilt": "deg"}, f"occluded: coverage {cov_ht:.2f}", cov_ht, len(tilts), cfg)
    if cov_hl >= cfg["coverage"]["cue_min"]:
        lat, hgt = _median(lats), _median(heights)
        features["head_lat_sw"], features["head_height_sw"] = lat, hgt
        cues.append(_cue("head_offset", lat, "SW", abs(lat) >= hc["lat_sw"], (abs(lat) - hc["lat_sw"]) / hc["lat_sw"], cov_hl, len(lats), qs, cfg, [f"head_lat_sw={lat:.3f}"]))
        cues.append(_cue("head_low", hgt, "SW", hgt <= hc["low_sw"], (hc["low_sw"] - hgt) / hc["low_sw"], cov_hl, len(heights), qs, cfg,
                         [f"head_height_sw={hgt:.3f}", "also looking at a phone or desk"]))
    else:
        cues += _unavailable(["head_offset", "head_low"], {}, f"occluded: coverage {cov_hl:.2f}", cov_hl, len(lats), cfg)

    # --- F7 gesture rate -----------------------------------------------------------------------------------------------
    gc = cfg["gesture"]
    if hreason is None:
        by_k = {p[0]: p for p in pairs}
        ks = sorted(by_k)
        # runs of consecutive pair indices with s_k >= stroke_v
        run_list, cur = [], []
        for k in ks:
            if by_k[k][1] >= gc["stroke_v_sw_per_s"] and (not cur or k == cur[-1] + 1):
                cur.append(k)
            else:
                if len(cur) >= gc["stroke_min_pairs"]:
                    run_list.append(cur)
                cur = [k] if by_k[k][1] >= gc["stroke_v_sw_per_s"] else []
        if len(cur) >= gc["stroke_min_pairs"]:
            run_list.append(cur)
        n_strokes = 0
        for run in run_list:
            start, end = run[0] - 1, run[-1]
            if touch_flags[start] or touch_flags[end]:
                continue
            speeds = {s: _median([by_k[k][2][s] for k in run if s in by_k[k][2]]) for s in ("L", "R") if any(s in by_k[k][2] for k in run)}
            fast = max(speeds, key=lambda s: speeds[s])
            a, b = ctx.q(start, WR_L if fast == "L" else WR_R), ctx.q(end, WR_L if fast == "L" else WR_R)
            if a is None or b is None:
                continue
            if math.hypot(b[0] - a[0], b[1] - a[1]) >= gc["stroke_net_sw"]:
                n_strokes += 1
        rate = n_strokes / span if span > 0 else 0.0
        band = "low" if rate < gc["band_low"] else "mid" if rate <= gc["band_high"] else "high"
        features["gesture_rate"], features["n_strokes"] = rate, n_strokes
        c = _cue("gesture_rate", rate, "strokes/s", False, 0.0, cov_h, valid_hand, qs, cfg,
                 [f"gesture_rate={rate:.3f} strokes/s", f"band={band}", "no meaning without a speech channel; reported only"])
        cues.append(c)
    else:
        cues += _unavailable(["gesture_rate"], f1_units, hreason, cov_h, valid_hand, cfg)

    return {"heuristic": HEURISTIC, "status": "ok", "reason": None, "window": window, "cues": cues, "features": features,
            "bystander_checked": checked, "pinned": True}


# --- evidence adapter: evidence items and capped confidence support only --------------------------------------

def _cue_conf(c, checked) -> float:
    """Finite cue confidence capped at the frozen cap, and at UNCHECKED_CUE_CAP when the bystander check did not run."""
    v = c.get("confidence")
    if isinstance(v, bool) or not _fin(v):
        return 0.0
    v = _clamp(float(v), 0.0, MODULE_CUE_CAP)
    return min(v, UNCHECKED_CUE_CAP) if not checked else v


def body_evidence(body: dict) -> list:
    """Evidence items for present cues. Items only: this never returns, creates or implies a state.

    Never raises. Items from a body block whose bystander check did not run (``bystander_checked`` not True) are
    flagged and capped at UNCHECKED_CUE_CAP.
    """
    items = []
    try:
        if not isinstance(body, dict) or body.get("status") != "ok" or not _shape_ok(body):
            return items
        checked = body.get("bystander_checked") is True
        for c in body.get("cues", []):
            if c.get("status") == "ok" and c.get("present"):
                ev = [str(e) for e in c.get("evidence", [])]
                if not checked:
                    ev.append("bystander: not checked (no face box supplied)")
                items.append({"source": "stage2-body", "cue": c["cue"], "value": c["value"], "unit": c["unit"],
                              "confidence": _cue_conf(c, checked), "wording": BODY_WORDING, "evidence": ev,
                              "bystander_checked": checked})
    except Exception:
        return []
    return items


def _cue_by_name(body, name):
    for c in body.get("cues", []):
        if isinstance(c, dict) and c.get("cue") == name:
            return c
    return None


def _shape_ok(body) -> bool:
    """An ok body block must carry a list of dict cues (anything else is malformed input)."""
    return isinstance(body, dict) and (body.get("status") != "ok" or
                                       (isinstance(body.get("cues"), list) and all(isinstance(c, dict) for c in body["cues"])
                                        and isinstance(body.get("features", {}), dict)))


def _invalid_support(reason="invalid_input") -> dict:
    return {"delta": 0.0, "items": [], "note": f"body: not available ({reason})"}


def support_delta(state_name: str, body: dict, cfg=None, allow_unpinned: bool = False) -> dict:
    """Confidence delta for one existing HB-1 state; 0 when the body block is unavailable. Never raises.

    The clamp [-0.05, +0.10] is a safety rail. Under the frozen weights the largest reachable positive delta is +0.0375
    (nervous-like) and +0.015 (unsure-like, needs-help-like), see MAX_POSITIVE_SUPPORT; the weak negative delta is -0.05
    when the bystander check ran and half of that (-0.025, scaled by UNCHECKED_CUE_CAP / MODULE_CUE_CAP) when it did not.
    """
    try:
        cfg = BODY_CONFIG if cfg is None else cfg
        if cfg is not BODY_CONFIG and not _is_pinned(cfg) and allow_unpinned is not True:
            return _invalid_support("config_not_pinned")
        cc = cfg["confidence"]
        if not isinstance(body, dict) or not _shape_ok(body):
            return _invalid_support()
        if body.get("status") != "ok":
            return _invalid_support(body.get("reason"))
        checked = body.get("bystander_checked") is True
        weights = cfg["support"].get(state_name, {})
        support, items = 0.0, []
        for cue_name, w in weights.items():
            c = _cue_by_name(body, cue_name)
            if c and c.get("status") == "ok" and c.get("present"):
                support += w * _cue_conf(c, checked)
                items.append(cue_name)
        delta = min(cc["delta_max"], cc["delta_per_support"] * support)
        note = "hypothesised support from descriptive movement cues; " + BODY_WORDING
        if not checked:
            note += "; bystander not checked, support capped"
        if state_name == "nervous-like" and not items:
            fid, touch, still = _cue_by_name(body, "fidget"), _cue_by_name(body, "self_touch"), _cue_by_name(body, "stillness")
            f = body.get("features", {})
            if (fid and touch and still and fid["status"] == "ok" and touch["status"] == "ok" and still["status"] == "ok"
                    and fid["coverage"] >= cfg["coverage"]["cue_min"] and touch["coverage"] >= cfg["coverage"]["cue_min"]
                    and f.get("fidget_energy", 1.0) < cc["negative_fidget_max"] and f.get("self_touch_frac", 1.0) == 0 and still["present"]):
                delta, note = cc["delta_negative"], "weak evidence of absence: hands visible and still; " + BODY_WORDING
                if not checked:  # positive support from unchecked data is capped; the negative claim must not be stronger
                    delta *= UNCHECKED_CUE_CAP / MODULE_CUE_CAP
                    note += "; bystander not checked, absence evidence scaled down"
        delta = _clamp(delta, max(cc["delta_negative"], MODULE_DELTA_MIN), min(cc["delta_max"], MODULE_DELTA_MAX))
        if not _fin(delta):
            return _invalid_support()
        return {"delta": round(delta, 4), "items": items, "note": note}
    except Exception:
        return _invalid_support()


INVALID_MARKER = "invalid_input"


def _json_safe(obj, depth=0):
    """Strict-JSON copy: non-finite numbers become 0.0, unsupported values become INVALID_MARKER."""
    if depth > 8:
        return INVALID_MARKER
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else 0.0
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v, depth + 1) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v, depth + 1) for k, v in obj.items()}
    return INVALID_MARKER


def _body_block(body, status=None, reason=None) -> dict:
    """Body block for attach_body. Caller-supplied contents are sanitised so the output is strict JSON."""
    ok = isinstance(body, dict)
    return {"heuristic": HEURISTIC, "status": status or (_json_safe(body.get("status")) if ok else "none"),
            "reason": reason or (_json_safe(body.get("reason")) if ok else "invalid_input"),
            "window": _json_safe(body.get("window")) if ok else None,
            "cues": _json_safe(body.get("cues", [])) if ok else [],
            "bystander_checked": bool(ok and body.get("bystander_checked") is True),
            "pinned": bool(ok and body.get("pinned", True) is True)}


def attach_body(hb1_result: dict, body: dict, cfg=None, allow_unpinned: bool = False) -> dict:
    """Return a copy of an HB-1 result with a ``body`` block and per-state ``body_support``.

    Body cues never create a state and never rescue an insufficient Stage-1 window; the HB-1 cap of 0.8 is final.
    Never raises: malformed input gives the unmodified copy plus an ``invalid_input`` body block. The state cap is the
    module constant 0.8 (and the body delta is bounded to [-0.05, +0.10]) even if an unpinned config says otherwise. The
    body block is sanitised to strict JSON; a malformed HB-1 input is returned as given (the caller's own values).
    """
    try:
        cfg = BODY_CONFIG if cfg is None else cfg
        out = copy.deepcopy(hb1_result)
        cap = min(MODULE_STATE_CAP, float(cfg["confidence"]["state_cap"]))  # an opted-in config can never raise the 0.8 cap
        if not isinstance(out, dict) or not isinstance(out.get("states", []), list) or (isinstance(body, dict) and not _shape_ok(body)):
            raise TypeError("hb1_result must be a dict with a states list and body a well-formed block")
        if bool(out.get("insufficient", {}).get("flag")):
            out["states"] = []
        staged = []
        for st in out.get("states", []):
            if not isinstance(st, dict) or not isinstance(st.get("name"), str):
                raise TypeError("each state needs a string name")
            sup = support_delta(st["name"], body, cfg, allow_unpinned)
            conf = st["confidence"]
            if isinstance(conf, bool) or not _fin(conf):
                raise TypeError("state confidence must be a finite number")
            staged.append((st, sup, round(min(cap, max(0.0, conf + sup["delta"])), 4)))
        for st, sup, conf in staged:
            st["body_support"] = sup
            st["confidence"] = conf
        out["body"] = _body_block(body)
        return out
    except Exception:
        try:
            base = copy.deepcopy(hb1_result) if isinstance(hb1_result, dict) else {"states": [], "insufficient": {"flag": True, "reasons": ["invalid_input"]}}
        except Exception:
            base = {"states": [], "insufficient": {"flag": True, "reasons": ["invalid_input"]}}
        base["body"] = _body_block(None, "none", "invalid_input")
        return base
