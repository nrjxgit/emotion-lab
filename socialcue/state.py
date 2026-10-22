"""HB-1 heuristic state module (EL-006 S1). Research prototype, unvalidated.

Pure and deterministic: no IO at import, no filesystem, no network, no model
loading. It turns a short stream of per-frame face records into hedged
"signs consistent with ..." hypotheses. Every threshold lives in the frozen
config below (mirrored byte-for-byte by ``socialcue/hb1_config.json``) and is
a Designer/Command choice that was never tuned on real data.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

FER7 = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")

HEURISTIC = "HB-1 hypothesis, unvalidated"
# Fixed hedge for S4/S5 to pass through to the UI. Kept out of analyze() output so the
# section 4 exact-keys contract is unchanged.
WORDING = "signs consistent with observable behaviour, not a reading of feelings; hypothesis, unvalidated"
# Plain-language labels for raw feature keys; the gaze key is only a head-orientation proxy.
FEATURE_LABELS = {
    "gaze_away": "gaze_away (head-orientation proxy from eye-midpoint/nose offset; not gaze)",
}
STATE_NAMES = ("nervous-like", "unsure-like", "needs-help-like")

# Frozen HB-1 config. Keep identical to socialcue/hb1_config.json (a test
# asserts equality and pins the canonical-JSON sha256).
HB1_CONFIG = {
    "name": "HB-1",
    "version": 1,
    "comment": "Designer/Command choices from the EL-006 cycle order, section 2. Hypothesis thresholds, NEVER tuned on real data; synthetic streams only. Changing any value changes the pinned hash.",
    "window": {
        "comment": "W = 3 s; minimum 2 s at >= 5 usable fps; expected_fps is the paced-loop target midpoint (8-10 fps).",
        "seconds": 3.0,
        "min_seconds": 2.0,
        "min_usable_fps": 5.0,
        "expected_fps": 8.0,
    },
    "gates": {
        "comment": "Below these the output is INSUFFICIENT EVIDENCE; more than one face in the window is never analysed.",
        "min_coverage": 0.6,
        "min_quality": 0.3,
    },
    "confidence": {
        "comment": "confidence = rule margin x coverage x mean detector quality, capped below the fusion caps.",
        "cap": 0.8,
    },
    "features": {
        "comment": "gaze_away is a head-orientation PROXY (eye-midpoint to nose offset over inter-ocular distance), not gaze. Weights apply to FER7 probabilities.",
        "neg_weights": {"fear": 1.0, "surprise": 0.8, "sad": 0.5},
        "yaw_away_threshold": 0.25,
        "osc_min_delta": 0.05,
        "still_radius_iod": 1.5,
        "dwell_gap_seconds": 1.0,
    },
    "nervous_like": {
        "comment": "high p_neg_arousal AND (head_jitter high OR arousal high). Margin = clip((x - thr) / span, 0, 1); AND = min, OR = max.",
        "p_neg_arousal_thr": 0.30,
        "p_neg_arousal_span": 0.30,
        "head_jitter_thr": 0.04,
        "head_jitter_span": 0.06,
        "arousal_hi_thr": 0.50,
        "arousal_hi_span": 0.40,
    },
    "unsure_like": {
        "comment": "(neutral OR low-confidence expression) AND hesitation high AND gaze_away moderate (band) AND low arousal.",
        "p_neutral_thr": 0.40,
        "p_neutral_span": 0.30,
        "top_prob_thr": 0.50,
        "top_prob_span": 0.20,
        "hesitation_thr": 0.25,
        "hesitation_span": 0.35,
        "gaze_away_lo": 0.15,
        "gaze_away_hi": 0.60,
        "gaze_away_edge_span": 0.10,
        "arousal_lo_thr": 0.35,
        "arousal_lo_span": 0.35,
    },
    "needs_help_like": {
        "comment": "temporal escalation only: dwell > dwell_s AND (nervous-like or unsure-like firing in >= sustain_fraction of the evaluation points over the last 2 windows) AND gaze_away episodes >= episodes_min.",
        "dwell_s": 5.0,
        "dwell_span_s": 5.0,
        "sustain_fraction": 0.5,
        "sustain_points": 4,
        "sustain_step_s": 1.5,
        "episodes_min": 2,
        "episodes_span": 2.0,
    },
}


def canonical_json(cfg: Mapping) -> str:
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_sha256(cfg: Mapping) -> str:
    return hashlib.sha256(canonical_json(cfg).encode("ascii")).hexdigest()


CONFIG_SHA256 = config_sha256(HB1_CONFIG)


@dataclass(frozen=True)
class FrameRecord:
    """One detector tick. ``box``/``landmarks`` are None when no face was found.

    landmarks: 5 (x, y) points in YuNet order: eye_R, eye_L, nose, mouth_R, mouth_L.
    """

    t: float
    expression: Optional[Mapping[str, float]]
    quality: float
    arousal: float
    valence: float
    box: Optional[tuple]
    landmarks: Optional[tuple]
    face_count: int = 1


class PacedRecords(list):
    """Internal session snapshot with the fixed 125 ms bin origin used at retention.

    Ordinary lists keep the historical newest-usable coverage anchor. Session
    snapshots share their retention origin with coverage, avoiding phase drift.

    ``current`` (set by the session store, else None) is the record just added by the request, which can differ from
    the retained one when two requests fall into the same 125 ms bin (earliest wins at retention).
    """
    current = None

    def __init__(self, records, bin_origin):
        super().__init__(records)
        self.bin_origin = bin_origin


def _finite(*vals) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in vals)


def _iod(r: FrameRecord) -> float:
    (ax, ay), (bx, by) = r.landmarks[0], r.landmarks[1]
    return math.hypot(bx - ax, by - ay)


def _count(r: FrameRecord) -> Optional[int]:
    """Numeric face count, or None when it is missing or not a number."""
    c = r.face_count
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c):
        return None
    return int(c)


def _usable(r: FrameRecord) -> bool:
    """True only for a well-formed single-face frame; any malformed field makes it unusable."""
    try:
        if _count(r) != 1 or r.box is None or r.landmarks is None or not isinstance(r.expression, Mapping):
            return False
        if len(r.landmarks) != 5 or len(r.box) != 4:
            return False
        flat = []
        for p in r.landmarks:
            if len(p) != 2:
                return False
            flat.extend(p)
        if not _finite(r.t, r.quality, r.arousal, *flat, *r.box):
            return False
        if not _finite(*(r.expression.get(k, 0.0) for k in FER7)):
            return False
        return _iod(r) > 1e-6
    except Exception:
        return False


def _q(r: FrameRecord) -> float:
    return _clip01(float(r.quality))


def _ar(r: FrameRecord) -> float:
    return _clip01(float(r.arousal))


def _p(r: FrameRecord, k: str) -> float:
    return _clip01(float(r.expression.get(k, 0.0)))


def _clean(records) -> list:
    """Finite capture instants, sorted and coalesced; unusable duplicates stay missing."""
    try:
        items = list(records) if records is not None else []
    except TypeError:
        return []
    good = [r for r in items if isinstance(r, FrameRecord) and _finite(r.t)]
    unique = []
    for r in sorted(good, key=lambda r: r.t):
        if unique and r.t == unique[-1].t:
            # Prefer multi-person/absent evidence over a usable duplicate.
            prior = unique[-1]
            if (_count(r) or 0) > 1 or ((_count(prior) or 0) <= 1 and not _usable(r)):
                unique[-1] = r
        else:
            unique.append(r)
    return unique


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _hi(x: float, thr: float, span: float) -> float:
    return _clip01((x - thr) / span)


def _lo(x: float, thr: float, span: float) -> float:
    return _clip01((thr - x) / span)


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _sd(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _roll_sd(angles: Sequence[float]) -> float:
    """SD of directed eye_R->eye_L roll; broad samples unwrap about their circular mean.

    Preserve plain-SD bytes whenever the sample range is <= pi. Otherwise use a
    sorted, accurately summed circular mean and a sorted unwrapped multiset, so
    order cannot choose the cut. A vanishing resultant has no unique direction;
    use a deterministic zero cut there, not a claim of global angular continuity.
    """
    if not angles or max(angles) - min(angles) <= math.pi:
        return _sd(angles)
    ordered = sorted(angles)
    sine = math.fsum(math.sin(a) for a in ordered)
    cosine = math.fsum(math.cos(a) for a in ordered)
    mean = math.atan2(sine, cosine) if math.hypot(sine, cosine) > 1e-12 * len(ordered) else 0.0
    unwrapped = sorted(mean + math.atan2(math.sin(a - mean), math.cos(a - mean)) for a in ordered)
    return _sd(unwrapped)


def _top(r: FrameRecord) -> tuple:
    best, bp = FER7[0], _p(r, FER7[0])
    for k in FER7[1:]:
        p = _p(r, k)
        if p > bp:
            best, bp = k, p
    return best, bp


def _window_records(records: Sequence[FrameRecord], t_end: float, w: float) -> list:
    return [r for r in records if t_end - w < r.t <= t_end]


def _window_features(recs: Sequence[FrameRecord], t_end: float, cfg: Mapping, bin_origin=None) -> dict:
    """Features over (t_end - W, t_end]. ``recs`` is every record in that window."""
    wc, fc = cfg["window"], cfg["features"]
    use = [r for r in recs if _usable(r)]
    n = len(use)
    # Anchor to newest usable evidence: absent ticks cannot move the bin edges.
    # Distinct bursts cannot supply temporal support for empty intervals.
    anchor = use[-1].t if use else t_end
    support = len({math.floor(((anchor - r.t) if bin_origin is None else (r.t - bin_origin)) * wc["expected_fps"] + 1e-9) for r in use})
    first_t = recs[0].t if recs else t_end
    eff = min(wc["seconds"], t_end - first_t + 1.0 / wc["expected_fps"])
    expected = wc["expected_fps"] * eff
    span = max(t_end - first_t, 1.0 / wc["expected_fps"])
    f = {
        "n_records": len(recs),
        "usable_frames": n,
        "span_s": span,
        # Contract key retained: bin-credit rate per second, NOT measured usable FPS.
        "fps": support / span if math.isfinite(support / span) else 0.0,
        "coverage": min(1.0, support / expected) if expected > 0 else 0.0,
        "quality": _mean([_q(r) for r in use]),
        "multiple_faces": any((_count(r) or 0) > 1 for r in recs),
    }
    if n == 0:
        f.update(p_neg_arousal=0.0, p_calm=0.0, p_neutral=0.0, mean_top_prob=0.0, arousal=0.0,
                 head_jitter=0.0, gaze_away=0.0, gaze_away_episodes=0, hesitation=0.0)
        return f
    nw = fc["neg_weights"]
    f["p_neg_arousal"] = _mean([sum(w * _p(r, k) for k, w in nw.items()) for r in use])
    f["p_calm"] = _mean([_p(r, "neutral") + _p(r, "happy") for r in use])
    f["p_neutral"] = _mean([_p(r, "neutral") for r in use])
    tops = [_top(r) for r in use]
    f["mean_top_prob"] = _mean([p for _, p in tops])
    f["arousal"] = _mean([_ar(r) for r in use])

    miod = _mean([_iod(r) for r in use])
    nx = [r.landmarks[2][0] / miod for r in use]
    ny = [r.landmarks[2][1] / miod for r in use]
    roll = [math.atan2(r.landmarks[1][1] - r.landmarks[0][1], r.landmarks[1][0] - r.landmarks[0][0]) for r in use]
    f["head_jitter"] = math.sqrt(_sd(nx) ** 2 + _sd(ny) ** 2 + _roll_sd(roll) ** 2)

    away, episodes, prev = 0, 0, False
    for r in use:
        mid_x = (r.landmarks[0][0] + r.landmarks[1][0]) / 2.0
        a = abs(r.landmarks[2][0] - mid_x) / _iod(r) > fc["yaw_away_threshold"]
        away += a
        episodes += a and not prev
        prev = a
    f["gaze_away"] = away / n
    f["gaze_away_episodes"] = episodes

    if n < 2:
        f["hesitation"] = 0.0
    else:
        switches = sum(1 for i in range(1, n) if tops[i][0] != tops[i - 1][0])
        deltas = [tops[i][1] - tops[i - 1][1] for i in range(1, n)]
        d0 = fc["osc_min_delta"]
        osc = sum(1 for i in range(1, len(deltas))
                  if abs(deltas[i]) >= d0 and abs(deltas[i - 1]) >= d0 and deltas[i] * deltas[i - 1] < 0)
        f["hesitation"] = min(1.0, (switches + 0.5 * osc) / (n - 1))
    return f


def _insufficient_reasons(f: Mapping, cfg: Mapping) -> list:
    wc, gc = cfg["window"], cfg["gates"]
    reasons = []
    if f["multiple_faces"]:
        reasons.append("multiple_faces")
    if f["usable_frames"] == 0:
        reasons.append("no_face")
    if f["span_s"] < wc["min_seconds"] - 1e-9:
        reasons.append("window_filling")
    else:
        if f["usable_frames"] and f["coverage"] < gc["min_coverage"]:
            reasons.append("low_coverage")
        if f["usable_frames"] and f["fps"] < wc["min_usable_fps"]:
            reasons.append("low_fps")
    if f["usable_frames"] and f["quality"] < gc["min_quality"]:
        reasons.append("low_quality")
    return reasons


def _ev(name: str, value) -> str:
    if name == "gaze_away":
        name = "gaze_away[head-orientation proxy]"
    return f"{name}={value:.3f}" if isinstance(value, float) else f"{name}={value}"


def _base_margins(f: Mapping, cfg: Mapping) -> dict:
    """Return {state: (margin, [feature names used])} for nervous-like and unsure-like."""
    nc, uc = cfg["nervous_like"], cfg["unsure_like"]
    m_neg = _hi(f["p_neg_arousal"], nc["p_neg_arousal_thr"], nc["p_neg_arousal_span"])
    m_jit = _hi(f["head_jitter"], nc["head_jitter_thr"], nc["head_jitter_span"])
    m_aro = _hi(f["arousal"], nc["arousal_hi_thr"], nc["arousal_hi_span"])
    nervous = min(m_neg, max(m_jit, m_aro))

    m_neu = max(_hi(f["p_neutral"], uc["p_neutral_thr"], uc["p_neutral_span"]),
                _lo(f["mean_top_prob"], uc["top_prob_thr"], uc["top_prob_span"]))
    m_hes = _hi(f["hesitation"], uc["hesitation_thr"], uc["hesitation_span"])
    m_gaze = min(_hi(f["gaze_away"], uc["gaze_away_lo"], uc["gaze_away_edge_span"]),
                 _lo(f["gaze_away"], uc["gaze_away_hi"], uc["gaze_away_edge_span"]))
    m_low = _lo(f["arousal"], uc["arousal_lo_thr"], uc["arousal_lo_span"])
    unsure = min(m_neu, m_hes, m_gaze, m_low)
    return {
        "nervous-like": (nervous, ["p_neg_arousal", "head_jitter", "arousal"]),
        "unsure-like": (unsure, ["p_neutral", "mean_top_prob", "hesitation", "gaze_away", "arousal"]),
    }


def _fires_base(f: Mapping, cfg: Mapping) -> bool:
    if _insufficient_reasons(f, cfg):
        return False
    return any(m > 0.0 for m, _ in _base_margins(f, cfg).values())


def _dwell_seconds(records: Sequence[FrameRecord], t_end: float, cfg: Mapping) -> float:
    fc = cfg["features"]
    cur = None
    start = None
    last_good = None
    newest = None
    for r in reversed(records):
        if r.t > t_end or not _usable(r):
            continue
        nose = (r.landmarks[2][0] / _iod(r), r.landmarks[2][1] / _iod(r))
        if cur is None:
            if t_end - r.t > fc["dwell_gap_seconds"]:
                return 0.0
            cur, start, last_good = nose, r.t, r.t
            newest = r.t
            continue
        if last_good - r.t > fc["dwell_gap_seconds"]:
            break
        if math.hypot(nose[0] - cur[0], nose[1] - cur[1]) > fc["still_radius_iod"]:
            break
        start, last_good = r.t, r.t
    return 0.0 if start is None else max(0.0, newest - start)


def extract_features(records: Sequence[FrameRecord], config: Optional[Mapping] = None) -> dict:
    """Window features at the last record's time (public, for tests and the UI 'why').

    Malformed records are ignored, never raised on. ``gaze_away`` is only a
    head-orientation PROXY (eye-midpoint to nose offset over inter-ocular
    distance), not gaze; see FEATURE_LABELS for the label to show next to it.
    """
    cfg = config if config is not None else HB1_CONFIG
    bin_origin = records.bin_origin if type(records) is PacedRecords else None
    recs = _clean(records)
    if not recs:
        f = _window_features([], 0.0, cfg)
        f["dwell"] = 0.0
        return f
    t_end = recs[-1].t
    f = _window_features(_window_records(recs, t_end, cfg["window"]["seconds"]), t_end, cfg, bin_origin)
    f["dwell"] = _dwell_seconds(recs, t_end, cfg)
    return f


def analyze(records: Sequence[FrameRecord], config: Optional[Mapping] = None, *, evaluation_time: Optional[float] = None) -> dict:
    """Evaluate HB-1 on (evaluation_time - W, evaluation_time].

    None preserves the last-record evaluation time. Explicit finite server/pilot
    time ages support even without new observations; all history remains for dwell.

    Returns {states, insufficient, window, heuristic, proxy_eval_ref}. States are
    hedged hypotheses ("-like"), not mutually exclusive, and empty means no cue.
    """
    cfg = config if config is not None else HB1_CONFIG
    wc, cap = cfg["window"], cfg["confidence"]["cap"]
    bin_origin = records.bin_origin if type(records) is PacedRecords else None
    recs = _clean(records)
    if evaluation_time is not None and not _finite(evaluation_time):
        raise ValueError("evaluation_time must be a finite numeric instant")
    t_end = evaluation_time if evaluation_time is not None else (recs[-1].t if recs else 0.0)
    f = _window_features(_window_records(recs, t_end, wc["seconds"]), t_end, cfg, bin_origin)
    reasons = _insufficient_reasons(f, cfg)
    out = {
        "states": [],
        "insufficient": {"flag": bool(reasons), "reasons": reasons},
        # window.fps retains its public name but reports expected-time-bin credits / span.
        "window": {"seconds": wc["seconds"], "usable_frames": f["usable_frames"], "fps": round(f["fps"], 2)},
        "heuristic": HEURISTIC,
        "proxy_eval_ref": None,
    }
    if reasons:
        return out

    scale = f["coverage"] * f["quality"]
    margins = _base_margins(f, cfg)
    states = []
    for name in ("nervous-like", "unsure-like"):
        margin, feats = margins[name]
        if margin > 0.0:
            states.append({
                "name": name,
                "score": round(margin, 4),
                "confidence": round(min(cap, margin * scale), 4),
                "evidence": [_ev(k, f[k]) for k in feats] + [_ev("coverage", f["coverage"]), _ev("quality", f["quality"])],
            })

    hc = cfg["needs_help_like"]
    dwell = _dwell_seconds(recs, t_end, cfg)
    sustained = 0
    for j in range(hc["sustain_points"]):
        te = t_end - j * hc["sustain_step_s"]
        fj = f if j == 0 else _window_features(_window_records(recs, te, wc["seconds"]), te, cfg, bin_origin)
        sustained += _fires_base(fj, cfg)
    frac = sustained / hc["sustain_points"]
    episodes = f["gaze_away_episodes"]
    if frac >= hc["sustain_fraction"] and dwell > hc["dwell_s"] and episodes >= hc["episodes_min"]:
        margin = min(_hi(dwell, hc["dwell_s"], hc["dwell_span_s"]),
                     _hi(float(episodes), hc["episodes_min"] - 1, hc["episodes_span"]),
                     frac)
        if margin > 0.0:
            states.append({
                "name": "needs-help-like",
                "score": round(margin, 4),
                "confidence": round(min(cap, margin * scale), 4),
                "evidence": [_ev("dwell_s", dwell), _ev("gaze_away_episodes", episodes),
                             _ev("sustained_fraction", frac), _ev("coverage", f["coverage"]),
                             _ev("quality", f["quality"])],
            })
    out["states"] = states
    return out


def load_config(path: str) -> dict:
    """Explicit-call helper (never used at import): read a config JSON file."""
    with open(path, "r", encoding="ascii") as fh:
        return json.load(fh)


def default_config() -> dict:
    return copy.deepcopy(HB1_CONFIG)
