"""Synthetic FrameRecord streams for HB-1 (EL-006 S1). Manufactured numbers only.

Every stream is fabricated: no camera, media, weights or dataset is read. The
per-frame noise draws are taken in a fixed order, independent of the cue
strengths, so the same seed yields the same noise at every strength.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Optional

from .state import FER7, FrameRecord

LABEL = "SYNTHETIC - manufactured numbers, not a measurement"
SCENARIOS = ("calm", "nervous-like", "unsure-like", "needs-help-like", "insufficient", "multi-face", "dropout")

# What the final analyze() of each default scenario is designed to yield.
EXPECTED = {
    "calm": {"states": set(), "insufficient": False},
    "nervous-like": {"states": {"nervous-like"}, "insufficient": False},
    "unsure-like": {"states": {"unsure-like"}, "insufficient": False},
    "needs-help-like": {"states": {"unsure-like", "needs-help-like"}, "insufficient": False},
    "insufficient": {"states": set(), "insufficient": True, "reasons": {"low_quality"}},
    "multi-face": {"states": set(), "insufficient": True, "reasons": {"multiple_faces"}},
    "dropout": {"states": set(), "insufficient": True, "reasons": {"low_coverage"}},
}

_IOD = 60.0
_CYCLE = ("neutral", "happy", "angry")
_AWAY_CYCLE_FRAMES = 12
_AWAY_MAX_FRAMES = 5


def _lerp(a: float, b: float, s: float) -> float:
    return a + (b - a) * s


def _norm(p: dict) -> dict:
    total = sum(p.values())
    return {k: p[k] / total for k in FER7} if total > 0 else {k: 1.0 / len(FER7) for k in FER7}


def _expression(i: int, nerv: float, uns: float) -> dict:
    """FER7 dict for frame ``i`` at nervous strength ``nerv`` and unsure strength ``uns``."""
    if uns > 0.0 or nerv <= 0.0:
        step = max(1, round(_lerp(12, 1, uns)))
        label = _CYCLE[(i // step) % len(_CYCLE)]
        top = _lerp(0.80, 0.35, uns) + (0.08 * uns if i % 2 else -0.08 * uns)
        rest = max(0.0, 1.0 - top - 0.06) / 3.0
        p = {k: 0.02 for k in FER7}
        for k in ("neutral", "happy", "angry", "disgust"):
            p[k] = rest
        p[label] = top
        if uns <= 0.0:
            p = {k: 0.02 for k in FER7}
            p.update(neutral=0.70, happy=0.15, angry=0.02)
        return _norm(p)
    p = {k: 0.02 for k in FER7}
    p["fear"] = _lerp(0.02, 0.50, nerv)
    p["surprise"] = _lerp(0.02, 0.20, nerv)
    p["sad"] = _lerp(0.02, 0.08, nerv)
    p["neutral"] = _lerp(0.70, 0.12, nerv)
    p["happy"] = _lerp(0.15, 0.04, nerv)
    return _norm(p)


def _away(i: int, uns: float) -> bool:
    return (i % _AWAY_CYCLE_FRAMES) < round(_lerp(0, _AWAY_MAX_FRAMES, uns))


def build(
    duration: float = 8.0,
    fps: float = 8.0,
    seed: int = 505,
    nerv: Callable[[float], float] = lambda t: 0.0,
    uns: Callable[[float], float] = lambda t: 0.0,
    quality: float = 0.95,
    face_count: Callable[[float], int] = lambda t: 1,
    drop: Callable[[float, float], bool] = lambda t, u: False,
) -> list:
    """Generic builder; ``nerv``/``uns`` give cue strength in [0, 1] at time t."""
    rng = random.Random(seed)
    out = []
    n = int(round(duration * fps))
    for i in range(n):
        t = round(i / fps, 6)
        zx, zy, zr, zu, zq = (rng.gauss(0, 1) for _ in range(5))
        u = rng.random()
        s_n, s_u = nerv(t), uns(t)
        if drop(t, u):
            out.append(FrameRecord(t, None, quality, 0.0, 0.0, None, None, 0))
            continue
        sd = _lerp(0.005, 0.10, s_n)
        cx = 320.0 + sd * _IOD * zx
        cy = 200.0 + sd * _IOD * zy
        roll = 0.25 * sd * zr
        yaw = (0.30 if _away(i, s_u) else 0.0) + 0.02 * zu
        dx, dy = (_IOD / 2) * math.cos(roll), (_IOD / 2) * math.sin(roll)
        eye_r, eye_l = (cx - dx, cy - dy), (cx + dx, cy + dy)
        nose = (cx + yaw * _IOD, cy + 35.0)
        lm = (eye_r, eye_l, nose, (cx - 20.0, cy + 65.0), (cx + 20.0, cy + 65.0))
        arousal = min(1.0, max(0.0, _lerp(0.10, 0.70, s_n) if s_n > 0 else 0.10 + 0.05 * s_u))
        q = min(1.0, max(0.0, quality + 0.01 * zq))
        out.append(FrameRecord(t, _expression(i, s_n, s_u), q, arousal, 0.0,
                               (cx - 50.0, cy - 40.0, 100.0, 120.0), lm, face_count(t)))
    return out


def generate(name: str, seed: int = 505, fps: float = 8.0, strength: float = 1.0,
             duration: Optional[float] = None) -> list:
    """Records for a named scenario. ``strength`` scales the cue of nervous/unsure/needs-help."""
    if name == "calm":
        return build(duration or 8.0, fps, seed)
    if name == "nervous-like":
        return build(duration or 8.0, fps, seed, nerv=lambda t: strength)
    if name == "unsure-like":
        return build(duration or 5.0, fps, seed, uns=lambda t: strength)
    if name == "needs-help-like":
        return build(duration or 14.0, fps, seed, uns=lambda t: strength if t >= 4.0 else 0.0)
    if name == "insufficient":
        return build(duration or 8.0, fps, seed, quality=0.20)
    if name == "multi-face":
        return build(duration or 8.0, fps, seed, face_count=lambda t: 2 if t >= 2.0 else 1)
    if name == "dropout":
        return build(duration or 8.0, fps, seed, drop=lambda t, u: t >= 5.0 and u < 0.65)
    raise ValueError(f"unknown scenario: {name!r}")
