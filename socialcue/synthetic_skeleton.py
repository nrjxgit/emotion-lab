"""Manufactured 33-landmark skeleton sequences for the Stage-2 body-cue module (EL-006 S6b).

Every sequence is fabricated from the closed-form generators of stage2-design.md section 3. No person, media,
camera, pose model or dataset is involved. Pure stdlib; deterministic (noise uses random.Random(505) in a single
fixed draw order, so the noisy values differ from the design's numpy draw and are measured, not derived).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

LABEL = "SYNTHETIC skeleton - manufactured numbers, not a person"
W0, H0 = 640, 480
SW_PX = 160.0
CX, CY = 320.0, 200.0

# canonical pose in shoulder-width units (q-space, origin = shoulder midpoint, y down)
CANON = {
    0: (0, -0.55), 1: (0.10, -0.68), 2: (0.17, -0.68), 3: (0.24, -0.68), 4: (-0.10, -0.68), 5: (-0.17, -0.68), 6: (-0.24, -0.68),
    7: (0.38, -0.62), 8: (-0.38, -0.62), 9: (0.10, -0.40), 10: (-0.10, -0.40), 11: (0.50, 0), 12: (-0.50, 0),
    13: (0.55, 0.75), 14: (-0.55, 0.75), 15: (0.50, 1.45), 16: (-0.50, 1.45), 17: (0.55, 1.60), 18: (-0.55, 1.60),
    19: (0.50, 1.65), 20: (-0.50, 1.65), 21: (0.45, 1.57), 22: (-0.45, 1.57), 23: (0.35, 2.20), 24: (-0.35, 2.20),
    25: (0.35, 3.30), 26: (-0.35, 3.30), 27: (0.35, 4.40), 28: (-0.35, 4.40), 29: (0.35, 4.50), 30: (-0.35, 4.50),
    31: (0.40, 4.60), 32: (-0.40, 4.60),
}
RIGHT_ARM = (14, 16, 18, 20, 22)
RIGHT_HAND = (16, 18, 20, 22)
LEFT_HAND = (15, 17, 19, 21)
MIRROR = {1: 4, 2: 5, 3: 6, 7: 8, 9: 10, 11: 12, 13: 14, 15: 16, 17: 18, 19: 20, 21: 22, 23: 24, 25: 26, 27: 28, 29: 30, 31: 32}
MIRROR.update({v: k for k, v in list(MIRROR.items())})
NAMES = ("calm", "fidget", "self_touch", "freeze", "lean_in", "arms_crossed", "occluded", "gesture", "tilt", "desk_crop", "swap")


@dataclass
class SkeletonSeq:
    name: str
    t: list
    lm: list  # per frame: 33 x [x_norm, y_norm, vis]
    W: int
    H: int
    params: dict = field(default_factory=dict)
    label: str = LABEL


def _base_vis():
    return [0.99 if i <= 22 else 0.30 if i <= 24 else 0.05 for i in range(33)]


def _px(q):
    return (CX + SW_PX * q[0], CY + SW_PX * q[1])


def build(name, *, fs=10.0, n=30, noise_sw=0.0, W=W0, H=H0, seed=505, **kw) -> SkeletonSeq:
    """One of the eleven manufactured sequences. Extra keyword params tune the generator (e.g. amplitude)."""
    if name not in NAMES:
        raise ValueError(f"unknown skeleton fixture: {name!r}")
    rng = random.Random(seed)
    noise = [[(rng.gauss(0, noise_sw), rng.gauss(0, noise_sw)) for _ in range(33)] for _ in range(n)] if noise_sw > 0 else None
    ts, frames = [], []
    for k in range(n):
        t = k / fs
        q = {i: list(c) for i, c in CANON.items()}
        vis = _base_vis()
        px_override = {}
        if name == "fidget":
            a, f = kw.get("amplitude", 0.08), kw.get("freq", 2.0)  # 0.06 before S60; the lagged F1 definition needs 0.08 for the fixture to stay above the unchanged threshold
            for i in RIGHT_HAND:
                q[i][0] += a * math.sin(2 * math.pi * f * t)
        elif name == "self_touch":
            u = _touch_u(t)
            touch = {14: (-0.60, 0.30), 16: (-0.12, -0.25), 18: (-0.17, -0.28), 20: (-0.12, -0.45), 22: (-0.07, -0.32)}
            for i in RIGHT_ARM:
                c = CANON[i]
                q[i] = [c[0] + u * (touch[i][0] - c[0]), c[1] + u * (touch[i][1] - c[1])]
        elif name == "freeze":
            dx = 0.10 * math.sin(2 * math.pi * min(t, 1.4))
            for i in q:
                q[i][0] += dx
        elif name == "arms_crossed":
            w15, w16 = (-0.20, 0.60), (0.20, 0.70)
            q[15], q[16], q[13], q[14] = list(w15), list(w16), [0.62, 0.85], [-0.62, 0.85]
            q[17], q[19], q[21] = [w15[0] + 0.05, w15[1] + 0.15], [w15[0], w15[1] + 0.20], [w15[0] - 0.05, w15[1] + 0.12]
            q[18], q[20], q[22] = [w16[0] - 0.05, w16[1] + 0.15], [w16[0], w16[1] + 0.20], [w16[0] + 0.05, w16[1] + 0.12]
            if kw.get("clasped"):  # hands together in front, not crossed: closure 0.8
                q[15], q[16] = [0.10, 0.60], [-0.10, 0.60]
                for i, (dx, dy) in {17: (0.05, 0.15), 19: (0.0, 0.20), 21: (-0.05, 0.12)}.items():
                    q[i] = [q[15][0] + dx, q[15][1] + dy]
                for i, (dx, dy) in {18: (-0.05, 0.15), 20: (0.0, 0.20), 22: (0.05, 0.12)}.items():
                    q[i] = [q[16][0] + dx, q[16][1] + dy]
            wv = kw.get("wrist_vis")
            if wv is not None:
                vis[15] = vis[16] = wv
        elif name == "gesture":
            a, f = kw.get("amplitude", 0.45), kw.get("freq", 1.0)
            for i in RIGHT_HAND:
                q[i][0] += a * math.sin(2 * math.pi * f * t)
        elif name == "tilt":
            al = math.radians(kw.get("degrees", 20.0))
            ny = -0.15
            for i in range(0, 11):
                x, y = q[i]
                q[i] = [x * math.cos(al) - (y - ny) * math.sin(al) + kw.get("head_dx", 0.0),
                        ny + x * math.sin(al) + (y - ny) * math.cos(al) + kw.get("head_dy", 0.0)]
        elif name == "desk_crop":
            for i in range(15, 23):
                vis[i] = 0.10
        elif name == "occluded":
            if 8 <= k <= 21:
                for i in range(15, 23):
                    vis[i] = 0.2
                    px_override[i] = (0.50, 0.20)
        # noise in SW units before the pixel conversion
        if noise is not None:
            for i in range(33):
                q[i][0] += noise[k][i][0]
                q[i][1] += noise[k][i][1]
        pts = {i: _px(q[i]) for i in range(33)}
        if name == "lean_in":
            fin = kw.get("final_scale", 1.2)
            s = 1.0 + (fin - 1.0) * min(max((t - 0.5) / 1.9, 0.0), 1.0)
            pts = {i: (320.0 + s * (p[0] - 320.0), 240.0 + s * (p[1] - 240.0)) for i, p in pts.items()}
        elif name == "swap" and k >= 15:
            pts = {i: (CX + 0.6 * (p[0] - CX) + 200.0, CY + 0.6 * (p[1] - CY)) for i, p in pts.items()}
        frame = []
        for i in range(33):
            if i in px_override:
                frame.append([px_override[i][0], px_override[i][1], vis[i]])
            else:
                frame.append([pts[i][0] / W, pts[i][1] / H, vis[i]])
        ts.append(t)
        frames.append(frame)
    return SkeletonSeq(name, ts, frames, W, H, dict(fs=fs, n=n, noise_sw=noise_sw, seed=seed, **kw))


def _touch_u(t):
    if t <= 1.1:
        return min(max((t - 0.7) / 0.4, 0.0), 1.0)
    if t <= 2.2:
        return 1.0
    return min(max(1.0 - (t - 2.2) / 0.4, 0.0), 1.0)


# --- the eleven named generators ---------------------------------------------------------------------------

def f_calm(**kw): return build("calm", **kw)
def f_fidget(**kw): return build("fidget", **kw)
def f_self_touch(**kw): return build("self_touch", **kw)
def f_freeze(**kw): return build("freeze", **kw)
def f_lean_in(**kw): return build("lean_in", **kw)
def f_arms_crossed(**kw): return build("arms_crossed", **kw)
def f_occluded(**kw): return build("occluded", **kw)
def f_gesture(**kw): return build("gesture", **kw)
def f_tilt(**kw): return build("tilt", **kw)
def f_desk_crop(**kw): return build("desk_crop", **kw)
def f_swap(**kw): return build("swap", **kw)


# --- transforms for the invariance tests -----------------------------------------------------------------------------

def _map_px(seq, fn, W=None, H=None):
    W, H = W or seq.W, H or seq.H
    lm = []
    for frame in seq.lm:
        out = []
        for x, y, v in frame:
            px, py = fn(x * seq.W, y * seq.H)
            out.append([px / W, py / H, v])
        lm.append(out)
    return SkeletonSeq(seq.name, list(seq.t), lm, W, H, dict(seq.params), seq.label)


def translate(seq, dx_px, dy_px):
    return _map_px(seq, lambda x, y: (x + dx_px, y + dy_px))


def scale_about_shoulders(seq, factor):
    """Scale the pixel geometry about the first frame's shoulder midpoint, re-centred on a canvas that still fits it."""
    f0 = seq.lm[0]
    px0 = ((f0[11][0] + f0[12][0]) / 2.0 * seq.W, (f0[11][1] + f0[12][1]) / 2.0 * seq.H)
    grow = max(factor, 1.0) * (1.5 if factor > 1.0 else 1.0)  # headroom so enlarged points stay inside the canvas
    W2, H2 = int(seq.W * grow), int(seq.H * grow)
    c = (W2 / 2.0, H2 / 2.0)
    return _map_px(seq, lambda x, y: (c[0] + factor * (x - px0[0]), c[1] + factor * (y - px0[1])), W2, H2)


def mirror(seq):
    lm = []
    for frame in seq.lm:
        out = [None] * 33
        for i in range(33):
            j = MIRROR.get(i, i)
            x, y, v = frame[j]
            out[i] = [1.0 - x, y, v]
        lm.append(out)
    return SkeletonSeq(seq.name, list(seq.t), lm, seq.W, seq.H, dict(seq.params), seq.label)


def every_other(seq):
    return SkeletonSeq(seq.name, seq.t[::2], seq.lm[::2], seq.W, seq.H, dict(seq.params), seq.label)


def resize_canvas(seq, W, H):
    """Same pixel geometry on a different canvas (aspect-trap fixture)."""
    return _map_px(seq, lambda x, y: (x, y), W, H)
