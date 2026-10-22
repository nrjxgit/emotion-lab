CLASSES = [
    "happiness",
    "sadness",
    "anger",
    "surprise",
    "fear",
    "disgust",
    "neutral",
    "unknown",
]


def _norm(d: dict) -> dict:
    full = {c: float(d.get(c, 0.05)) for c in CLASSES}
    s = sum(full.values()) or 1.0
    return {k: v / s for k, v in full.items()}


def fuse(face: dict | None, body: dict | None, quality: dict) -> dict:
    fq = float(quality.get("face_q", 0.0))
    bq = float(quality.get("body_q", 0.0))
    if face is None and body is None:
        return {
            "probs": _norm({"unknown": 1.0}),
            "confidence": 0.2,
            "uncertainty": 0.8,
            "evidence": ["no modality visible"],
            "mode": "none",
        }
    if face is None:
        p = _norm(body["probs"])
        c = body["confidence"] * (0.5 + 0.5 * bq)
        return {
            "probs": p,
            "confidence": float(min(0.9, c)),
            "uncertainty": float(1 - min(0.9, c)),
            "evidence": ["body-only"],
            "mode": "body-only",
        }
    if body is None:
        p = _norm(face["probs"])
        c = face["confidence"] * (0.5 + 0.5 * fq)
        return {
            "probs": p,
            "confidence": float(min(0.9, c)),
            "uncertainty": float(1 - min(0.9, c)),
            "evidence": ["face-only"],
            "mode": "face-only",
        }
    fp, bp = _norm(face["probs"]), _norm(body["probs"])
    tot = fq + bq or 1.0
    wf, wb = fq / tot, bq / tot
    probs = {c: wf * fp[c] + wb * bp[c] for c in CLASSES}
    # anti-overconfidence: cap when quality low or modalities disagree
    agree = sum(min(fp[c], bp[c]) for c in CLASSES)
    cap = 0.55 + 0.35 * max(fq, bq)  # max 0.9, low-q ~0.6
    conf = min(max(probs.values()), cap, 0.5 + 0.5 * agree + 0.2)
    conf = float(min(conf, 0.94))
    return {
        "probs": probs,
        "confidence": conf,
        "uncertainty": float(1 - conf),
        "evidence": [f"face_w={wf:.2f}", f"body_w={wb:.2f}", f"agree={agree:.2f}"],
        "mode": "fusion",
    }
