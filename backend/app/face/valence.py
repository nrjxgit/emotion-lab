def valence_arousal(aus: list) -> tuple:
    m = {a.get("code"): float(a.get("intensity", 0)) for a in (aus or [])}
    v = m.get("AU12", 0) * 0.9 - m.get("AU04", 0) * 0.6 - m.get("AU15", 0) * 0.7
    a = max(m.get("AU05", 0), m.get("AU25", 0), 0.0)
    return (float(max(-1, min(1, v))), float(max(0, min(1, a))))
