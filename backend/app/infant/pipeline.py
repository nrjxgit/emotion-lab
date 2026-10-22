"""Infant pipeline — EXPERIMENTAL. No diagnosis of hunger/pain/illness."""


def infant_predict(face: dict, aus: list) -> dict:
    m = {a.get("code"): float(a.get("intensity", 0)) for a in (aus or [])}
    smile = m.get("AU12", 0)
    cry = max(m.get("AU04", 0), m.get("AU25", 0))
    if smile > 0.6:
        label = "positive-affect-cue"
    elif cry > 0.6:
        label = "cry-related-facial-movement"
    else:
        label = "no-strong-cue"
    return {
        "label": label,
        "experimental": True,
        "warning": "Adult models not assumed valid for infants; guardian consent required; no medical inference.",
        "uncertainty": 0.8,
    }
