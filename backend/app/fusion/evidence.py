"""Explicit insufficient-evidence states. Never silently confident."""


def evidence_status(
    face_found: bool, quality: float, top_prob: float, occluded: bool = False
) -> dict:
    reasons = []
    if not face_found:
        reasons.append("no-face-detected")
    if face_found and quality < 0.3:
        reasons.append("poor-detection-quality")
    if face_found and top_prob < 0.4:
        reasons.append("high-uncertainty")
    if occluded:
        reasons.append("possible-occlusion")
    return {"reasons": reasons, "insufficient": bool(reasons)}
