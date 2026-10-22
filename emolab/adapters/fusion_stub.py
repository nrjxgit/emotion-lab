"""Face+body fusion stub.

Side-by-side reporting only: face and body predictions are kept separate with
their own probabilities — no fusion weights are learned and no claim is made
that combining them improves accuracy. This stub is used until matched
labelled evaluation demonstrates improvement.
"""

from __future__ import annotations

from typing import Optional


def fused_side_by_side(
    face: dict,
    body: dict | None,
) -> dict:
    """Return face-only or face+body side-by-side results without fusion.

    Each modality keeps its own probabilities and top-1 label. If body is
    None, returns face-only. Face probabilities are never modified.
    """
    if face is None:
        raise ValueError("face predictions are required")

    out: dict = {
        "mode": "face-only" if body is None else "side-by-side",
        "face": dict(face),
        "body": None,
    }
    if body is not None:
        out["body"] = dict(body)
    return out
