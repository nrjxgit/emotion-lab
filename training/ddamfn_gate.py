"""DDAMFN++ evaluation gate.

DDAMFN++ weights and licence must be verified before the adapter is allowed
to run. Returns (eligible: bool, reason: str) -- never silently integrates
unlicensed weights.
"""

from __future__ import annotations

from pathlib import Path

# Licences considered safe for research use without additional review.
ALLOWED_LICENCES = (
    "mit",
    "apache-2.0",
    "apache 2.0",
    "bsd-3-clause",
    "bsd 3-clause",
    "bsd-2-clause",
    "bsd 2-clause",
    "gpl-3.0-only",
    "gpl-3.0",
    "gpl-2.0-only",
    "gpl-2.0",
)

# Licence text that indicates restrictive/proprietary terms.
RESTRICTIVE_TERMS = (
    "all rights reserved",
    "proprietary",
    "non-commercial",
    "nc",
    "no commercial",
)


def evaluate_gate(path: str) -> tuple[bool, str]:
    """Check DDAMFN++ weights and licence availability.

    Returns (eligible, reason). eligible=False means the adapter must NOT
    run.
    """
    d = Path(path)
    if not d.exists() or not d.is_dir():
        return False, f"weights directory not found: {path}"

    weight_files = (
        "model.pt",
        "model.pth",
        "weights.pt",
        "ddamfn++.pt",
        "model.onnx",
    )
    if not any((d / w).exists() for w in weight_files):
        return False, "no model weights file found"

    licence_files = list(d.glob("LICENSE*")) + list(d.glob("LICENCE*"))
    if not licence_files:
        return False, "no licence file found -- cannot verify redistribution rights"

    licence_text = licence_files[0].read_text(errors="ignore").lower()

    for trigger in RESTRICTIVE_TERMS:
        if trigger in licence_text:
            return (
                False,
                f"licence may restrict research use: {trigger} "
                f"({licence_files[0].name})",
            )

    for allowed in ALLOWED_LICENCES:
        if allowed in licence_text:
            return (
                True,
                f"licence acceptable: {allowed} ({licence_files[0].name})",
            )

    return False, f"licence not in allowlist: {licence_files[0].name}"
