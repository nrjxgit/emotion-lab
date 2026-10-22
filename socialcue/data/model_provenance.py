"""Pure fitting-input declaration record; no fitting, IO or asset certification."""
from .rights import ALLOWLIST_SHA256, sha256_of

# Explicit reviewed policy. Never infer permission from a name suffix, NC-tag
# spelling, caller-supplied scope, or a registry factory supplied at runtime.
_COMMERCIAL_SYNTHETIC = {
    "synthetic": "synthetic-manufactured",
    "synthetic_commercial_clean": "synthetic-commercial-clean",
}


def build_model_provenance(adapter_names):
    """Record the complete caller-declared list of fitting adapter registry names.

    yes: all names are these reviewed, commercial-training-permitted synthetic
    adapters. no: a blocked, unknown or malformed name/input. unknown: no fitting
    history supplied. This checks declarations, not history completeness, actual
    dataset contents, rights evidence, weights or legal obligations. It does not
    authorize training or certify an existing model; real retraining stays gated.
    Pure: no registry factory, adapter data method, allowlist read or other IO.
    """
    valid = type(adapter_names) in (list, tuple) and all(type(n) is str for n in adapter_names)
    names = sorted(set(adapter_names)) if valid else []
    rows = [{"name": name, "licence_tag": _COMMERCIAL_SYNTHETIC.get(name),
             "commercial_training_reviewed": name in _COMMERCIAL_SYNTHETIC} for name in names]
    clean = "no" if not valid or any(not r["commercial_training_reviewed"] for r in rows) else "yes" if rows else "unknown"
    body = {"schema": "socialcue-model-provenance-v1", "commercial_clean": clean,
            "adapters": rows, "allowlist_sha256": ALLOWLIST_SHA256,
            "training_history_verified": False,
            "basis": "caller-declared fitting adapter names and reviewed synthetic policy only",
            "reason": "invalid_adapter_list" if not valid else "missing_fitting_history" if not rows
                      else "all_reviewed_commercial_synthetic" if clean == "yes" else "blocked_or_unknown_adapter"}
    return {**body, "record_sha256": sha256_of(body)}
