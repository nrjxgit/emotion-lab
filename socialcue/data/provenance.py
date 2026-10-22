"""Provenance manifest builder (EL-006 S2). Pure and deterministic; file IO only via an explicit path."""
from __future__ import annotations

import json
import re
from typing import Optional

from .rights import licence_banner, DatasetAdapter, Decision, RightsRefusal, authorize, canonical_json, sha256_of

SCHEMA = "socialcue-provenance-v1"


def manifest_from_decision(d: Decision, counts: dict, config_sha256=None) -> dict:
    """Pure provenance from the SAME accepted decision, never re-authorize/read adapter.

    Integrity record only; accepting a Decision does not authenticate authority.
    """
    if type(d) is not Decision or d.decision != "allowed":
        raise ValueError("accepted authorization decision required")
    if not isinstance(counts, dict) or any(type(v) is not int or v < 0 for v in counts.values()):
        raise ValueError("counts must be non-negative integers")
    body = {"schema": SCHEMA,
            "adapter": {"name": d.adapter, "licence_tag": d.licence_tag,
                        "source_class": d.source_class, "scope": d.data_scope},
            "use": d.use, "run_scope": d.run_scope,
            "licence_banner": licence_banner(d.run_scope, d.source_class),
            "decision": {"result": "allowed", "code": None, "reason": None,
                         "allowlist_sha256": d.allowlist_sha256},
            "counts": dict(sorted(counts.items())), "hb1_config_sha256": config_sha256}
    return {**body, "manifest_sha256": sha256_of(body)}


def build_manifest(adapter: DatasetAdapter, use: str, run_scope: str, counts: dict,
                   config_sha256: Optional[str] = None) -> dict:
    """Record the gate decision (allowed or refused) with the run's counts and a canonical sha256.

    A refusal is recorded, not raised, so a refused attempt still leaves an auditable manifest.
    """
    use = use if isinstance(use, str) else repr(use)
    run_scope = run_scope if isinstance(run_scope, str) else repr(run_scope)
    if not isinstance(counts, dict) or any(not isinstance(v, int) or isinstance(v, bool) or v < 0
                                           for v in counts.values()):
        raise ValueError("counts must be a dict of non-negative integers")
    try:
        d = authorize(adapter, use, run_scope)
        decision = {"result": "allowed", "code": None, "reason": None, "allowlist_sha256": d.allowlist_sha256}
    except RightsRefusal as r:
        decision = {"result": "refused", "code": r.code, "reason": r.reason, "allowlist_sha256": None}
    body = {
        "schema": SCHEMA,
        "adapter": {
            "name": str(getattr(adapter, "name", None)),
            "licence_tag": str(getattr(adapter, "licence_tag", None)),
            "source_class": str(getattr(adapter, "source_class", None)),
            "scope": str(getattr(adapter, "scope", None)),
        },
        "use": use,
        "run_scope": run_scope,
        "licence_banner": None,
        "decision": decision,
        "counts": dict(sorted(counts.items())),
        "hb1_config_sha256": config_sha256,
    }
    body["licence_banner"] = licence_banner(run_scope, body["adapter"]["source_class"])
    return {**body, "manifest_sha256": sha256_of(body)}


def verify_manifest(manifest) -> bool:
    """Integrity check only (tamper-evident against accidental edits); it does not authenticate the author."""
    if not isinstance(manifest, dict):
        return False
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    try:
        return manifest.get("manifest_sha256") == sha256_of(body)
    except (TypeError, ValueError):
        return False


def write_manifest(manifest: dict, path: str) -> None:
    """Write to an explicit path; refuses to overwrite an existing file and any '..' path component."""
    if ".." in re.split(r"[\\/]+", str(path)):
        raise ValueError("path must not contain '..' components")
    with open(path, "x", encoding="utf-8") as fh:
        fh.write(json.dumps(manifest, sort_keys=True, indent=2) + "\n")


__all__ = ["build_manifest", "verify_manifest", "write_manifest", "canonical_json", "SCHEMA"]
