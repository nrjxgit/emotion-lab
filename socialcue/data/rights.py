"""Fail-closed rights gate and DatasetAdapter protocol (EL-006 S2). Research prototype.

The gate has no override: no keyword, flag or environment variable relaxes it.
Allowlist changes are reviewed source changes and must update ALLOWLIST_SHA256.
Nothing here touches the filesystem at import; the allowlist is read only when
a check runs, from the one file next to this module.

What the gate does and does not check: it checks an adapter's DECLARATIONS
(name, licence tag, source class, scope, allowed uses) against the reviewed
allowlist, once per call from a snapshot, and it is the only path the data
methods use. It cannot check BEHAVIOUR: an adapter that declares a harmless
dataset but reads something else would pass. Adapters are reviewed source, and
the registry binds each registry key to the adapter's declared name. Blocked
datasets are blocked in the allowlist itself, not by adapter code. For entries marked ``requires_ethics_approval`` the
gate also requires a non-empty ``ethics_approval_token`` string on the adapter: that checks the field EXISTS, not that
an approval is real or valid (a string, like a boolean declaration, is not an approval; the operator and the ethics
body decide).
"""
from __future__ import annotations

import abc
import functools
import hashlib
import json
import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional

FORBIDDEN_SOURCE_CLASSES = frozenset({"movie", "tv", "youtube", "web_scrape", "social_media", "unknown"})
KNOWN_SOURCE_CLASSES = frozenset({"synthetic", "research_corpus", "by_application_corpus", "consented_pilot"})
STATUSES = frozenset({"granted", "not_granted", "blocked_pending_operator"})
USES = frozenset({"development_test", "research_evaluation", "research_training",
                  "commercial_evaluation", "commercial_training"})
SCOPES = frozenset({"synthetic", "research_noncommercial", "commercial_clean"})
# data scope -> run scopes the data may serve
SCOPE_COVERS = MappingProxyType({
    "synthetic": frozenset({"synthetic", "research_noncommercial", "commercial_clean"}),
    "commercial_clean": frozenset({"commercial_clean", "research_noncommercial"}),
    "research_noncommercial": frozenset({"research_noncommercial"}),
})
LICENCE_BANNER = MappingProxyType({
    "synthetic": "synthetic data, no third-party rights",
    "research_noncommercial": "research, non-commercial",
    "commercial_clean": "commercial-clean scope, declarations only",
})
def licence_banner(run_scope: str, source_class: str) -> str:
    """Describe declared scope without inventing human consent for manufactured data."""
    banner = LICENCE_BANNER.get(run_scope, "unknown scope")
    if source_class == "synthetic" and run_scope in {"commercial_clean", "research_noncommercial"}:
        scope = "commercial-clean" if run_scope == "commercial_clean" else "research non-commercial"
        return f"{scope} scope, SYNTHETIC manufactured data - not real data; no human-consent claim"
    return banner


# Names a subclass may never redefine (the gate lives in them).
_FINAL_METHODS = frozenset({"authorize", "participants", "items"})

_ALLOWLIST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rights_allowlist.json")
# sha256 of the canonical JSON of rights_allowlist.json (sorted keys, compact separators).
ALLOWLIST_SHA256 = "bf94ef47cf55da6090a32875989a69cd3f02a36b183b8e90f6c68c526dde7b03"


class RightsRefusal(Exception):
    """Typed refusal. ``code`` is a stable reason code; there is no way to waive it."""

    def __init__(self, code: str, reason: str):
        super().__init__(f"{code}: {reason}")
        self.code = code
        self.reason = reason


@dataclass(frozen=True)
class Decision:
    adapter: str
    licence_tag: str
    source_class: str
    data_scope: str
    run_scope: str
    use: str
    decision: str
    allowlist_sha256: str


class DatasetAdapter(abc.ABC):
    """Adapter protocol. Data access goes through ``participants``/``items``, which run the gate first.

    ``authorize``, ``participants`` and ``items`` are final: subclasses that define them
    (or assign them on an instance) are rejected with TypeError.
    """

    name: str
    licence_tag: str
    source_class: str
    allowed_uses: tuple
    scope: str
    blocked_reason: Optional[str] = None  # informational only; the allowlist decides

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        clash = _FINAL_METHODS & set(cls.__dict__)
        if clash:
            raise TypeError(f"{cls.__name__} may not override {sorted(clash)}: the rights gate is final")

    def __setattr__(self, key, value):
        if key in _FINAL_METHODS:
            raise TypeError(f"{key!r} is final and cannot be replaced on an adapter")
        super().__setattr__(key, value)

    def authorize(self, use: str, run_scope: str) -> Decision:
        return authorize(self, use, run_scope)

    def participants(self, use: str, run_scope: str) -> list:
        authorize(self, use, run_scope)  # module-level gate, never self.authorize
        return self._participants()

    def items(self, use: str, run_scope: str) -> list:
        authorize(self, use, run_scope)
        return self._items()

    @abc.abstractmethod
    def _participants(self) -> list: ...

    @abc.abstractmethod
    def _items(self) -> list: ...

    @abc.abstractmethod
    def label_map(self) -> dict: ...

    @abc.abstractmethod
    def group_id(self, item) -> str: ...


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_of(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode("ascii")).hexdigest()


def _freeze(obj):
    if isinstance(obj, dict):
        return MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_freeze(v) for v in obj)
    return obj


def parse_allowlist(doc: dict) -> MappingProxyType:
    """Validate a parsed allowlist document; returns a frozen {name: entry}. Raises RightsRefusal(ALLOWLIST_INVALID)."""
    try:
        entries = doc["datasets"]
        out = {}
        for e in entries:
            name = e["name"]
            if name in out:
                raise ValueError(f"duplicate entry {name}")
            if e["status"] not in STATUSES:
                raise ValueError(f"bad status for {name}")
            if e["source_class"] in FORBIDDEN_SOURCE_CLASSES or e["source_class"] not in KNOWN_SOURCE_CLASSES:
                raise ValueError(f"forbidden or unknown source class for {name}")
            if not set(e["allowed_uses"]) <= USES or not set(e["scopes"]) <= SCOPES:
                raise ValueError(f"bad uses or scopes for {name}")
            if e["status"] == "not_granted" and (e["allowed_uses"] or e["scopes"]):
                raise ValueError(f"not_granted entry {name} must list no uses or scopes")
            if e["status"] == "blocked_pending_operator" and not (isinstance(e.get("reason"), str) and e["reason"]):
                raise ValueError(f"blocked_pending_operator entry {name} needs a reason")
            if "requires_ethics_approval" in e and not isinstance(e["requires_ethics_approval"], bool):
                raise ValueError(f"requires_ethics_approval for {name} must be a boolean")
            out[name] = _freeze(e)
        return MappingProxyType(out)
    except RightsRefusal:
        raise
    except Exception as exc:  # malformed allowlist => refuse everything
        raise RightsRefusal("ALLOWLIST_INVALID", f"allowlist malformed: {exc}") from exc


@functools.lru_cache(maxsize=1)
def load_allowlist() -> tuple:
    """Read the shipped allowlist once; returns (frozen entries, sha256). Fails closed on any mismatch.

    The result is deeply immutable, so callers cannot poison the cache. A success is cached for
    the process lifetime: an on-disk edit after the first load is not noticed until restart.
    """
    try:
        with open(_ALLOWLIST_PATH, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except Exception as exc:
        raise RightsRefusal("ALLOWLIST_INVALID", f"allowlist unreadable: {exc}") from exc
    digest = sha256_of(doc)
    if digest != ALLOWLIST_SHA256:
        raise RightsRefusal("ALLOWLIST_TAMPERED", "allowlist hash does not match ALLOWLIST_SHA256")
    return parse_allowlist(doc), digest


def _snapshot(adapter) -> dict:
    """Read every declaration exactly once so stateful properties cannot change between checks."""
    try:
        return {
            "name": adapter.name,
            "licence_tag": adapter.licence_tag,
            "source_class": adapter.source_class,
            "scope": adapter.scope,
            "allowed_uses": tuple(adapter.allowed_uses),
            "blocked_reason": adapter.blocked_reason,
            "ethics_approval_token": getattr(adapter, "ethics_approval_token", None),
        }
    except Exception as exc:
        raise RightsRefusal("ADAPTER_METADATA_MISMATCH", f"adapter declarations unreadable: {type(exc).__name__}") from exc


def authorize(adapter: DatasetAdapter, use: str, run_scope: str) -> Decision:
    """Allow ``adapter`` for ``use`` under ``run_scope`` or raise RightsRefusal. No bypass parameters."""
    try:
        source_class = adapter.source_class
    except Exception:
        source_class = None
    if not isinstance(source_class, str) or source_class in FORBIDDEN_SOURCE_CLASSES \
            or source_class not in KNOWN_SOURCE_CLASSES:
        raise RightsRefusal("SOURCE_CLASS_FORBIDDEN", f"source class {source_class!r} is never permitted")
    snap = _snapshot(adapter)
    if snap["source_class"] != source_class:  # changed between reads: refuse
        raise RightsRefusal("ADAPTER_METADATA_MISMATCH", "source class changed between reads")
    entries, digest = load_allowlist()
    name = snap["name"]
    if not isinstance(name, str) or name not in entries:
        raise RightsRefusal("DATASET_NOT_ALLOWLISTED", f"dataset {name!r} is not on the allowlist")
    entry = entries[name]
    if entry["status"] == "not_granted":
        raise RightsRefusal("NOT_GRANTED", f"dataset {name!r} is by application and not granted")
    try:
        meta_ok = (snap["licence_tag"] == entry["licence_tag"] and source_class == entry["source_class"]
                   and snap["scope"] in entry["scopes"] and set(snap["allowed_uses"]) <= set(entry["allowed_uses"]))
    except TypeError:
        meta_ok = False
    if not meta_ok:
        raise RightsRefusal("ADAPTER_METADATA_MISMATCH", f"adapter {name!r} metadata disagrees with the allowlist")
    if not isinstance(use, str) or use not in USES:
        raise RightsRefusal("USE_NOT_ALLOWED", f"unknown use {use!r}")
    if use not in entry["allowed_uses"] or use not in snap["allowed_uses"]:
        raise RightsRefusal("USE_NOT_ALLOWED", f"use {use!r} is outside the licence ({entry['licence_tag']})")
    if not isinstance(run_scope, str) or run_scope not in SCOPES or run_scope not in SCOPE_COVERS.get(snap["scope"], frozenset()):
        raise RightsRefusal("SCOPE_MISMATCH", f"data scope {snap['scope']!r} cannot serve run scope {run_scope!r}")
    if entry.get("requires_ethics_approval") is True:
        token = snap["ethics_approval_token"]
        if not isinstance(token, str) or len(token.strip()) < 8:
            raise RightsRefusal("ETHICS_APPROVAL_MISSING",
                                f"dataset {name!r} needs an ethics_approval_token string on the adapter "
                                "(the gate checks that the field exists; it cannot verify that an approval is real)")
    if entry["status"] == "blocked_pending_operator":
        raise RightsRefusal("BLOCKED_NEEDS_OPERATOR_DECISION", entry["reason"])
    if snap["blocked_reason"]:
        raise RightsRefusal("BLOCKED_NEEDS_OPERATOR_DECISION", str(snap["blocked_reason"]))
    return Decision(name, entry["licence_tag"], source_class, snap["scope"], run_scope, use, "allowed", digest)
