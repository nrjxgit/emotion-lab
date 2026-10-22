"""Rights-gate hardening from the independent S2 review (EL-006 S24b). Synthetic/declaration adapters only."""
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from socialcue.data import provenance, registry, rights  # noqa: E402
from socialcue.data.declarations import CremaDDeclaration, RavdessDeclaration  # noqa: E402
from socialcue.data.rights import DatasetAdapter, RightsRefusal, authorize  # noqa: E402
from socialcue.data.synthetic_adapter import SyntheticAdapter  # noqa: E402


def code(adapter, use="development_test", scope="synthetic"):
    with pytest.raises(RightsRefusal) as exc:
        authorize(adapter, use, scope)
    return exc.value.code


# --- F1: the operator block lives in the allowlist --------------------------------

@pytest.mark.parametrize("decl", [RavdessDeclaration, CremaDDeclaration])
def test_blocked_status_is_enforced_by_the_allowlist_not_by_adapter_code(decl):
    entry = rights.load_allowlist()[0][decl.name]
    assert entry["status"] == "blocked_pending_operator" and "operator" in entry["reason"]

    class Cleared(decl):
        blocked_reason = None

    class AlsoCleared(decl):
        @property
        def blocked_reason(self):
            return ""

    for cls in (decl, Cleared, AlsoCleared):
        assert code(cls(), "research_training", "research_noncommercial") == "BLOCKED_NEEDS_OPERATOR_DECISION"
        assert code(cls(), "research_evaluation", "research_noncommercial") == "BLOCKED_NEEDS_OPERATOR_DECISION"
    with pytest.raises(RightsRefusal):
        Cleared().items("research_evaluation", "research_noncommercial")


def test_use_and_scope_refusals_still_take_precedence_over_the_block():
    a = RavdessDeclaration()
    assert code(a, "commercial_training", "commercial_clean") == "USE_NOT_ALLOWED"
    assert code(a, "research_evaluation", "commercial_clean") == "SCOPE_MISMATCH"


def test_blocked_entry_needs_a_reason_and_valid_status():
    base = {"name": "x", "licence_tag": "t", "source_class": "research_corpus", "allowed_uses": [],
            "scopes": [], "status": "blocked_pending_operator"}
    with pytest.raises(RightsRefusal):
        rights.parse_allowlist({"datasets": [base]})
    ok = rights.parse_allowlist({"datasets": [{**base, "reason": "pending"}]})
    assert ok["x"]["reason"] == "pending"
    with pytest.raises(RightsRefusal):
        rights.parse_allowlist({"datasets": [{**base, "status": "mostly_granted", "reason": "x"}]})


def test_only_synthetic_is_granted():
    entries, _ = rights.load_allowlist()
    assert {n for n, e in entries.items() if e["status"] == "granted"} == {"synthetic", "synthetic_commercial_clean"}


# --- F2: the gate is final ----------------------------------------------------------

def test_overriding_authorize_items_or_participants_is_rejected():
    for method in ("authorize", "participants", "items"):
        with pytest.raises(TypeError, match="rights gate is final"):
            type("Evil", (SyntheticAdapter,), {method: lambda self, *a, **k: None})
    with pytest.raises(TypeError):
        type("Evil2", (DatasetAdapter,), {"items": lambda self, u, s: []})


def test_instance_level_replacement_of_the_gate_is_rejected():
    a = SyntheticAdapter()
    for method in ("authorize", "participants", "items"):
        with pytest.raises(TypeError, match="final"):
            setattr(a, method, lambda *x, **k: [])


def test_data_methods_use_the_module_level_gate(monkeypatch):
    calls = []
    real = rights.authorize
    monkeypatch.setattr(rights, "authorize", lambda adapter, use, scope: calls.append((use, scope)) or real(adapter, use, scope))
    a = SyntheticAdapter()
    a.items("development_test", "synthetic")
    a.participants("development_test", "synthetic")
    assert calls == [("development_test", "synthetic")] * 2
    with pytest.raises(RightsRefusal):
        a.items("whatever", "whatever")


# --- F3: registry key, one-shot snapshot ----------------------------------------------

def test_registry_key_must_equal_the_declared_name():
    class Liar(SyntheticAdapter):
        pass

    registry.register_adapter("foo", Liar)
    try:
        with pytest.raises(RightsRefusal) as exc:
            registry.open_adapter("foo", "development_test", "synthetic")
        assert exc.value.code == "ADAPTER_METADATA_MISMATCH"
    finally:
        registry._REGISTRY.pop("foo", None)
    assert registry.open_adapter("synthetic", "development_test", "synthetic").name == "synthetic"


def test_stateful_declarations_are_read_once():
    reads = []

    class Flaky(SyntheticAdapter):
        @property
        def scope(self):
            reads.append(1)
            return "synthetic" if len(reads) == 1 else "commercial_clean"

    d = authorize(Flaky(), "development_test", "synthetic")
    assert d.data_scope == "synthetic" and len(reads) == 1  # snapshot, not re-read per check

    class ChangingClass(SyntheticAdapter):
        n = 0

        @property
        def source_class(self):
            type(self).n += 1
            return "synthetic" if type(self).n == 1 else "movie"

    assert code(ChangingClass()) == "ADAPTER_METADATA_MISMATCH"


def test_unreadable_declarations_refuse_instead_of_crashing():
    class Broken(SyntheticAdapter):
        @property
        def licence_tag(self):
            raise RuntimeError("no")

    assert code(Broken()) == "ADAPTER_METADATA_MISMATCH"


def test_gate_scope_is_documented():
    doc = rights.__doc__
    assert "DECLARATIONS" in doc and "BEHAVIOUR" in doc


# --- F4: frozen tables ----------------------------------------------------------------

def test_allowlist_entries_cannot_be_mutated():
    entries, digest = rights.load_allowlist()
    with pytest.raises(TypeError):
        entries["casme2"]["status"] = "granted"
    with pytest.raises(TypeError):
        entries["casme2"] = {}
    assert isinstance(entries["synthetic"]["allowed_uses"], tuple)
    with pytest.raises(AttributeError):
        entries["synthetic"]["allowed_uses"].append("commercial_training")
    assert rights.load_allowlist()[0]["casme2"]["status"] == "not_granted" and digest == rights.ALLOWLIST_SHA256


def test_module_tables_are_read_only():
    with pytest.raises(TypeError):
        rights.SCOPE_COVERS["research_noncommercial"] = frozenset(rights.SCOPES)
    with pytest.raises(TypeError):
        rights.LICENCE_BANNER["x"] = "y"
    with pytest.raises(TypeError):
        registry._FACTORIES["evil"] = SyntheticAdapter
    with pytest.raises(AttributeError):
        rights.SCOPE_COVERS["synthetic"].add("x")
    assert "evil" not in registry.registered_names()


# --- F5: provenance helpers --------------------------------------------------------------

def test_write_manifest_rejects_parent_components(tmp_path):
    (tmp_path / "sub").mkdir()
    m = provenance.build_manifest(SyntheticAdapter(), "development_test", "synthetic", {"items": 1})
    for bad in (str(tmp_path / "sub" / ".." / "x.json"), "..\\x.json", "a/../b.json", "../x.json"):
        with pytest.raises(ValueError, match=r"\.\."):
            provenance.write_manifest(m, bad)
    assert not (tmp_path / "x.json").exists()
    provenance.write_manifest(m, str(tmp_path / "sub" / "ok.json"))
    assert json.loads((tmp_path / "sub" / "ok.json").read_text(encoding="utf-8")) == m


@pytest.mark.parametrize("bad", [None, [], "x", 5, {"a": {1, 2}}, {"manifest_sha256": object()}])
def test_verify_manifest_returns_false_on_garbage(bad):
    assert provenance.verify_manifest(bad) is False


def test_verify_manifest_is_integrity_not_authenticity():
    assert "authenticate" in provenance.verify_manifest.__doc__


# --- F6: unhashable inputs ----------------------------------------------------------------

@pytest.mark.parametrize("use,scope", [(["x"], "synthetic"), ("development_test", ["synthetic"]),
                                       ({"a": 1}, {"b": 2}), (None, None), (5, 7)])
def test_unhashable_or_non_string_inputs_are_refusals(use, scope):
    with pytest.raises(RightsRefusal):
        authorize(SyntheticAdapter(), use, scope)


def test_unhashable_declarations_are_refusals():
    class BadName(SyntheticAdapter):
        name = ["synthetic"]

    class BadUses(SyntheticAdapter):
        allowed_uses = (["development_test"],)

    assert code(BadName()) == "DATASET_NOT_ALLOWLISTED"
    assert code(BadUses()) == "ADAPTER_METADATA_MISMATCH"


def test_build_manifest_records_unhashable_attempts_as_refused():
    m = provenance.build_manifest(SyntheticAdapter(), ["x"], {"y": 1}, {"items": 0})
    assert m["decision"]["result"] == "refused" and m["decision"]["code"] in {"USE_NOT_ALLOWED", "SCOPE_MISMATCH"}
    assert provenance.verify_manifest(m)
    m2 = provenance.build_manifest(SyntheticAdapter(), "development_test", ["synthetic"], {})
    assert m2["decision"]["result"] == "refused" and m2["licence_banner"] == "unknown scope"
