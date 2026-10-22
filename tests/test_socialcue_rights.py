"""Rights gate, adapter protocol, registry and provenance manifest (EL-006 S2). No real data paths."""
import hashlib
import inspect
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from socialcue import state as hb1  # noqa: E402
from socialcue.data import declarations, provenance, registry, rights  # noqa: E402
from socialcue.data.rights import DatasetAdapter, RightsRefusal, authorize  # noqa: E402
from socialcue.data.synthetic_adapter import SyntheticAdapter  # noqa: E402

PINNED_ALLOWLIST_SHA256 = "bf94ef47cf55da6090a32875989a69cd3f02a36b183b8e90f6c68c526dde7b03"
DATA_DIR = ROOT / "socialcue" / "data"


class FakeAdapter(DatasetAdapter):
    """Test double; every attribute can be set per instance."""

    name = "synthetic"
    licence_tag = "synthetic-manufactured"
    source_class = "synthetic"
    allowed_uses = ("development_test",)
    scope = "synthetic"
    blocked_reason = None

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def _participants(self):
        return ["p"]

    def _items(self):
        return ["i"]

    def label_map(self):
        return {}

    def group_id(self, item):
        return "g"


def _code(adapter, use="development_test", scope="synthetic"):
    with pytest.raises(RightsRefusal) as exc:
        authorize(adapter, use, scope)
    return exc.value.code


# --- source classes ----------------------------------------------------------

@pytest.mark.parametrize("cls", ["movie", "tv", "youtube", "web_scrape", "social_media", "unknown"])
def test_forbidden_source_classes_are_refused_even_under_an_allowlisted_name(cls):
    assert _code(FakeAdapter(source_class=cls)) == "SOURCE_CLASS_FORBIDDEN"
    # a real allowlisted name cannot launder a forbidden class
    assert _code(FakeAdapter(name="ravdess", licence_tag="CC BY-NC-SA 4.0", source_class=cls,
                             allowed_uses=("research_evaluation",), scope="research_noncommercial"),
                 "research_evaluation", "research_noncommercial") == "SOURCE_CLASS_FORBIDDEN"


@pytest.mark.parametrize("cls", [None, "", "streaming", "dvd_rip", 7])
def test_unrecognised_source_class_is_treated_as_unknown(cls):
    assert _code(FakeAdapter(source_class=cls)) == "SOURCE_CLASS_FORBIDDEN"


# --- allowlist ---------------------------------------------------------------

def test_unknown_dataset_name_is_refused():
    assert _code(FakeAdapter(name="affectnet")) == "DATASET_NOT_ALLOWLISTED"
    assert _code(FakeAdapter(name="")) == "DATASET_NOT_ALLOWLISTED"
    assert _code(FakeAdapter(name=None)) == "DATASET_NOT_ALLOWLISTED"
    assert _code(FakeAdapter(name="Synthetic")) == "DATASET_NOT_ALLOWLISTED"  # exact match only


@pytest.mark.parametrize("name", ["casme2", "samm", "smic"])
def test_by_application_sets_are_not_granted(name):
    entries, _ = rights.load_allowlist()
    assert entries[name]["status"] == "not_granted" and entries[name]["licence_tag"] == "by-application"
    a = FakeAdapter(name=name, licence_tag="by-application", source_class="by_application_corpus",
                    allowed_uses=("research_evaluation",), scope="research_noncommercial")
    assert _code(a, "research_evaluation", "research_noncommercial") == "NOT_GRANTED"


def test_allowlist_lists_only_the_intended_datasets():
    entries, _ = rights.load_allowlist()
    assert set(entries) == {"synthetic", "synthetic_commercial_clean", "ravdess", "crema_d", "casme2", "samm", "smic", "pilot_consented"}
    assert entries["ravdess"]["licence_tag"] == "CC BY-NC-SA 4.0"
    assert entries["crema_d"]["licence_tag"] == "ODbL 1.0 / DbCL 1.0"
    assert [n for n, e in entries.items() if e["status"] == "granted" and e["source_class"] == "synthetic"] == ["synthetic", "synthetic_commercial_clean"]
    assert all(e["source_class"] not in rights.FORBIDDEN_SOURCE_CLASSES for e in entries.values())


def test_adapter_cannot_claim_more_than_the_allowlist():
    base = dict(name="ravdess", licence_tag="CC BY-NC-SA 4.0", source_class="research_corpus",
                allowed_uses=("research_evaluation",), scope="research_noncommercial")
    assert _code(FakeAdapter(**{**base, "licence_tag": "CC0"}), "research_evaluation", "research_noncommercial") \
        == "ADAPTER_METADATA_MISMATCH"
    assert _code(FakeAdapter(**{**base, "allowed_uses": ("research_evaluation", "commercial_training")}),
                 "research_evaluation", "research_noncommercial") == "ADAPTER_METADATA_MISMATCH"
    assert _code(FakeAdapter(**{**base, "scope": "commercial_clean"}), "research_evaluation",
                 "research_noncommercial") == "ADAPTER_METADATA_MISMATCH"


@pytest.mark.parametrize("doc", [
    {},
    {"datasets": [{"name": "x"}]},
    {"datasets": [{"name": "a", "licence_tag": "t", "source_class": "youtube", "allowed_uses": [], "scopes": [], "status": "granted"}]},
    {"datasets": [{"name": "a", "licence_tag": "t", "source_class": "research_corpus", "allowed_uses": ["research_evaluation"], "scopes": ["research_noncommercial"], "status": "maybe"}]},
    {"datasets": [{"name": "a", "licence_tag": "t", "source_class": "by_application_corpus", "allowed_uses": ["research_evaluation"], "scopes": ["research_noncommercial"], "status": "not_granted"}]},
])
def test_malformed_allowlist_documents_are_rejected(doc):
    with pytest.raises(RightsRefusal) as exc:
        rights.parse_allowlist(doc)
    assert exc.value.code == "ALLOWLIST_INVALID"


def test_duplicate_allowlist_entries_are_rejected():
    e = {"name": "a", "licence_tag": "t", "source_class": "research_corpus", "allowed_uses": [], "scopes": [], "status": "granted"}
    with pytest.raises(RightsRefusal):
        rights.parse_allowlist({"datasets": [e, dict(e)]})


def test_allowlist_file_hash_is_pinned():
    doc = json.loads((DATA_DIR / "rights_allowlist.json").read_text(encoding="utf-8"))
    canonical = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert hashlib.sha256(canonical.encode("ascii")).hexdigest() == PINNED_ALLOWLIST_SHA256
    assert rights.ALLOWLIST_SHA256 == PINNED_ALLOWLIST_SHA256
    assert rights.load_allowlist()[1] == PINNED_ALLOWLIST_SHA256


def test_tampered_allowlist_fails_closed(monkeypatch, tmp_path):
    doc = json.loads((DATA_DIR / "rights_allowlist.json").read_text(encoding="utf-8"))
    for e in doc["datasets"]:
        if e["name"] == "casme2":
            e.update(status="granted", allowed_uses=["research_evaluation"], scopes=["research_noncommercial"])
    p = tmp_path / "rights_allowlist.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    monkeypatch.setattr(rights, "_ALLOWLIST_PATH", str(p))
    rights.load_allowlist.cache_clear()
    try:
        assert _code(SyntheticAdapter()) == "ALLOWLIST_TAMPERED"
    finally:
        monkeypatch.undo()
        rights.load_allowlist.cache_clear()
    assert authorize(SyntheticAdapter(), "development_test", "synthetic").decision == "allowed"


def test_missing_allowlist_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(rights, "_ALLOWLIST_PATH", str(tmp_path / "absent.json"))
    rights.load_allowlist.cache_clear()
    try:
        assert _code(SyntheticAdapter()) == "ALLOWLIST_INVALID"
    finally:
        monkeypatch.undo()
        rights.load_allowlist.cache_clear()


# --- uses and scopes ---------------------------------------------------------

def test_commercial_use_of_a_non_commercial_licence_is_refused():
    for name in ("ravdess", "crema_d"):
        a = registry._FACTORIES[name]()
        assert _code(a, "commercial_training", "commercial_clean") == "USE_NOT_ALLOWED"
        assert _code(a, "commercial_evaluation", "research_noncommercial") == "USE_NOT_ALLOWED"
        assert _code(a, "made_up_use", "research_noncommercial") == "USE_NOT_ALLOWED"


def test_scope_mismatch_is_refused():
    a = registry._FACTORIES["ravdess"]()
    assert _code(a, "research_evaluation", "commercial_clean") == "SCOPE_MISMATCH"
    assert _code(a, "research_evaluation", "synthetic") == "SCOPE_MISMATCH"
    assert _code(a, "research_evaluation", "nonsense") == "SCOPE_MISMATCH"


def test_scope_rules_for_commercial_clean_and_synthetic_data():
    assert rights.SCOPE_COVERS["research_noncommercial"] == {"research_noncommercial"}
    assert "research_noncommercial" in rights.SCOPE_COVERS["commercial_clean"]
    assert rights.SCOPE_COVERS["synthetic"] == set(rights.SCOPES)


def test_declaration_adapters_always_report_blocked():
    for name in ("ravdess", "crema_d"):
        a = registry._FACTORIES[name]()
        assert a.blocked_reason == "blocked: needs operator rights/scope decision"
        assert _code(a, "research_evaluation", "research_noncommercial") == "BLOCKED_NEEDS_OPERATOR_DECISION"
        for call in (lambda: a.participants("research_evaluation", "research_noncommercial"),
                     lambda: a.items("research_evaluation", "research_noncommercial")):
            with pytest.raises(RightsRefusal):
                call()
        for call in (a.label_map, lambda: a.group_id(None), a._items, a._participants):
            with pytest.raises(RightsRefusal):
                call()


def test_declarations_contain_no_file_reading_code():
    src = (DATA_DIR / "declarations.py").read_text(encoding="utf-8")
    for needle in ("open(", "pathlib", "os.", "glob", "np.load", "read_", "import cv2", "import numpy"):
        assert needle not in src, needle


# --- synthetic adapter and registry -------------------------------------------

def test_synthetic_adapter_runs_without_further_grant():
    a = registry.open_adapter("synthetic", "development_test", "synthetic")
    d = a.authorize("research_evaluation", "research_noncommercial")
    assert d.decision == "allowed" and d.adapter == "synthetic"
    parts = a.participants("development_test", "synthetic")
    items = a.items("development_test", "synthetic")
    assert len(parts) == 6 and len(items) == 6 * 7
    assert {a.group_id(i) for i in items} == set(parts)
    first = items[0].records()
    assert first == items[0].records() and first[0].t == 0.0
    assert set(a.label_map().values()) == {"proxy_positive", "proxy_negative", "excluded"}
    assert hb1.analyze(items[0].records())["heuristic"] == hb1.HEURISTIC


def test_adapter_data_methods_run_the_gate():
    a = SyntheticAdapter()
    with pytest.raises(RightsRefusal) as exc:
        a.items("development_test", "no-such-scope")
    assert exc.value.code == "SCOPE_MISMATCH"


def test_registry_swap_does_not_grant_anything():
    class Commercial(FakeAdapter):
        name = "commercial_pilot_x"
        licence_tag = "consented-commercial"
        source_class = "consented_pilot"
        allowed_uses = ("commercial_training",)
        scope = "commercial_clean"

    registry.register_adapter("commercial_pilot_x", Commercial)
    try:
        with pytest.raises(RightsRefusal) as exc:
            registry.open_adapter("commercial_pilot_x", "commercial_training", "commercial_clean")
        assert exc.value.code == "DATASET_NOT_ALLOWLISTED"  # needs a reviewed allowlist entry
        with pytest.raises(ValueError):
            registry.register_adapter("commercial_pilot_x", Commercial)
        with pytest.raises(ValueError):
            registry.register_adapter("synthetic", Commercial)
    finally:
        registry._REGISTRY.pop("commercial_pilot_x", None)
    with pytest.raises(RightsRefusal) as exc:
        registry.open_adapter("no_such_dataset", "development_test", "synthetic")
    assert exc.value.code == "DATASET_NOT_ALLOWLISTED"
    assert {"crema_d", "ravdess", "synthetic"} <= set(registry.registered_names())


def test_refusal_carries_code_and_reason():
    with pytest.raises(RightsRefusal) as exc:
        registry.open_adapter("casme2", "research_evaluation", "research_noncommercial")
    e = exc.value
    assert e.code and e.reason and str(e).startswith(e.code)
    assert isinstance(e, Exception)


# --- no bypass ---------------------------------------------------------------

def test_no_bypass_parameters_or_environment_switches():
    forbidden = re.compile(r"force|override|bypass|skip|disable|unsafe|allow_all|trust|waive|ignore_", re.I)
    for fn in (authorize, rights.load_allowlist.__wrapped__, rights.DatasetAdapter.authorize,
               rights.DatasetAdapter.items, rights.DatasetAdapter.participants,
               provenance.build_manifest, registry.open_adapter):
        for p in inspect.signature(fn).parameters:
            assert not forbidden.search(p), (fn.__name__, p)
    for path in sorted(DATA_DIR.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        assert "os.environ" not in src and "environ[" not in src and "getenv" not in src, path.name
        assert not re.search(r"sys\.argv|argparse", src), path.name


def test_extra_kwargs_and_env_vars_do_not_open_the_gate(monkeypatch):
    for var in ("SOCIALCUE_ALLOW_ALL", "SOCIALCUE_BYPASS", "RIGHTS_GATE_DISABLE", "FORCE", "CI"):
        monkeypatch.setenv(var, "1")
    with pytest.raises(TypeError):
        authorize(registry._FACTORIES["ravdess"](), "research_evaluation", "research_noncommercial", force=True)
    with pytest.raises(TypeError):
        registry.open_adapter("ravdess", "research_evaluation", "research_noncommercial", bypass=True)
    with pytest.raises(RightsRefusal):
        registry.open_adapter("ravdess", "research_evaluation", "research_noncommercial")
    with pytest.raises(RightsRefusal):
        registry.open_adapter("casme2", "research_evaluation", "research_noncommercial")


# --- provenance manifest -----------------------------------------------------

def test_manifest_is_deterministic_and_self_verifying():
    kw = dict(use="development_test", run_scope="synthetic", counts={"items": 42, "participants": 6},
              config_sha256=hb1.CONFIG_SHA256)
    m1 = provenance.build_manifest(SyntheticAdapter(), **kw)
    m2 = provenance.build_manifest(SyntheticAdapter(), **kw)
    assert m1 == m2 and json.dumps(m1, sort_keys=True) == json.dumps(m2, sort_keys=True)
    assert provenance.verify_manifest(m1)
    assert m1["decision"]["result"] == "allowed" and m1["decision"]["allowlist_sha256"] == PINNED_ALLOWLIST_SHA256
    assert m1["adapter"] == {"name": "synthetic", "licence_tag": "synthetic-manufactured",
                             "source_class": "synthetic", "scope": "synthetic"}
    assert m1["hb1_config_sha256"] == hb1.CONFIG_SHA256 and m1["counts"] == {"items": 42, "participants": 6}
    assert m1["licence_banner"] == "synthetic data, no third-party rights"
    assert len(m1["manifest_sha256"]) == 64
    # key order of counts does not matter
    m3 = provenance.build_manifest(SyntheticAdapter(), **{**kw, "counts": {"participants": 6, "items": 42}})
    assert m3["manifest_sha256"] == m1["manifest_sha256"]
    # any change changes the hash, and tampering is detected
    m4 = provenance.build_manifest(SyntheticAdapter(), **{**kw, "counts": {"items": 43, "participants": 6}})
    assert m4["manifest_sha256"] != m1["manifest_sha256"]
    bad = {**m1, "counts": {"items": 1}}
    assert not provenance.verify_manifest(bad)


def test_manifest_records_a_refusal_and_banner():
    m = provenance.build_manifest(registry._FACTORIES["ravdess"](), "commercial_training", "commercial_clean", {"items": 0})
    assert m["decision"]["result"] == "refused" and m["decision"]["code"] == "USE_NOT_ALLOWED"
    assert provenance.verify_manifest(m)
    m2 = provenance.build_manifest(registry._FACTORIES["ravdess"](), "research_evaluation", "research_noncommercial", {})
    assert m2["licence_banner"] == "research, non-commercial"
    assert m2["decision"]["code"] == "BLOCKED_NEEDS_OPERATOR_DECISION"


def test_manifest_rejects_bad_counts_and_has_no_real_paths():
    for bad in ({"items": -1}, {"items": 1.5}, {"items": True}, "x"):
        with pytest.raises(ValueError):
            provenance.build_manifest(SyntheticAdapter(), "development_test", "synthetic", bad)
    text = json.dumps(provenance.build_manifest(SyntheticAdapter(), "development_test", "synthetic", {"items": 1}))
    assert not re.search(r"[A-Za-z]:\\|/Users/|\.npz|\.mp4|\.onnx", text)


def test_manifest_file_write_only_with_explicit_path_and_no_overwrite(tmp_path):
    m = provenance.build_manifest(SyntheticAdapter(), "development_test", "synthetic", {"items": 1})
    p = tmp_path / "m.json"
    provenance.write_manifest(m, str(p))
    assert json.loads(p.read_text(encoding="utf-8")) == m
    with pytest.raises(FileExistsError):
        provenance.write_manifest(m, str(p))


# --- purity ------------------------------------------------------------------

def test_data_package_import_has_no_side_effects():
    code = (
        "import sys, json, hashlib, math, copy, random, dataclasses, typing, builtins, io, os, socket, abc, functools, inspect\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "before = set(sys.modules)\n"
        "def boom(*a, **k):\n"
        "    raise RuntimeError('side effect at import')\n"
        "builtins.open = boom; io.open = boom; os.open = boom\n"
        "os.listdir = boom; os.scandir = boom; os.makedirs = boom; os.mkdir = boom\n"
        "socket.socket = boom; socket.create_connection = boom; socket.getaddrinfo = boom\n"
        "import socialcue.data, socialcue.data.rights, socialcue.data.provenance, socialcue.data.synthetic_adapter, socialcue.data.declarations, socialcue.data.registry\n"
        "new = set(sys.modules) - before\n"
        "bad = [m for m in new if m.split('.')[0] in ('numpy','cv2','torch','onnxruntime','requests','urllib3','http','subprocess','tempfile','socket')]\n"
        "assert not bad, bad\n"
        "from socialcue.data import rights\n"
        "assert rights.load_allowlist.cache_info().currsize == 0\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "OK"
