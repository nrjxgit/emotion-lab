"""S28 manufactured actor IDs/records only; no real grants, media or models."""
from copy import deepcopy
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from socialcue.guard import (ExposureLedger, ExposureOverride, LedgerCheckpoint,
                            SyntheticPayloadSession, GuardedSyntheticAdapter, verify_grant_closed)
from socialcue.data import rights
from socialcue.data.rights import RightsRefusal
from socialcue.eval import proxy_eval

A, B, C = (f"synthetic-actor-{i:04d}" for i in range(3))


def reserve(ledger, ids=(A,), protocol="a", stage="HELD_OUT", **kw):
    return ledger.reserve(ids, stage=stage, protocol_sha256=protocol * 64, **kw)


@pytest.fixture
def ledger(tmp_path):
    result = ExposureLedger(tmp_path, "synthetic:s28")
    result.initialize(explicit_genesis=True)
    return result


def _race(directory, protocol, ready, go, result):
    ledger = ExposureLedger(directory, "synthetic:s28", lock_timeout=10)
    ready.put(True)
    if not go.wait(20):
        raise RuntimeError("synthetic start barrier timed out")
    try:
        reserve(ledger, protocol=protocol)
        result.put("reserved")
    except RightsRefusal as exc:
        result.put(exc.code)


def _crash(directory):
    reserve(ExposureLedger(directory, "synthetic:s28"))
    os._exit(37)


def _locked_crash(directory, ready, exit_now):
    with ExposureLedger(directory, "synthetic:s28")._locked():
        ready.put(True)
        exit_now.wait(20)
        os._exit(39)


def test_different_protocol_processes_serialize_same_actor(ledger):
    ctx = mp.get_context("spawn")
    ready, results, go = ctx.Queue(), ctx.Queue(), ctx.Event()
    children = [ctx.Process(target=_race, args=(str(ledger.directory), protocol, ready, go, results))
                for protocol in ("a", "b")]
    try:
        for child in children:
            child.start()
        for _ in children:
            assert ready.get(timeout=25)
        go.set()
        outcomes = sorted(results.get(timeout=25) for _ in children)
        assert outcomes == ["EXPOSURE_ALREADY_CONSUMED", "reserved"]
        for child in children:
            child.join(25)
            assert child.exitcode == 0
        rows = ledger.history()
        assert len(rows) == 2 and rows[-1]["actors"] == [A]
    finally:
        go.set()
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(10)


def test_crash_after_reservation_before_decode_consumes_actors(ledger):
    child = mp.get_context("spawn").Process(target=_crash, args=(str(ledger.directory),))
    child.start()
    child.join(25)
    try:
        assert child.exitcode == 37
        with pytest.raises(RightsRefusal, match="EXPOSURE_ALREADY_CONSUMED"):
            reserve(ledger, protocol="b")
    finally:
        if child.is_alive():
            child.terminate()
            child.join(10)


def test_os_releases_lock_after_owner_crash_without_deleting_lock_file(ledger):
    ctx = mp.get_context("spawn")
    ready, exit_now = ctx.Queue(), ctx.Event()
    child = ctx.Process(target=_locked_crash, args=(str(ledger.directory), ready, exit_now))
    child.start()
    try:
        assert ready.get(timeout=25)
        with pytest.raises(RightsRefusal, match="LEDGER_LOCK_BUSY"):
            ExposureLedger(ledger.directory, "synthetic:s28", lock_timeout=.1).history()
        exit_now.set()
        child.join(25)
        assert child.exitcode == 39
        assert len(ledger.history()) == 1
        assert ledger.lock_path.is_file()
    finally:
        exit_now.set()
        if child.is_alive():
            child.terminate()
            child.join(10)


def test_missing_history_requires_separate_explicit_genesis(tmp_path):
    ledger = ExposureLedger(tmp_path, "synthetic:s28")
    with pytest.raises(RightsRefusal, match="LEDGER_UNAVAILABLE"):
        reserve(ledger)
    assert not ledger.path.exists()
    with pytest.raises(RightsRefusal, match="LEDGER_GENESIS_REQUIRED"):
        ledger.initialize()
    ledger.initialize(explicit_genesis=True)
    before = ledger.path.read_bytes()
    with pytest.raises(RightsRefusal, match="LEDGER_COMMIT_FAILED"):
        ledger.initialize(explicit_genesis=True)
    assert ledger.path.read_bytes() == before


@pytest.mark.parametrize("damage", [b"", b'{"partial":', b"not-json\n", b"{}\n", b"[]\n"])
def test_partial_or_corrupt_history_never_means_unused(tmp_path, damage):
    ledger = ExposureLedger(tmp_path, "synthetic:s28")
    ledger.path.write_bytes(damage)
    with pytest.raises(RightsRefusal):
        reserve(ledger)
    assert ledger.path.read_bytes() == damage


def test_unavailable_history_is_not_unused(ledger, monkeypatch):
    old = Path.read_bytes
    def denied(path):
        if path == ledger.path:
            raise PermissionError("SYNTHETIC unavailable history")
        return old(path)
    monkeypatch.setattr(Path, "read_bytes", denied)
    with pytest.raises(RightsRefusal, match="LEDGER_UNAVAILABLE"):
        reserve(ledger)


def test_parseable_tampered_chain_is_refused_without_append(ledger):
    reserve(ledger)
    lines = ledger.path.read_text(encoding="ascii").splitlines()
    changed = json.loads(lines[-1])
    changed["protocol_sha256"] = "f" * 64
    lines[-1] = json.dumps(changed)
    data = ("\n".join(lines) + "\n").encode("ascii")
    ledger.path.write_bytes(data)
    with pytest.raises(RightsRefusal, match="LEDGER_HISTORY_INVALID"):
        reserve(ledger, ids=(B,))
    assert ledger.path.read_bytes() == data


@pytest.mark.parametrize("actor", ["1001", "crema_d:1001", "synthetic-actor-../0", 1001])
def test_real_or_noncanonical_actor_identifiers_cannot_enter_guard(ledger, actor):
    with pytest.raises(RightsRefusal, match="ACTOR_INVALID"):
        reserve(ledger, ids=(actor,))
    with pytest.raises(RightsRefusal, match="ACTOR_INVALID"):
        SyntheticPayloadSession((actor,), "HELD_OUT", 20)
    assert len(ledger.history()) == 1


def test_failed_partial_commit_blocks_future_decodes_and_reservations(ledger, monkeypatch):
    write = os.write
    def partial(fd, data):
        write(fd, data[:len(data) // 2])
        raise OSError("SYNTHETIC crash during append")
    monkeypatch.setattr(os, "write", partial)
    with pytest.raises(RightsRefusal, match="LEDGER_COMMIT_FAILED"):
        reserve(ledger)
    monkeypatch.setattr(os, "write", write)
    with pytest.raises(RightsRefusal, match="LEDGER_HISTORY_INVALID"):
        reserve(ledger, protocol="b")


def test_external_high_water_rejects_valid_prefix_rollback(ledger):
    genesis = ledger.path.read_bytes()
    row = reserve(ledger)
    checked = ExposureLedger(ledger.directory, "synthetic:s28", checkpoint=LedgerCheckpoint(row["seq"], row["digest"]))
    ledger.path.write_bytes(genesis)  # manufactured restore, never production repair
    with pytest.raises(RightsRefusal, match="LEDGER_ROLLBACK"):
        checked.history()


def test_namespace_aliases_overlap_and_role_migration_do_not_reset_history(ledger):
    first = reserve(ledger, ids=("SYNTHETIC-actor-0",))
    assert first["actors"] == [A]
    for stage in ("PILOT", "DEV_VAL", "HELD_OUT"):
        with pytest.raises(RightsRefusal, match="EXPOSURE_ALREADY_CONSUMED"):
            reserve(ledger, ids=(A, B), stage=stage, protocol="b")
    with pytest.raises(RightsRefusal, match="LEDGER_HISTORY_INVALID"):
        ExposureLedger(ledger.directory, "synthetic:different").history()


@pytest.mark.parametrize("change", ["extra_actor", "wrong_actor", "missing_exposure", "extra_exposure"])
def test_override_requires_exact_intersection_and_prior_exposures(ledger, change):
    row = reserve(ledger)
    ids, exposures = (A,), (row["reservation_id"],)
    if change == "extra_actor": ids = (A, B)
    if change == "wrong_actor": ids = (B,)
    if change == "missing_exposure": exposures = ()
    if change == "extra_exposure": exposures += ("f" * 32,)
    override = ExposureOverride("SYNTHETIC-override-test", ids, tuple(sorted(exposures)), "manufactured repeat")
    with pytest.raises(RightsRefusal, match="OVERRIDE_INVALID"):
        reserve(ledger, ids=(A, B), protocol="b", override=override)
    assert len(ledger.history()) == 2


def test_explicit_override_appends_and_is_not_replayable(ledger):
    first = reserve(ledger)
    override = ExposureOverride("SYNTHETIC-override-repeat", (A,), (first["reservation_id"],), "synthetic exact repeat")
    second = reserve(ledger, ids=(A, B), protocol="b", override=override)
    assert second["override"]["actor_ids"] == [A]
    assert len(ledger.history()) == 3
    with pytest.raises(RightsRefusal, match="OVERRIDE_INVALID"):
        reserve(ledger, protocol="c", override=override)


@pytest.fixture
def capability():
    now = [10.]
    adapter = GuardedSyntheticAdapter(n_participants=2)
    item = adapter.items("research_evaluation", "synthetic")[0]
    session = SyntheticPayloadSession((A,), "HELD_OUT", 20., clock=lambda: now[0])
    return item, session, now


def test_capability_required_and_validated_on_every_decode(capability):
    item, session, now = capability
    with pytest.raises(RightsRefusal, match="CAPABILITY_REQUIRED"):
        item.records()
    cap = session.issue(item, stage="HELD_OUT")
    assert item.records(session, cap, stage="HELD_OUT")
    now[0] = 20.
    with pytest.raises(RightsRefusal, match="SESSION_EXPIRED"):
        item.records(session, cap, stage="HELD_OUT")


def test_capabilities_revoke_expire_and_reject_wrong_stage_actor_object(capability):
    item, session, now = capability
    cap = session.issue(item, stage="HELD_OUT")
    with pytest.raises(RightsRefusal, match="STAGE_MISMATCH"):
        item.records(session, cap, stage="DEV_VAL")
    other = GuardedSyntheticAdapter(2).items("research_evaluation", "synthetic")
    with pytest.raises(RightsRefusal, match="ACTOR_MISMATCH"):
        session.issue(next(i for i in other if i.group == "synthetic-actor-01"), stage="HELD_OUT")
    with pytest.raises(RightsRefusal, match="CAPABILITY_INVALID"):
        other[0].records(session, cap, stage="HELD_OUT")
    with pytest.raises(TypeError):
        pickle.dumps(cap)
    with pytest.raises(AttributeError):
        cap.item = other[0]  # token has no writable binding attributes
    forged = object.__new__(type(cap))
    with pytest.raises(RightsRefusal, match="CAPABILITY_INVALID"):
        item.records(session, forged, stage="HELD_OUT")
    session.revoke()
    with pytest.raises(RightsRefusal, match="SESSION_REVOKED"):
        item.records(session, cap, stage="HELD_OUT")


def test_missing_cross_session_and_expired_capabilities_never_generate(capability, monkeypatch):
    from socialcue import synthetic
    item, session, now = capability
    cap = session.issue(item, stage="HELD_OUT")
    called = []
    monkeypatch.setattr(synthetic, "generate", lambda *a, **kw: called.append(True))
    with pytest.raises(RightsRefusal, match="CAPABILITY_INVALID"):
        item.records(session, None, stage="HELD_OUT")
    other = SyntheticPayloadSession((A,), "HELD_OUT", 20, clock=lambda: now[0])
    with pytest.raises(RightsRefusal, match="CAPABILITY_INVALID"):
        item.records(other, cap, stage="HELD_OUT")
    now[0] = 20
    with pytest.raises(RightsRefusal, match="SESSION_EXPIRED"):
        item.records(session, cap, stage="HELD_OUT")
    assert called == []


def test_expiry_or_revocation_during_generation_discards_result(capability, monkeypatch):
    from socialcue import synthetic
    item, session, now = capability
    cap = session.issue(item, stage="HELD_OUT")
    def revoke(*a, **kw):
        session.revoke()
        return ["manufactured result"]
    monkeypatch.setattr(synthetic, "generate", revoke)
    with pytest.raises(RightsRefusal, match="SESSION_REVOKED"):
        item.records(session, cap, stage="HELD_OUT")


def test_legacy_evaluator_refuses_granted_real_declaration_before_any_data(monkeypatch):
    from socialcue.data.declarations import CremaDDeclaration
    class NeverOpen(CremaDDeclaration):
        blocked_reason = None  # Manufactured active declaration as well as granted snapshot.
        def _participants(self): raise AssertionError("real metadata must not open")
        def _items(self): raise AssertionError("real items must not open")
        def label_map(self): raise AssertionError("real labels must not open")
    doc = json.loads((ROOT / "socialcue/data/rights_allowlist.json").read_text(encoding="utf-8"))
    next(e for e in doc["datasets"] if e["name"] == "crema_d")["status"] = "granted"
    monkeypatch.setattr(rights, "load_allowlist", lambda: (rights.parse_allowlist(doc), "SYNTHETIC grant snapshot"))
    adapter = NeverOpen()
    assert rights.authorize(adapter, "research_evaluation", "research_noncommercial").decision == "allowed"
    with pytest.raises(RightsRefusal, match="NON_SYNTHETIC_EVALUATION_DISABLED"):
        proxy_eval.evaluate(adapter, "research_noncommercial")


@pytest.mark.parametrize("name", sorted({"ravdess", "crema_d", "casme2", "samm", "smic", "pilot_consented"}))
def test_grant_close_pure_check_refuses_each_real_grant(name):
    doc = json.loads((ROOT / "socialcue/data/rights_allowlist.json").read_text(encoding="utf-8"))
    for entry in doc["datasets"]:
        if entry["status"] == "blocked_pending_operator":
            entry["allowed_uses"], entry["scopes"] = [], []
    entries = rights.parse_allowlist(doc)
    assert verify_grant_closed(entries)
    changed = {k: dict(v) for k, v in entries.items()}
    changed[name]["status"] = "granted"
    with pytest.raises(RightsRefusal, match="REAL_GRANT_STILL_OPEN"):
        verify_grant_closed(changed)
    changed.pop(name)
    with pytest.raises(RightsRefusal, match="GRANT_CLOSE_INVALID"):
        verify_grant_closed(changed)


def test_grant_close_checks_extra_entries_without_mutating_snapshot():
    doc = json.loads((ROOT / "socialcue/data/rights_allowlist.json").read_text(encoding="utf-8"))
    for entry in doc["datasets"]:
        if entry["status"] == "blocked_pending_operator":
            entry["allowed_uses"], entry["scopes"] = [], []
    entries = {e["name"]: e for e in doc["datasets"]}
    entries["future_real"] = {"name": "future_real", "source_class": "research_corpus", "status": "granted"}
    original = deepcopy(entries)
    with pytest.raises(RightsRefusal, match="REAL_GRANT_STILL_OPEN"):
        verify_grant_closed(entries)
    assert entries == original


def test_imports_and_constructors_have_no_application_io(tmp_path):
    script = r'''
import builtins, io, os, pathlib, sys, json, hashlib, uuid, dataclasses, time, re
sys.path.insert(0, sys.argv[1])
from unittest.mock import patch
def deny(*a, **kw): raise AssertionError("constructor/import application IO forbidden")
with patch.object(builtins,'open',deny), patch.object(io,'open',deny), patch.object(os,'open',deny):
    from socialcue.guard import ExposureLedger, SyntheticPayloadSession, GuardedSyntheticAdapter
    ExposureLedger(sys.argv[2], 'synthetic:s28')
    SyntheticPayloadSession(('synthetic-actor-0',), 'HELD_OUT', 20, clock=lambda: 10)
    GuardedSyntheticAdapter()
print('application IO zero; import machinery source reads not traced')
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(ROOT), str(tmp_path)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "application IO zero" in result.stdout


# S29: fail-first probes on the S28 implementation.
def test_s29_one_authorization_decision(monkeypatch):
    from socialcue.data import provenance
    from socialcue.data.synthetic_adapter import SyntheticAdapter
    original = rights.authorize
    calls = []
    def observed(*a, **kw):
        calls.append(True)
        return original(*a, **kw)
    monkeypatch.setattr(rights, "authorize", observed)
    monkeypatch.setattr(proxy_eval, "authorize", observed)
    monkeypatch.setattr(provenance, "authorize", observed)
    proxy_eval.evaluate(SyntheticAdapter(3), "synthetic", n_draws=2)
    assert len(calls) == 1


def test_s29_stateful_declaration_cannot_reauthorize_into_real_items(monkeypatch):
    from socialcue.data.synthetic_adapter import SyntheticAdapter
    doc = json.loads((ROOT / "socialcue/data/rights_allowlist.json").read_text(encoding="utf-8"))
    next(e for e in doc["datasets"] if e["name"] == "crema_d")["status"] = "granted"
    monkeypatch.setattr(rights, "load_allowlist", lambda: (rights.parse_allowlist(doc), "SYNTHETIC"))
    opened = []
    class Stateful(SyntheticAdapter):
        def _items(self):
            opened.append(True)
            return []
    original = rights.authorize
    def flip(adapter, *args):
        decision = original(adapter, *args)
        adapter.name, adapter.licence_tag = "crema_d", "ODbL 1.0 / DbCL 1.0"
        adapter.source_class, adapter.scope = "research_corpus", "research_noncommercial"
        adapter.allowed_uses = ("research_evaluation",)
        return decision
    monkeypatch.setattr(proxy_eval, "authorize", flip)
    with pytest.raises(RightsRefusal, match="UNREVIEWED_EVALUATOR_ADAPTER"):
        proxy_eval.evaluate(Stateful(3), "research_noncommercial", n_draws=2)
    assert opened == []


def test_s29_unreviewed_synthetic_subclass_records_never_run():
    from socialcue.data.synthetic_adapter import SyntheticAdapter
    from socialcue import synthetic
    opened = []
    class Item:
        scenario = "calm"
        def __init__(self, group): self.group = group
        def records(self):
            opened.append(True)
            return synthetic.generate("calm")
    class Unreviewed(SyntheticAdapter):
        def _items(self): return [Item(a) for a in self._participants()]
    with pytest.raises(RightsRefusal, match="UNREVIEWED_EVALUATOR_ADAPTER"):
        proxy_eval.evaluate(Unreviewed(3), "synthetic", n_draws=2)
    assert opened == []


@pytest.mark.parametrize("operation", ["copy", "deepcopy", "pickle"])
def test_s29_session_cannot_be_copied_or_pickled(operation):
    import copy, time
    session = SyntheticPayloadSession((A,), "HELD_OUT", time.monotonic() + 10)
    with pytest.raises(TypeError):
        {"copy": copy.copy, "deepcopy": copy.deepcopy, "pickle": pickle.dumps}[operation](session)


def test_s29_session_self_trust_is_explicit_and_not_a_ledger_gate(capability):
    item, session, _ = capability
    assert getattr(session, "authority_model", None) == "SELF_TRUSTED_SYNTHETIC_ONLY"
    cap = session.issue(item, stage="HELD_OUT")  # intentionally no ledger/supervisor exists
    assert item.records(session, cap, stage="HELD_OUT")


def test_s29_oversized_request_refused_before_write_without_bricking(ledger):
    before = ledger.path.read_bytes()
    ids = [f"synthetic-actor-{i}" for i in range(1025)]
    with pytest.raises(RightsRefusal, match="ACTOR_LIMIT"):
        reserve(ledger, ids=ids)
    assert ledger.path.read_bytes() == before
    assert reserve(ledger)["actors"] == [A]


def test_s29_mandatory_checkpoint_option_and_pure_check(ledger):
    from socialcue.guard import ledger as mod
    helper = getattr(mod, "require_external_checkpoint", None)
    assert callable(helper)
    with pytest.raises(RightsRefusal, match="LEDGER_CHECKPOINT_REQUIRED"):
        helper(None)
    checked = ExposureLedger(ledger.directory, "synthetic:s28", require_checkpoint=True)
    with pytest.raises(RightsRefusal, match="LEDGER_CHECKPOINT_REQUIRED"):
        reserve(checked)
    row = ledger.history()[-1]
    cp = LedgerCheckpoint(row["seq"], row["digest"])
    assert helper(cp) == cp
    assert reserve(ExposureLedger(ledger.directory, "synthetic:s28", checkpoint=cp, require_checkpoint=True))


@pytest.mark.parametrize("variant", ["crlf", "formfeed", "spaces"])
def test_s29_noncanonical_ledger_bytes_refused(ledger, variant):
    reserve(ledger)
    data = ledger.path.read_bytes()
    if variant == "crlf": data = data.replace(b"\n", b"\r\n")
    elif variant == "formfeed": data = data.replace(b"\n", b"\x0c", 1)
    else: data = data.replace(b"\n", b" \n")
    ledger.path.write_bytes(data)
    with pytest.raises(RightsRefusal, match="LEDGER_HISTORY_INVALID"):
        ledger.history()


@pytest.mark.parametrize("expiry", [1e300, 1700000000.])
def test_s29_expiry_rejects_epoch_units_and_excessive_ttl(expiry):
    with pytest.raises(RightsRefusal, match="SESSION_INVALID"):
        SyntheticPayloadSession((A,), "HELD_OUT", expiry, clock=lambda: 10.)


@pytest.mark.parametrize("status", [[], {}])
def test_s29_unhashable_close_status_is_typed_refusal(status):
    doc = json.loads((ROOT / "socialcue/data/rights_allowlist.json").read_text(encoding="utf-8"))
    entries = {e["name"]: e for e in doc["datasets"]}
    entries["synthetic"]["status"] = status
    with pytest.raises(RightsRefusal, match="GRANT_CLOSE_INVALID"):
        verify_grant_closed(entries)


def test_s29_blocked_close_requires_empty_uses_and_scopes():
    doc = json.loads((ROOT / "socialcue/data/rights_allowlist.json").read_text(encoding="utf-8"))
    with pytest.raises(RightsRefusal, match="GRANT_CLOSE_UNRESTRICTED"):
        verify_grant_closed({e["name"]: e for e in doc["datasets"]})


def test_s29_mutated_item_cannot_use_issued_capability(capability):
    item, session, _ = capability
    cap = session.issue(item, stage="HELD_OUT")
    seed = item.seed
    with pytest.raises(AttributeError):
        object.__setattr__(item, "seed", seed + 1)
    assert item.seed == seed
    assert item.records(session, cap, stage="HELD_OUT")


def test_s29_capabilities_are_consumed_once(capability):
    item, session, _ = capability
    cap = session.issue(item, stage="HELD_OUT")
    assert item.records(session, cap, stage="HELD_OUT")
    with pytest.raises(RightsRefusal, match="CAPABILITY_INVALID"):
        item.records(session, cap, stage="HELD_OUT")


@pytest.mark.parametrize("directory", [None, 123, []])
def test_s29_invalid_directory_is_typed_refusal(directory):
    with pytest.raises(RightsRefusal, match="LEDGER_NAMESPACE_INVALID"):
        ExposureLedger(directory, "synthetic:s28")


@pytest.mark.parametrize("value", [10**400, -(10**400)])
@pytest.mark.parametrize("where", ["expiry", "constructor_clock", "issue_clock", "decode_clock"])
def test_s30_huge_integer_time_is_typed_refusal(value, where, capability):
    item, session, now = capability
    with pytest.raises(RightsRefusal, match="SESSION_INVALID"):
        if where == "expiry":
            SyntheticPayloadSession((A,), "HELD_OUT", value, clock=lambda: 10.)
        elif where == "constructor_clock":
            SyntheticPayloadSession((A,), "HELD_OUT", 20., clock=lambda: value)
        elif where == "issue_clock":
            now[0] = value
            session.issue(item, stage="HELD_OUT")
        else:
            token = session.issue(item, stage="HELD_OUT")
            now[0] = value
            session.decode(item, token, stage="HELD_OUT")
