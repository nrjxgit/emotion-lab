"""S50: manufactured pipeline swap / commercial-use declarations, no real fitting."""
import builtins
import copy
import json
from pathlib import Path

import pytest

from socialcue import state as hb1, synthetic
from socialcue.data import registry, rights
from socialcue.data.declarations import RavdessDeclaration, CremaDDeclaration
from socialcue.data.model_provenance import build_model_provenance
from socialcue.data.provenance import verify_manifest
from socialcue.data.synthetic_adapter import SyntheticAdapter
from socialcue.data.synthetic_commercial_adapter import SyntheticCommercialCleanAdapter
from socialcue.eval import proxy_eval as E, report as R
from socialcue.pilot import window_scoring as W
from socialcue.guard.close import verify_grant_closed

CASES = [("synthetic", "synthetic", "synthetic-manufactured", SyntheticAdapter),
         ("synthetic_commercial_clean", "commercial_clean", "synthetic-commercial-clean", SyntheticCommercialCleanAdapter)]


def manufactured_pilot():
    streams, annotations = {}, []
    for i, scenario in enumerate(("calm", "unsure-like")):
        participant = f"PT-5eed{i:04x}"
        streams[participant] = synthetic.generate(scenario, seed=700 + i, duration=3.0)
        for annotator in ("ANN-0000a1", "ANN-0000b2"):
            annotations.append({"schema_version": 1, "annotator": annotator, "participant": participant,
                "window_start": 0.0, "window_end": 3.0,
                "labels": {label: "absent" for label in ("look_away", "pause", "asks_help", "fidget")},
                "unscorable": False, "unsure_rate": False, "confound_flag": False})
    return streams, annotations


@pytest.mark.parametrize("name,scope,tag,cls", CASES)
def test_registry_swap_uses_unchanged_evaluator_and_pilot_reports(name, scope, tag, cls):
    adapter = registry.open_adapter(name, "research_evaluation", scope,
        n_participants=6, scenarios=("calm", "nervous-like"))
    assert type(adapter) is cls
    evaluation = E.evaluate(adapter, scope, n_draws=5)
    report = R.build_report(evaluation)
    assert R.render_markdown(report)
    streams, annotations = manufactured_pilot()
    pilot = W.score_pilot(adapter, scope, streams, annotations, n_draws=5)
    pilot_report = W.build_pilot_report(pilot)
    assert W.render_pilot_markdown(pilot_report)
    for output, result in [(report, evaluation), (pilot_report, pilot)]:
        assert output["status"] == R.SYNTHETIC_LABEL
        manifest = result["manifest"]
        assert verify_manifest(manifest)
        assert manifest["adapter"]["name"] == name
        assert manifest["adapter"]["licence_tag"] == tag
        assert manifest["adapter"]["source_class"] == "synthetic"
        assert manifest["adapter"]["scope"] == scope
        assert manifest["run_scope"] == scope
        assert manifest["decision"]["allowlist_sha256"] == rights.ALLOWLIST_SHA256
        assert output["manifest_sha256"] == manifest["manifest_sha256"]
        assert output["hb1_config_sha256"] == hb1.config_sha256(hb1.HB1_CONFIG)
        assert json.loads(json.dumps(output))["status"] == R.SYNTHETIC_LABEL


@pytest.mark.parametrize("name,scope,tag,cls", CASES)
def test_commercial_training_gate_for_manufactured_adapters(name, scope, tag, cls):
    adapter = registry.open_adapter(name, "commercial_training", "commercial_clean")
    assert rights.authorize(adapter, "commercial_training", "commercial_clean").licence_tag == tag
    assert adapter.items("commercial_training", "commercial_clean")
    assert adapter.participants("commercial_training", "commercial_clean")


def test_swap_keeps_distinct_manifest_and_manufactured_inputs():
    a = registry.open_adapter("synthetic", "research_evaluation", "synthetic")
    b = registry.open_adapter("synthetic_commercial_clean", "research_evaluation", "commercial_clean")
    assert a.items("research_evaluation", "synthetic")[0].seed != b.items("research_evaluation", "commercial_clean")[0].seed
    results = [E.evaluate(adapter, scope, n_draws=2) for adapter, scope in [(a, "synthetic"), (b, "commercial_clean")]]
    assert results[0]["manifest"]["manifest_sha256"] != results[1]["manifest"]["manifest_sha256"]


@pytest.mark.parametrize("cls", [RavdessDeclaration, CremaDDeclaration])
def test_real_declarations_refuse_commercial_training_even_with_clean_run_scope(cls):
    with pytest.raises(rights.RightsRefusal, match="USE_NOT_ALLOWED"):
        rights.authorize(cls(), "commercial_training", "commercial_clean")


def test_nc_adapter_cannot_launder_commercial_scope():
    class NC(SyntheticAdapter):
        licence_tag = "CC BY-NC-SA 4.0"
        scope = "commercial_clean"
    with pytest.raises(rights.RightsRefusal, match="ADAPTER_METADATA_MISMATCH"):
        rights.authorize(NC(), "commercial_training", "commercial_clean")
    class ResearchNC(RavdessDeclaration):
        blocked_reason = None
    with pytest.raises(rights.RightsRefusal, match="USE_NOT_ALLOWED"):
        rights.authorize(ResearchNC(), "commercial_training", "commercial_clean")


@pytest.mark.parametrize("cls,scope", [(SyntheticAdapter, "synthetic"), (SyntheticCommercialCleanAdapter, "commercial_clean")])
def test_exact_class_pin_rejects_subclass_and_instance_payload_overrides(cls, scope):
    opened = []
    class Unreviewed(cls):
        def _items(self): opened.append(True); return []
    with pytest.raises(rights.RightsRefusal, match="UNREVIEWED_EVALUATOR_ADAPTER"):
        E.evaluate(Unreviewed(), scope, n_draws=2)
    adapter = cls()
    adapter._items = lambda: opened.append(True)
    with pytest.raises(rights.RightsRefusal, match="UNREVIEWED_EVALUATOR_ADAPTER"):
        E.evaluate(adapter, scope, n_draws=2)
    assert not opened


@pytest.mark.parametrize("names,expected", [
    (["synthetic"], "yes"), (["synthetic_commercial_clean"], "yes"),
    (["synthetic", "synthetic_commercial_clean"], "yes"),
    (["synthetic", "ravdess"], "no"), (["crema_d"], "no"),
    (["pilot_consented"], "no"), (["casme2"], "no"), (["samm"], "no"), (["smic"], "no"),
    (["unknown_adapter"], "no"), (["synthetic", "unknown_adapter"], "no"),
    ([], "unknown"), (None, "no"), ("synthetic", "no"), ([{}], "no")])
def test_fitting_provenance_fails_closed_without_io(names, expected, monkeypatch):
    before = copy.deepcopy(names)
    def deny(*a, **k): raise AssertionError("pure provenance performed IO/gate/factory access")
    monkeypatch.setattr(builtins, "open", deny)
    monkeypatch.setattr(rights, "authorize", deny)
    monkeypatch.setattr(registry, "open_adapter", deny)
    record = build_model_provenance(names)
    assert names == before
    assert record["commercial_clean"] == expected
    assert record["training_history_verified"] is False
    assert record["allowlist_sha256"] == rights.ALLOWLIST_SHA256
    assert rights.sha256_of({k: v for k, v in record.items() if k != "record_sha256"}) == record["record_sha256"]
    assert build_model_provenance(names) == record


def test_provenance_reviewed_yes_entries_match_shipped_commercial_permissions():
    for name, _, tag, _ in CASES:
        decision = rights.authorize(registry.open_adapter(name, "commercial_training", "commercial_clean"),
                                    "commercial_training", "commercial_clean")
        row = build_model_provenance([name])["adapters"][0]
        assert row["commercial_training_reviewed"] and row["licence_tag"] == decision.licence_tag == tag


def test_close_snapshot_accepts_only_two_reviewed_synthetic_names():
    doc = json.loads((Path(__file__).resolve().parents[1] / "socialcue/data/rights_allowlist.json").read_text(encoding="utf-8"))
    for entry in doc["datasets"]:
        if entry["status"] == "blocked_pending_operator":
            entry["allowed_uses"], entry["scopes"] = [], []
    snapshot = {entry["name"]: entry for entry in doc["datasets"]}
    assert verify_grant_closed(snapshot)
    snapshot["unreviewed_synthetic"] = {"name": "unreviewed_synthetic", "source_class": "synthetic", "status": "granted"}
    with pytest.raises(rights.RightsRefusal, match="REAL_GRANT_STILL_OPEN"):
        verify_grant_closed(snapshot)


@pytest.mark.parametrize("cls", [SyntheticAdapter, SyntheticCommercialCleanAdapter])
def test_s52_rendered_commercial_scope_report_never_claims_synthetic_consent(cls):
    from socialcue.data.provenance import build_manifest
    adapter = cls(n_participants=6, scenarios=("calm", "nervous-like"))
    results = E.evaluate(adapter, "commercial_clean", n_draws=2)
    rendered = R.render_markdown(R.build_report(results))
    assert "commercial-clean scope, SYNTHETIC manufactured data" in rendered
    assert "consented" not in rendered.lower()
    for manifest in (results["manifest"], build_manifest(adapter, "research_evaluation", "commercial_clean", {"items": 1})):
        assert "SYNTHETIC manufactured data" in manifest["licence_banner"]
        assert "consented" not in manifest["licence_banner"].lower()
        assert verify_manifest(manifest)


@pytest.mark.parametrize("names,expected", [
    (["synthetic", "synthetic", "synthetic_commercial_clean"], "yes"),
    (["synthetic", "ravdess", "ravdess"], "no"),
    (["synthetic", "unknown", "unknown"], "no")])
def test_s52_duplicate_fitting_names_collapse_without_laundering(names, expected):
    result = build_model_provenance(names)
    assert result["commercial_clean"] == expected
    assert result == build_model_provenance(sorted(set(names)))


@pytest.mark.parametrize("kind", ["str", "list", "tuple"])
def test_s52_helper_rejects_subclass_elements_and_containers(kind):
    class Text(str): pass
    class Names(list): pass
    class TupleNames(tuple): pass
    names = {"str": [Text("synthetic")], "list": Names(["synthetic"]),
             "tuple": TupleNames(("synthetic",))}[kind]
    result = build_model_provenance(names)
    assert result["commercial_clean"] == "no"
    assert result["reason"] == "invalid_adapter_list"
