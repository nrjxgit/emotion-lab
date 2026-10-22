"""Manufactured U3 plumbing checks; no real annotation or capture."""
import copy
import ast
import importlib.util
import json
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
SPEC = importlib.util.spec_from_file_location("pilot_dryrun", ROOT / "scripts/socialcue_pilot_dryrun.py")
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


@pytest.fixture(scope="module")
def cases():
    return D.run_dryrun()


def test_determinism_and_header(cases):
    text = D.render(cases)
    assert text == D.render(D.run_dryrun())
    assert text.startswith("SYNTHETIC / NOT REAL MODEL PERFORMANCE / no real consent\n")
    assert "raw_kappa" not in text and "raw_alpha" not in text
    assert "no real consent" in text and "not findings" in text


def test_below_floor_is_null_with_reason(cases):
    from socialcue.pilot import agreement as G
    text = D.render(cases)
    assert G.WITHHELD in text
    assert '"kappa": null' in text and '"alpha": null' in text
    r = cases["below_floor"]["report"]["results"]
    for label in r["agreement"].values():
        assert label["pair"]["positives_total"] < G.MIN_POSITIVES_FOR_INTERPRETATION
        assert label["pair"]["kappa"] is None
        assert label["alpha"]["alpha"] is None
        assert label["pair"]["kappa_reason"] == "fewer_than_20_positive_windows"
        assert label["alpha"]["reason"] == "fewer_than_20_positive_windows"
        assert label["kappa_ci_by_participant"]["lo"] is None
    for views in r["scoring"].values():
        for roles in views.values():
            assert not roles["ALL"]["evaluable"]
            assert roles["ALL"]["not_evaluable_reason"] == "not_evaluable_at_this_n"


def test_schema_rejection_and_grid_convention(cases):
    assert cases["pii_shaped_field"]["reasons"] == ["pii_or_free_text_field:email"]
    assert not cases["pii_shaped_field"]["accepted"]
    assert not cases["overlapping_blocks"]["accepted"]
    assert "overlapping_or_duplicate_windows" in cases["overlapping_blocks"]["reasons"][0]
    assert cases["non_grid_start"]["accepted"]
    from socialcue.pilot import annotations as A
    for records in D.fixtures().values():
        assert all(A.is_synthetic_participant(r["participant"]) for r in records)
        assert all(len(r["annotator"]) == 10 for r in records)


def test_third_rater_and_missing_support(cases):
    normal = cases["normal_multi_participant"]["report"]["results"]
    third = cases["third_rater_partial"]["report"]["results"]
    for label in normal["agreement"]:
        assert normal["agreement"][label]["pair"] == third["agreement"][label]["pair"]
        assert normal["agreement"][label]["kappa_ci_by_participant"] == third["agreement"][label]["kappa_ci_by_participant"]
    target = "raw:pause"
    assert normal["scoring"][target]["consensus"]["ALL"]["n_windows"] > third["scoring"][target]["consensus"]["ALL"]["n_windows"]
    assert normal["scoring"][target]["ANN-000001"]["ALL"]["evaluable"]
    missing = cases["unscorable_missing_block"]["report"]["results"]
    assert missing["counts"]["unscorable_blocks"] == 1
    assert missing["scoring"][target]["consensus"]["ALL"]["n_windows"] == 0


GAP_IDS = {
    "tiers", "log_labels", "log_comparator", "pooled_primary", "recorder", "final_hash",
    "block_floor", "ambiguous", "report_claims", "practice", "tier_sufficiency",
    "block_counts", "paired_fer7", "mixed_rows", "adjudication", "outcome_validator",
}


def assert_gap_contract(module):
    assert len(module.GAPS) == 16, "gap count must be exactly 16"
    assert {g["id"] for g in module.GAPS} == GAP_IDS, "gap ids changed"
    assert {p.name for p in (ROOT / "socialcue/pilot").glob("*.py")} == {
        "__init__.py", "annotations.py", "agreement.py", "window_scoring.py",
    }, "pilot module inventory changed: review recorder/practice absence claims"
    for gap in module.GAPS:
        assert gap["status"] == "unimplemented", f"{gap['id']}: status changed"
        assert not re.search(r"\bimplemented\b", gap["title"] + " " + gap["detail"], re.I), f"{gap['id']}: implementation claim"
        assert gap["detail"] and gap["expected_substring"]
        assert re.fullmatch(re.escape(D.PROTOCOL) + r":\d+", gap["protocol_ref"])
        path, number = gap["code_ref"].rsplit(":", 1)
        assert path.startswith("socialcue/pilot/") and ".." not in path
        lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
        assert gap["expected_substring"] in lines[int(number) - 1], f"{gap['id']}: code anchor drift"
        for absent in gap.get("absent_constructs", []):
            # Inspect executable symbols/string literals, excluding comments/docstrings.
            tree = ast.parse((ROOT / absent["file"]).read_text(encoding="utf-8"))
            docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                          if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                          and n.body and isinstance(n.body[0], ast.Expr)
                          and isinstance(n.body[0].value, ast.Constant)
                          and isinstance(n.body[0].value.value, str)}
            symbols = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Name)):
                    symbols.append(node.id if isinstance(node, ast.Name) else node.name)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                    symbols.append(node.value)
            assert not any(re.search(absent["pattern"], s, re.I) for s in symbols), f"{gap['id']}: absent construct now present"


def test_verified_gaps_and_source_locations(cases):
    assert_gap_contract(D)
    text = D.render(cases)
    for gap in D.GAPS:
        assert gap["title"] in text and gap["code_ref"] in text
        assert gap["protocol_ref"] in text and gap["detail"] in text
    assert "all remain unimplemented" in text
    assert "distinct annotation blocks" in text


def test_floor_and_agreement_prose_uses_imported_constants(monkeypatch):
    from socialcue.pilot import window_scoring as W, agreement as G
    monkeypatch.setattr(W, "MIN_POS_PARTICIPANTS", 123)
    monkeypatch.setattr(W, "MIN_POS_WINDOWS", 456)
    monkeypatch.setattr(G, "MIN_POSITIVES_FOR_INTERPRETATION", 789)
    text = D.render({})
    assert ">=123 participants WITH positives and >=456 positive aligned HB windows" in text
    assert "Agreement: <789 positives summed across raters" in text


@pytest.mark.parametrize("mutation", ["M1", "M3", "M4"])
def test_gap_mutations_fail_on_temporary_source_copy(tmp_path, mutation):
    assert_gap_contract(D)  # establish a passing control before each mutation
    source = (ROOT / "scripts/socialcue_pilot_dryrun.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "GAPS" for t in n.targets))
    if mutation == "M3":
        assignment.value.elts.pop(7)  # reviewer deleted Ambiguous flag rename
    else:
        row = assignment.value.elts[0]
        field = "detail" if mutation == "M1" else "code_ref"
        index = next(i for i, key in enumerate(row.keys) if key.value == field)
        row.values[index] = ast.Constant("IMPLEMENTED: tiers done" if mutation == "M1" else "socialcue/pilot/annotations.py:3")
    target = tmp_path / f"{mutation}_pilot_dryrun.py"
    target.write_text(ast.unparse(ast.fix_missing_locations(tree)), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("mutated_dryrun", target)
    mutated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mutated)
    reason = {"M1": "implementation claim", "M3": "gap count must be exactly 16", "M4": "code anchor drift"}[mutation]
    with pytest.raises(AssertionError, match=reason) as rejection:
        assert_gap_contract(mutated)
    print(f"{mutation}: temporary source copy rejected: {rejection.value}")


def test_ascii_console_matches_utf8_file(cases, tmp_path, monkeypatch):
    import io
    text = D.render(cases)
    assert text.isascii()
    destination = tmp_path / "utf8.txt"
    monkeypatch.setattr(D, "run_dryrun", lambda: cases)
    D.main([str(destination)])
    raw = io.BytesIO()
    console = io.TextIOWrapper(raw, encoding="cp1252", errors="strict", newline="\n")
    monkeypatch.setattr(sys, "stdout", console)
    D.main([])
    console.flush()
    assert raw.getvalue().decode("cp1252") == destination.read_text(encoding="utf-8") == text


def test_import_has_no_io_or_global_mutations(monkeypatch):
    import builtins
    before_path, before_modules = list(sys.path), set(sys.modules)
    def denied(*args, **kwargs):
        raise AssertionError("import performed IO")
    # Compile before denying opens; execute only the module's import-time body.
    code = compile((ROOT / "scripts/socialcue_pilot_dryrun.py").read_text(encoding="utf-8"), "dryrun", "exec")
    monkeypatch.setattr(builtins, "open", denied)
    monkeypatch.setattr(pathlib.Path, "open", denied)
    monkeypatch.setattr(subprocess, "Popen", denied)
    exec(code, {"__name__": "dryrun_import_check"})
    assert sys.path == before_path
    assert set(sys.modules) == before_modules


def test_only_explicit_output_is_written(tmp_path, monkeypatch, cases):
    import builtins
    import os
    target = tmp_path / "reconciliation.txt"
    opened, original = [], pathlib.Path.open
    def checked(path, mode="r", *args, **kwargs):
        if any(c in mode for c in "wax+"):
            assert path == target
            opened.append(path)
        return original(path, mode, *args, **kwargs)
    def denied(*args, **kwargs):
        raise AssertionError("unexpected alternate file write API")
    monkeypatch.setattr(pathlib.Path, "open", checked)
    monkeypatch.setattr(builtins, "open", denied)
    monkeypatch.setattr(os, "open", denied)
    D.main([str(target)])
    assert opened == [target]
    assert target.read_text(encoding="utf-8") == D.render(cases)
    assert list(tmp_path.iterdir()) == [target]
    with pytest.raises(FileExistsError):
        D.main([str(target)])
