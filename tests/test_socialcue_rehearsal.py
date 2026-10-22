"""U1 scripted route rehearsal (EL-006 S75): runs scripts/socialcue_rehearsal.py in-process. Synthetic fakes only."""
import importlib.util
import pathlib
import sys
import subprocess
import copy

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "app"))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

_spec = importlib.util.spec_from_file_location("socialcue_rehearsal", ROOT / "scripts" / "socialcue_rehearsal.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


@pytest.fixture(scope="module")
def runs():
    a, b = R.run_rehearsal(), R.run_rehearsal()
    return a, b, R.render(a), R.render(b)


def test_two_runs_are_identical_and_headed_synthetic(runs):
    a, b, ta, tb = runs
    assert ta == tb and a == b
    assert ta.splitlines()[0] == R.HEADER_LINE and "SYNTHETIC / NOT REAL MODEL PERFORMANCE" in ta.splitlines()[0]
    assert ta.rstrip().endswith("SYNTHETIC / NOT REAL MODEL PERFORMANCE")
    from socialcue import body, state
    assert f"hb1_config_sha256: {state.CONFIG_SHA256}" in ta and f"body_config_sha256: {body.BODY_CONFIG_SHA256}" in ta
    assert "git_head: " in ta and ta.count("## compare ") == 6  # mismatches are reported, whatever their number


def test_no_models_file_no_network_no_heavy_import(runs):
    iso = runs[0]["isolation"]
    assert iso["models_opens"] == [] and iso["network_connects"] == [] and iso["heavy_import_attempts"] == []
    assert iso["preimport_stub_modules"] == ["body.pose", "face.detector", "face.expression"]
    assert "native file IO" in iso["audit_limit"]
    assert not [m for m in sys.modules if m.split(".")[0] in R.BLOCKED]


def test_every_frame_goes_through_the_route_with_status_200(runs):
    for sc in runs[0]["scenarios"].values():
        assert sc["frames_total"] == sc["frames_http_200"] > 0
        assert all(f["http"] == 200 and "body" in f and "window" in f for f in sc["frames"])


def test_expected_qualitative_sequences(runs):
    sc = runs[0]["scenarios"]
    t0 = 1000.0 + R.STEP
    calm = sc["calm"]
    assert calm["first_seen_t"] == {} and all(f["states"] == [] for f in calm["frames"])
    hes = sc["hesitation"]["first_seen_t"]
    assert "unsure-like" in hes and "nervous-like" not in hes
    esc = sc["escalation_with_absence_tail"]
    seen = esc["first_seen_t"]
    assert "nervous-like" in seen and "unsure-like" in seen and "needs-help-like" in seen
    assert seen["nervous-like"] < seen["needs-help-like"]  # nervous-like leads; needs-help-like precedes the first unsure-like in this script (real behaviour, see receipt)
    for sess in (hes, seen):  # needs-help-like only after more than the 5 s dwell
        assert sess["needs-help-like"] - t0 > 5.0
    last = esc["last_frame"]  # absence tail: abstains
    assert last["states"] == [] and last["insufficient"] is True and "no_face" in last["reasons"]
    assert all(f["states"] == [] for f in esc["frames"][-8:])


def test_compare_section_reports_instead_of_hiding(runs):
    for sc in runs[0]["scenarios"].values():
        for c in sc["compare"].values():
            assert c["frames_compared"] == sc["frames_http_200"]
            assert len(c["mismatches"]) == min(25, c["mismatch_count"])
        assert sc["compare"]["retained-snapshot reference"]["mismatch_count"] == 0
    assert runs[0]["scenarios"]["calm"]["compare"]["full-history reference"]["mismatch_count"] == 0


def _assert_strengthened(t):
    from socialcue import state as hb1
    expected_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or "unknown"
    assert t["git_head"] == expected_head
    assert R.render(t).splitlines()[1] == f"git_head: {expected_head}"
    nonzero = []
    for name, sc in t["scenarios"].items():
        actual = {}
        expected_mismatches = {"retained-snapshot reference": [], "full-history reference": []}
        for f in sc["frames"]:
            record = hb1.FrameRecord(**f["route_record"])
            actual[record.t] = record
            retained = hb1.PacedRecords([actual[v] for v in f["retained_times"]], f["bin_origin"])
            retained.current = record
            oracle = hb1.analyze(retained, evaluation_time=f["t"])
            deltas = dict(f["body_deltas"])
            states = [[s["name"], round(min(.8, max(0., s["confidence"] + deltas.get(s["name"], 0.))), 4)] for s in oracle["states"]]
            assert states == f["states"]
            assert oracle["insufficient"]["flag"] == f["insufficient"]
            assert oracle["insufficient"]["reasons"] == f["reasons"]
            assert oracle["window"] == f["window"]
            features = hb1.extract_features(retained)
            for state, confidence in f["states"]:
                nonzero.append(confidence)
                if name == "escalation_with_absence_tail" and state == "needs-help-like":
                    assert confidence <= .575 * features["coverage"] * features["quality"] + .00011
            route = {k: f[k] for k in ("states", "insufficient", "reasons", "window", "body")}
            for label, ref in f["references"].items():
                if label == "retained-snapshot reference":
                    assert ref == route
                    mismatch = ref != route
                else:
                    a, b = dict(route["states"]), dict(ref["states"])
                    mismatch = (a.keys() != b.keys() or route["insufficient"] != ref["insufficient"]
                                or any(abs(a[k] - b[k]) > .0005 for k in a.keys() & b.keys()))
                if mismatch: expected_mismatches[label].append(f["frame"])
        for label, indexes in expected_mismatches.items():
            c = sc["compare"][label]
            assert c["frames_compared"] == len(sc["frames"])
            assert c["mismatch_count"] == len(indexes)
            assert len(c["mismatches"]) == min(25, len(indexes))
            assert [d["frame"] for d in c["mismatches"]] == indexes[:25]
        if name != "calm": assert expected_mismatches["full-history reference"]
    assert nonzero and max(nonzero) > .1
    assert max(conf for f in t["scenarios"]["escalation_with_absence_tail"]["frames"]
               for name, conf in f["references"]["full-history reference"]["states"] if name == "needs-help-like") == .8


def test_strengthened_independent_references_confidences_provenance_and_counts(runs):
    _assert_strengthened(runs[0])


@pytest.mark.parametrize("mutation", ["zero_confidences", "wrong_head", "zero_mismatches"])
def test_strengthened_assertions_reject_review_output_mutations(runs, mutation):
    changed = copy.deepcopy(runs[0])
    if mutation == "wrong_head": changed["git_head"] = "wrong-head"
    elif mutation == "zero_confidences":
        for sc in changed["scenarios"].values():
            for f in sc["frames"]:
                f["states"] = [[name, 0.] for name, _ in f["states"]]
    else:
        for sc in changed["scenarios"].values():
            for c in sc["compare"].values(): c.update(mismatch_count=0, mismatches=[])
    with pytest.raises(AssertionError): _assert_strengthened(changed)


def test_head_falls_back_to_unknown_without_git_checkout(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(R.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="", returncode=128))
    assert R._git_head() == "unknown"


def test_inert_module_boundaries_precede_app_import(monkeypatch):
    import builtins
    original = builtins.__import__
    observed = []
    def checking(name, *args, **kwargs):
        if name == "main":
            observed.append(all(getattr(sys.modules.get(n), "__rehearsal_inert__", False)
                                for n in ("face.detector", "face.expression", "body.pose")))
        return original(name, *args, **kwargs)
    # Reuse the two-fixture replay for inference; an import-time boundary check
    # deliberately stops before requests, with cleanup exercised by finally.
    class ImportBoundaryChecked(Exception): pass
    def stop_at_main(name, *args, **kwargs):
        if name == "main":
            checking(name, *args, **kwargs)
            raise ImportBoundaryChecked()
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", stop_at_main)
    with pytest.raises(ImportBoundaryChecked): R.run_rehearsal()
    assert observed == [True]
