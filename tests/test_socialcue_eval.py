"""Proxy-label evaluation harness (EL-006 S3). Synthetic actors only; reports go to pytest tmp_path."""
import copy
import dataclasses
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from socialcue import state as hb1  # noqa: E402
from socialcue import synthetic as syn  # noqa: E402
from socialcue.data import registry  # noqa: E402
from socialcue.data.rights import DatasetAdapter, RightsRefusal  # noqa: E402
from socialcue.data.synthetic_adapter import SyntheticAdapter  # noqa: E402
from socialcue.eval import proxy_eval as ev  # noqa: E402
from socialcue.eval import report as rp  # noqa: E402

EVAL_DIR = ROOT / "socialcue" / "eval"


def O(actor, truth, pred):
    return (actor, truth, pred)


# --- metrics against hand-computed confusion matrices ------------------------

def test_confusion_and_metrics_hand_computed():
    outs = ([O("a", True, "pos")] * 3 + [O("a", True, "neg")] * 1
            + [O("b", False, "pos")] * 2 + [O("b", False, "neg")] * 4)
    c = ev.confusion(outs)
    assert c == {"tp": 3, "fp": 2, "tn": 4, "fn": 1, "abstained": 0, "n": 10}
    m = ev.metrics(c)
    assert m["precision"] == pytest.approx(3 / 5)
    assert m["recall"] == pytest.approx(3 / 4)
    assert m["specificity"] == pytest.approx(4 / 6)
    assert m["f1"] == pytest.approx(6 / 9)
    assert m["balanced_accuracy"] == pytest.approx((0.75 + 4 / 6) / 2)
    assert m["abstention_rate"] == 0.0


def test_abstentions_count_as_failures_in_the_full_cohort():
    outs = [O("a", True, "pos"), O("a", True, "abstain"), O("b", False, "neg"), O("b", False, "abstain")]
    full = ev.confusion(outs, "full_cohort")
    assert full == {"tp": 1, "fn": 1, "tn": 1, "fp": 1, "abstained": 2, "n": 4}  # abstain = wrong on both classes
    ans = ev.confusion(outs, "answered_only")
    assert (ans["tp"], ans["fn"], ans["tn"], ans["fp"], ans["abstained"]) == (1, 0, 1, 0, 2)
    assert ev.metrics(full)["abstention_rate"] == 0.5
    assert ev.metrics(ans)["balanced_accuracy"] == 1.0


def test_undefined_metrics_are_none_not_zero():
    m = ev.metrics(ev.confusion([O("a", True, "neg")]))
    assert m["precision"] is None and m["recall"] == 0.0 and m["specificity"] is None and m["balanced_accuracy"] is None
    assert ev.metrics(ev.confusion([]))["recall"] is None


# --- bootstrap ---------------------------------------------------------------

def _known(n_actors=20, wrong=4):
    outs = []
    for i in range(n_actors):
        a = f"actor-{i:02d}"
        outs.append(O(a, True, "neg" if i < wrong else "pos"))
        outs.append(O(a, False, "pos" if i >= n_actors - wrong else "neg"))
    return outs


def test_bootstrap_is_deterministic_and_seed_sensitive():
    outs = _known()
    a = ev.bootstrap_ci(outs)
    assert a == ev.bootstrap_ci(outs)
    assert a != ev.bootstrap_ci(outs, seed=7)
    assert ev.N_DRAWS == 1000 and ev.SEED == 505
    assert a["balanced_accuracy"]["n_valid"] <= 1000


def test_bootstrap_ci_contains_the_point_estimate_on_a_known_case():
    ci = ev.bootstrap_ci(_known())
    for name in ev.METRICS:
        d = ci[name]
        assert d["lo"] <= d["estimate"] <= d["hi"], (name, d)
    ba = ci["balanced_accuracy"]
    assert ba["estimate"] == pytest.approx(0.8) and ba["lo"] < 0.8 < ba["hi"] and ba["hi"] <= 1.0


def test_bootstrap_degenerate_cases():
    perfect = ev.bootstrap_ci([O(f"a{i}", i % 2 == 0, "pos" if i % 2 == 0 else "neg") for i in range(8)])
    # resamples where one class vanishes are skipped, the rest are perfect
    assert perfect["balanced_accuracy"]["lo"] == perfect["balanced_accuracy"]["hi"] == 1.0
    empty = ev.bootstrap_ci([])
    assert empty["recall"] == {"estimate": None, "lo": None, "hi": None, "n_valid": 0}


def test_bootstrap_resamples_actors_not_clips():
    # one actor holds every error: resampling clips would average it out, resampling actors cannot
    outs = [O("bad", True, "neg")] * 5 + [O("bad", False, "pos")] * 5
    outs += [O(f"g{i}", True, "pos") for i in range(5)] + [O(f"g{i}", False, "neg") for i in range(5)]
    ci = ev.bootstrap_ci(outs)["balanced_accuracy"]
    assert ci["lo"] < 0.75 and ci["hi"] == 1.0


# --- actor-disjoint splits -----------------------------------------------------

def test_split_is_disjoint_complete_and_deterministic():
    actors = [f"a{i:02d}" for i in range(12)]
    s = ev.actor_disjoint_split(actors)
    assert s == ev.actor_disjoint_split(list(reversed(actors)))
    assert set(s) == set(ev.ROLES) and sum(len(v) for v in s.values()) == 12
    assert set().union(*map(set, s.values())) == set(actors)
    assert s != ev.actor_disjoint_split(actors, seed=1)


def test_disjointness_assertion_fails_when_violated():
    with pytest.raises(ValueError, match="appears in both"):
        ev.assert_actor_disjoint({"DEV": ["a", "b"], "VAL": ["b"], "FINAL": ["c"]})
    with pytest.raises(ValueError, match="at least one"):
        ev.assert_actor_disjoint({"DEV": ["a"], "VAL": [], "FINAL": ["c"]})
    with pytest.raises(ValueError):
        ev.actor_disjoint_split(["a", "b"])
    ev.assert_actor_disjoint({"DEV": ["a"], "VAL": ["b"], "FINAL": ["c"]})


# --- end to end on the synthetic adapter -----------------------------------------

@pytest.fixture(scope="module")
def results():
    return ev.evaluate(SyntheticAdapter(n_participants=12), "synthetic")


def test_end_to_end_structure_and_actor_disjointness(results):
    r = results
    assert r["splits"].keys() == {"DEV", "VAL", "FINAL"}
    ev.assert_actor_disjoint(r["splits"])
    assert sum(len(v) for v in r["splits"].values()) == 12
    for ep in ("P1", "P2"):
        for role, actors in r["splits"].items():
            for method in ev.METHODS:
                d = r["endpoints"][ep][role][method]
                assert d["n_clips"] == 2 * len(actors)  # one calm + one nervous-like clip per actor
                assert set(d["ci_full_cohort"]) == set(ev.METRICS)
    assert r["manifest"]["decision"]["result"] == "allowed" and r["config_sha256"] == hb1.CONFIG_SHA256
    assert r["confusers"]["P1"]["n_clips"] == 0 and r["confusers"]["P1"]["excluded_n"] == 12 * 5 and r["confusers"]["P1"]["classes"] == []  # S60: sentinel is not a confuser


def test_comparators_always_no_cue_and_fer7_only(results):
    for role in ev.ROLES:
        none = results["endpoints"]["P1"][role]["always_no_cue"]["full_cohort"]
        assert none["recall"] == 0.0 and none["specificity"] == 1.0 and none["balanced_accuracy"] == 0.5
        assert none["abstention_rate"] == 0.0 and none["confusion"]["tp"] == 0
        assert results["endpoints"]["P1"][role]["hb1"]["full_cohort"]["balanced_accuracy"] == 1.0
        assert results["endpoints"]["P1"][role]["fer7_only"]["full_cohort"]["balanced_accuracy"] == 1.0
    assert results["confusers"]["P1"]["hb1_positive_rate"] is None and results["confusers"]["P2"]["hb1_positive_rate"] is None  # S60: no real confuser class


@dataclasses.dataclass(frozen=True)
class _Item:
    group: str
    label: str
    recs: tuple

    def records(self):
        return list(self.recs)


class _CraftedAdapter(SyntheticAdapter):
    """Passes the gate as 'synthetic'; supplies crafted clips where FER7-only and HB-1 disagree."""

    def _items(self):
        nerv, calm = syn.generate("nervous-like"), syn.generate("calm")
        still = tuple(dataclasses.replace(a, box=b.box, landmarks=b.landmarks, arousal=0.1) for a, b in zip(nerv, calm))
        out = []
        for i in range(6):
            actor = f"synthetic-actor-{i:02d}"
            out.append(_Item(actor, "nervous-like", still))   # high p_neg but still head and low arousal
            out.append(_Item(actor, "calm", tuple(calm)))
        return out

    def _participants(self):
        return [f"synthetic-actor-{i:02d}" for i in range(6)]

    def group_id(self, item):
        return item.group


def test_custom_synthetic_ablation_adapter_is_refused_by_exact_class_pin():
    # Custom record producers no longer enter evaluate, even under synthetic declarations.
    with pytest.raises(RightsRefusal, match="UNREVIEWED_EVALUATOR_ADAPTER"):
        ev.evaluate(_CraftedAdapter(), "synthetic", n_draws=50)


def test_evaluation_is_deterministic():
    a = ev.evaluate(SyntheticAdapter(n_participants=6), "synthetic", n_draws=100)
    b = ev.evaluate(SyntheticAdapter(n_participants=6), "synthetic", n_draws=100)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# --- the harness cannot be pointed at real data ----------------------------------

@pytest.mark.parametrize("name", ["ravdess", "crema_d"])
def test_declarations_are_refused(name):
    adapter = registry._FACTORIES[name]()
    with pytest.raises(RightsRefusal) as exc:
        ev.evaluate(adapter, "research_noncommercial")
    assert exc.value.code == "BLOCKED_NEEDS_OPERATOR_DECISION"
    with pytest.raises(RightsRefusal):
        registry.open_adapter(name, "research_evaluation", "research_noncommercial")


class _Fake(DatasetAdapter):
    name = "synthetic"
    licence_tag = "synthetic-manufactured"
    source_class = "synthetic"
    allowed_uses = ("research_evaluation",)
    scope = "synthetic"

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def _participants(self):
        raise AssertionError("data must not be read after a refusal")

    def _items(self):
        raise AssertionError("data must not be read after a refusal")

    def label_map(self):
        return {}

    def group_id(self, item):
        return "g"


@pytest.mark.parametrize("kw,code", [
    ({"source_class": "movie"}, "SOURCE_CLASS_FORBIDDEN"),
    ({"source_class": "youtube"}, "SOURCE_CLASS_FORBIDDEN"),
    ({"source_class": "unknown"}, "SOURCE_CLASS_FORBIDDEN"),
    ({"name": "affectnet"}, "DATASET_NOT_ALLOWLISTED"),
    ({"name": "casme2", "licence_tag": "by-application", "source_class": "by_application_corpus"}, "NOT_GRANTED"),
])
def test_forbidden_unknown_and_ungranted_adapters_are_refused(kw, code):
    with pytest.raises(RightsRefusal) as exc:
        ev.evaluate(_Fake(**kw), "synthetic")
    assert exc.value.code == code


def test_scope_and_use_mismatches_are_refused():
    with pytest.raises(RightsRefusal) as exc:
        ev.evaluate(SyntheticAdapter(), "no-such-scope")
    assert exc.value.code == "SCOPE_MISMATCH"
    with pytest.raises(RightsRefusal) as exc:
        ev.evaluate(registry._FACTORIES["ravdess"](), "commercial_clean", use="commercial_training")
    assert exc.value.code in {"USE_NOT_ALLOWED", "SCOPE_MISMATCH"}


@pytest.mark.parametrize("not_an_adapter", ["C:/data/ravdess", pathlib.Path("."), {"items": []}, None, 5])
def test_only_dataset_adapters_are_accepted(not_an_adapter):
    with pytest.raises(TypeError):
        ev.evaluate(not_an_adapter, "synthetic")


def test_eval_source_has_no_file_network_or_media_code():
    for path in (EVAL_DIR / "proxy_eval.py", EVAL_DIR / "__init__.py"):
        src = path.read_text(encoding="utf-8")
        for needle in ("open(", "glob", "listdir", "scandir", "imread", "VideoCapture", "np.load", "numpy",
                       "socket", "requests", "urllib", "pathlib", "os.environ"):
            assert needle not in src, (path.name, needle)
    rep = (EVAL_DIR / "report.py").read_text(encoding="utf-8")
    assert rep.count("open(") == 2 and rep.count(', "x"') == 2 and "os.environ" not in rep


# --- report ------------------------------------------------------------------------

def test_report_always_carries_labels_caveat_claims_and_hashes(results, tmp_path):
    report = rp.build_report(results)
    assert report["status"] == "SYNTHETIC / NOT REAL MODEL PERFORMANCE"
    md = rp.render_markdown(report)
    for text in (rp.SYNTHETIC_LABEL, rp.CAVEAT, *rp.ALLOWED_CLAIMS, *rp.DISALLOWED_CLAIMS,
                 report["manifest_sha256"], report["hb1_config_sha256"], hb1.CONFIG_SHA256, "Barrett", "10.1177/1529100619832930",
                 "balanced accuracy", "not for decisions about people"):
        assert text in md, text
    assert md.splitlines()[0] == "# SYNTHETIC / NOT REAL MODEL PERFORMANCE"
    jp, mp = rp.write_report(report, str(tmp_path))
    j = json.loads(pathlib.Path(jp).read_text(encoding="utf-8"))
    assert j["status"] == rp.SYNTHETIC_LABEL and j["caveat"] == rp.CAVEAT and j["report_sha256"] == report["report_sha256"]
    assert pathlib.Path(mp).read_text(encoding="utf-8") == md
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.json", "report.md"]
    with pytest.raises(FileExistsError):
        rp.write_report(report, str(tmp_path))


def test_report_is_deterministic(results):
    assert rp.build_report(results) == rp.build_report(copy.deepcopy(results))


@pytest.mark.parametrize("key", rp.REQUIRED)
def test_report_refuses_to_render_without_a_required_part(results, key, tmp_path):
    report = rp.build_report(results)
    report[key] = ""
    with pytest.raises(rp.ReportIncomplete):
        rp.render_markdown(report)
    with pytest.raises(rp.ReportIncomplete):
        rp.write_report(report, str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_report_refuses_edited_caveat_claims_status_or_numbers(results):
    base = rp.build_report(results)
    for mutate in (lambda r: r.update(caveat=r["caveat"].replace("Acted affect is not", "Acted affect is")),
                   lambda r: r.update(allowed_claims=["Our system reads nervousness."]),
                   lambda r: r.update(disallowed_claims=[]),
                   lambda r: r.update(status="REAL MODEL PERFORMANCE"),
                   lambda r: r["results"]["endpoints"]["P1"]["FINAL"]["hb1"]["full_cohort"].update(balanced_accuracy=0.99),
                   lambda r: r.update(manifest_sha256="0" * 64)):
        report = copy.deepcopy(base)
        mutate(report)
        with pytest.raises(rp.ReportIncomplete):
            rp.render_markdown(report)


def test_non_synthetic_runs_get_the_proxy_label_not_the_synthetic_one(results):
    fake = copy.deepcopy(results)
    fake["adapter"]["source_class"] = "research_corpus"
    fake["banner"] = rp.PROXY_LABEL  # what evaluate() records for a non-synthetic adapter
    report = rp.build_report(fake)
    assert report["status"] == rp.PROXY_LABEL and "NOT CUE-DETECTION ACCURACY" in report["status"]
    assert rp.CAVEAT in rp.render_markdown(report)
    report["status"] = rp.SYNTHETIC_LABEL  # and a real-data report may not pose as synthetic either way
    with pytest.raises(rp.ReportIncomplete):
        rp.render_markdown(report)


# --- purity ---------------------------------------------------------------------------

def test_eval_package_import_has_no_side_effects():
    code = (
        "import sys, json, hashlib, math, copy, random, dataclasses, typing, builtins, io, os, socket, abc, functools, inspect\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import socialcue.data.rights, socialcue.data.provenance, socialcue.data.synthetic_adapter\n"
        "before = set(sys.modules)\n"
        "def boom(*a, **k):\n"
        "    raise RuntimeError('side effect at import')\n"
        "builtins.open = boom; io.open = boom; os.open = boom\n"
        "os.listdir = boom; os.scandir = boom; os.makedirs = boom; os.mkdir = boom\n"
        "socket.socket = boom; socket.create_connection = boom; socket.getaddrinfo = boom\n"
        "import socialcue.eval, socialcue.eval.proxy_eval, socialcue.eval.report\n"
        "new = set(sys.modules) - before\n"
        "bad = [m for m in new if m.split('.')[0] in ('numpy','cv2','torch','onnxruntime','requests','urllib3','http','subprocess','tempfile','socket')]\n"
        "assert not bad, bad\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "OK"
