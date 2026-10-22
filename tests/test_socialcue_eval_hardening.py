"""Eval harness hardening from the independent S8 review (EL-006 S8b). Synthetic actors only."""
import copy
import itertools
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from socialcue.data.synthetic_adapter import SyntheticAdapter  # noqa: E402
from socialcue.eval import proxy_eval as ev  # noqa: E402
from socialcue.eval import report as rp  # noqa: E402


@pytest.fixture(scope="module")
def results():
    return ev.evaluate(SyntheticAdapter(n_participants=6), "synthetic", n_draws=50)


def _rehash(rep):
    rep["report_sha256"] = rp.sha256_of({k: v for k, v in rep.items() if k != "report_sha256"})
    return rep


# --- H1: the split works down to its stated minimum -------------------------------------------------

@pytest.mark.parametrize("n", range(3, 12))
def test_split_covers_every_actor_once_for_3_to_11_actors(n):
    actors = [f"actor-{i:02d}" for i in range(n)]
    s = ev.actor_disjoint_split(actors)
    assert set(s) == set(ev.ROLES) and all(len(v) >= 1 for v in s.values())
    assert sorted(itertools.chain.from_iterable(s.values())) == sorted(actors)
    assert sum(len(v) for v in s.values()) == n
    ev.assert_actor_disjoint(s, expected=actors)
    assert s == ev.actor_disjoint_split(list(reversed(actors)))  # order-invariant


def test_three_actors_get_one_each_and_fewer_are_refused():
    s = ev.actor_disjoint_split(["a", "b", "c"])
    assert sorted(len(v) for v in s.values()) == [1, 1, 1]
    for n in (0, 1, 2):
        with pytest.raises(ValueError, match="at least 3 actors"):
            ev.actor_disjoint_split([f"a{i}" for i in range(n)])


def test_larger_splits_keep_their_previous_shape():
    assert [len(v) for v in ev.actor_disjoint_split([f"a{i}" for i in range(12)]).values()] == [6, 3, 3]
    assert [len(v) for v in ev.actor_disjoint_split([f"a{i}" for i in range(4)]).values()] == [2, 1, 1]


def test_evaluate_runs_with_three_actors():
    r = ev.evaluate(SyntheticAdapter(n_participants=3), "synthetic", n_draws=20)
    assert [len(v) for v in r["splits"].values()] == [1, 1, 1]
    ev.assert_actor_disjoint(r["splits"])


# --- H2: the FINAL-once limitation is visible, not hidden -----------------------------------------------

def test_results_record_that_final_once_and_protocol_hash_are_not_enforced(results):
    assert results["final_opened_once_enforced"] is False and results["protocol_hash"] is None
    assert any("FINAL-opened-once" in n for n in results["notes"])
    doc = ev.__doc__
    assert "TODO(before any real run" in doc and "final_opened_once_enforced: false" in doc and "protocol_hash: null" in doc
    md = rp.render_markdown(rp.build_report(results))
    assert "final_opened_once_enforced: false" in md and "protocol_hash: None" in md and "Known limitations" in md


def test_report_refuses_results_that_hide_the_limitation(results):
    for mutate in (lambda r: r["results"].update(final_opened_once_enforced=True),
                   lambda r: r["results"].pop("final_opened_once_enforced"),
                   lambda r: r["results"].pop("protocol_hash")):
        rep = copy.deepcopy(rp.build_report(results))
        mutate(rep)
        with pytest.raises(rp.ReportIncomplete):
            rp.render_markdown(_rehash(rep))  # self-rehashed: integrity alone would pass


# --- H3: labelling is checked, not only hashed -----------------------------------------------------------

def test_evaluate_output_carries_the_banner_and_caveat(results):
    assert results["banner"] == rp.SYNTHETIC_LABEL and results["caveat"] == rp.CAVEAT
    assert "Barrett" in results["caveat"] and "acted" in results["caveat"].lower()


def test_non_synthetic_status_must_be_the_proxy_label_even_after_a_rehash(results):
    fake = copy.deepcopy(results)
    fake["adapter"]["source_class"] = "research_corpus"
    fake["banner"] = rp.PROXY_LABEL
    ok = rp.build_report(fake)
    assert ok["status"] == rp.PROXY_LABEL and rp.render_markdown(ok)
    for bad_status in ("whatever", rp.SYNTHETIC_LABEL, "RESEARCH"):
        rep = copy.deepcopy(ok)
        rep["status"] = bad_status
        with pytest.raises(rp.ReportIncomplete):
            rp.render_markdown(_rehash(rep))


def test_banner_and_caveat_in_results_must_match_the_report(results):
    rep = rp.build_report(results)
    for mutate in (lambda r: r["results"].update(banner="SOMETHING ELSE"), lambda r: r["results"].update(caveat="softer caveat"),
                   lambda r: r["results"].pop("banner")):
        bad = copy.deepcopy(rep)
        mutate(bad)
        with pytest.raises(rp.ReportIncomplete):
            rp.render_markdown(_rehash(bad))


# --- H5: the comparator is labelled for what it is ------------------------------------------------------------

def test_fer7_only_is_documented_as_sharing_hb1_gates(results):
    note = results["method_notes"]["fer7_only"]
    assert "NOT a pure no-temporal ablation" in note and "abstain" in note and "ignoring head motion" in note
    assert set(results["method_notes"]) == set(ev.METHODS)
    assert "NOT a pure no-temporal ablation" in ev.__doc__
    assert "fer7_only" in " ".join(results["notes"])
    assert "NOT a pure no-temporal ablation" in rp.render_markdown(rp.build_report(results))


def test_fer7_only_behaviour_is_unchanged_and_abstains_with_hb1(results):
    for role in ev.ROLES:
        a = results["endpoints"]["P1"][role]
        assert a["fer7_only"]["full_cohort"]["abstention_rate"] == a["hb1"]["full_cohort"]["abstention_rate"]
        assert a["fer7_only"]["full_cohort"]["balanced_accuracy"] == 1.0


def test_results_stay_json_serialisable_and_deterministic():
    a = ev.evaluate(SyntheticAdapter(n_participants=6), "synthetic", n_draws=30)
    b = ev.evaluate(SyntheticAdapter(n_participants=6), "synthetic", n_draws=30)
    assert json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


# --- S60 (review S59 findings 2 and 6) ---------------------------------------------------------------------------------------------

def test_s60_f2_sentinel_excluded_clips_are_not_confusers(results):
    for ep in ("P1", "P2"):
        c = results["confusers"][ep]
        assert "excluded" not in c["classes"] and c["classes"] == []  # the synthetic adapter has no real confuser class
        assert c["n_clips"] == 0 and c["hb1_positive_rate"] is None and c["abstention_rate"] is None
        assert c["excluded_n"] == 30 and c["excluded_n"] + c["n_clips"] == 30  # separate count, nothing dropped silently
        assert "excluded by the adapter" in c["note"]
    md = rp.render_markdown(rp.build_report(results))
    assert "excluded by the adapter: 30" in md and "HB-1 positive rate" in md and "n/a" in md


def test_s60_f2_real_confusers_report_numerator_and_denominator(monkeypatch):
    orig = SyntheticAdapter.label_map

    def label_map(self):  # the evaluator accepts only the exact reviewed class, so patch it rather than subclass it
        m = dict(orig(self))
        m["unsure-like"] = "happy_confuser"
        return m

    monkeypatch.setattr(SyntheticAdapter, "label_map", label_map)
    r = ev.evaluate(SyntheticAdapter(n_participants=6), "synthetic", n_draws=20)
    c = r["confusers"]["P1"]
    assert c["classes"] == ["happy_confuser"] and c["n_clips"] == 6 and c["excluded_n"] == 24
    assert c["n_answered"] + c["n_abstained"] == c["n_clips"]
    assert c["hb1_positive_n"] <= c["n_answered"]
    if c["n_answered"]:
        assert c["hb1_positive_rate"] == c["hb1_positive_n"] / c["n_answered"]  # numerator and denominator both stated
    assert c["abstention_rate"] == c["n_abstained"] / c["n_clips"]
    assert "answered" in c["rate_definition"] and "abstain" in c["rate_definition"]


def test_s60_f6_split_uses_largest_remainder_with_a_documented_tie_rule():
    expected = {3: (1, 1, 1), 4: (2, 1, 1), 5: (3, 1, 1), 6: (3, 1, 2), 7: (3, 2, 2), 8: (4, 2, 2), 9: (5, 2, 2), 10: (5, 2, 3),
                11: (5, 3, 3), 12: (6, 3, 3), 14: (7, 3, 4), 16: (8, 4, 4), 25: (13, 6, 6)}
    for n, sizes in expected.items():
        s = ev.actor_disjoint_split([f"a{i:02d}" for i in range(n)])
        assert tuple(len(s[r]) for r in ev.ROLES) == sizes, (n, {r: len(s[r]) for r in ev.ROLES})
        assert sum(sizes) == n and min(sizes) >= 1
        ev.assert_actor_disjoint(s, expected=[f"a{i:02d}" for i in range(n)])
    for n in range(3, 61):  # VAL and FINAL differ by at most one actor, DEV never smaller than either
        s = ev.actor_disjoint_split([f"a{i:02d}" for i in range(n)])
        d, v, f = (len(s[r]) for r in ev.ROLES)
        assert abs(v - f) <= 1 and d >= max(v, f) and d + v + f == n and min(d, v, f) >= 1, (n, d, v, f)
    assert "largest remainder" in ev.actor_disjoint_split.__doc__ and "ties" in ev.actor_disjoint_split.__doc__


def test_s60_f6_split_is_still_deterministic_and_order_independent():
    ids = [f"a{i:02d}" for i in range(13)]
    assert ev.actor_disjoint_split(ids) == ev.actor_disjoint_split(list(reversed(ids)))
    assert ev.actor_disjoint_split(ids) != ev.actor_disjoint_split(ids, seed=1)
