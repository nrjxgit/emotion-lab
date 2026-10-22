"""Pilot annotation schema, agreement statistics and window scoring (EL-006 S15). Synthetic streams and manufactured annotations only."""
import copy
import json
import math
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from socialcue import state as hb1  # noqa: E402
from socialcue import synthetic as syn  # noqa: E402
from socialcue.data import declarations, registry, rights  # noqa: E402
from socialcue.data.rights import RightsRefusal, authorize  # noqa: E402
from socialcue.data.synthetic_adapter import SyntheticAdapter  # noqa: E402
from socialcue.eval import report as eval_report  # noqa: E402
from socialcue.pilot import agreement as G  # noqa: E402
from socialcue.pilot import annotations as A  # noqa: E402
from socialcue.pilot import window_scoring as W  # noqa: E402

PILOT_DIR = ROOT / "socialcue" / "pilot"
ANN1, ANN2 = "ANN-0000a1", "ANN-0000b2"


def pt(i):
    return f"PT-5eed{i:04x}"


_UNSET = object()


def rec(annotator=ANN1, participant=_UNSET, start=0.0, labels=None, **flags):
    base = {"schema_version": 1, "annotator": annotator, "participant": pt(1) if participant is _UNSET else participant, "window_start": start,
            "window_end": start + 3.0, "labels": {l: "absent" for l in A.LABELS}, "unscorable": False, "unsure_rate": False,
            "confound_flag": False}
    base["labels"].update(labels or {})
    base.update(flags)
    return base


# --- annotation schema ---------------------------------------------------------------------------------

def test_valid_record_roundtrips_and_label_set_is_observable_behaviour_only():
    r = A.validate_annotation(rec(labels={"look_away": "present"}))
    assert r["labels"]["look_away"] == "present" and r["window_end"] - r["window_start"] == 3.0
    assert A.LABELS == ("look_away", "pause", "asks_help", "fidget")
    for word in ("nervous", "unsure", "confus", "feel", "emotion", "help_needed"):
        assert not any(word in l for l in A.LABELS)
    assert A.derived_no_cue(r) is False and A.derived_no_cue(A.validate_annotation(rec())) is True
    assert A.derived_no_cue(A.validate_annotation(rec(unscorable=True))) is False  # unscorable is never "no cue"


@pytest.mark.parametrize("key", ["name", "full_name", "email", "phone", "address", "dob", "age", "face", "image", "video", "landmarks",
                                 "note", "notes", "comment", "free_text", "description", "emotion", "feeling", "nervous", "ip", "gender"])
def test_pii_shaped_and_free_text_fields_are_rejected(key):
    r = rec()
    r[key] = "anything"
    with pytest.raises(A.AnnotationError) as exc:
        A.validate_annotation(r)
    assert any(c.startswith("pii_or_free_text_field") for c in exc.value.codes)


def test_unknown_keys_missing_keys_and_wrong_types_are_rejected():
    r = rec()
    r["mystery"] = 1
    with pytest.raises(A.AnnotationError, match="unknown_key:mystery"):
        A.validate_annotation(r)
    r = rec()
    del r["participant"]
    with pytest.raises(A.AnnotationError, match="missing_key:participant"):
        A.validate_annotation(r)
    for bad in (None, [], "x", 5):
        with pytest.raises(A.AnnotationError, match="not_a_dict"):
            A.validate_annotation(bad)
    for mutate, code in [
        (lambda r: r["labels"].update(look_away="maybe"), "label_values"),
        (lambda r: r["labels"].update(look_away=True), "label_values"),
        (lambda r: r["labels"].update(no_cue="present"), "labels_must_be_exactly"),
        (lambda r: r["labels"].pop("pause"), "labels_must_be_exactly"),
        (lambda r: r.update(labels="free text feelings"), "labels_must_be_exactly"),
        (lambda r: r.update(unscorable="yes"), "flag_must_be_bool:unscorable"),
        (lambda r: r.update(confound_flag=1), "flag_must_be_bool:confound_flag"),
        (lambda r: r.update(window_end=r["window_start"] + 2.0), "window_must_be_3s"),
        (lambda r: r.update(window_start=-3.0, window_end=0.0), "window_must_be_3s"),
        (lambda r: r.update(window_start=float("nan")), "bad_window_times"),
        (lambda r: r.update(window_start="0"), "bad_window_times"),
        (lambda r: r.update(schema_version=2), "bad_schema_version"),
        (lambda r: r.update(schema_version=True), "bad_schema_version"),
    ]:
        r = rec()
        mutate(r)
        with pytest.raises(A.AnnotationError, match=re.escape(code)):
            A.validate_annotation(r)


@pytest.mark.parametrize("who", ["Alice", "alice@example.com", "PT-xyz", "PT-12", "pt-0000a1", "ANN-0000a1", "Bob Smith", "07700 900123", "", None, 7])
def test_participant_tokens_must_be_opaque_codes(who):
    with pytest.raises(A.AnnotationError, match="bad_participant_token"):
        A.validate_annotation(rec(participant=who))


@pytest.mark.parametrize("who", ["Alice", "PT-0000a1", "ann-0000a1", "ANN-12", "", None])
def test_annotator_tokens_must_be_opaque_codes(who):
    with pytest.raises(A.AnnotationError, match="bad_annotator_token"):
        A.validate_annotation(rec(annotator=who))


def test_all_problems_are_reported_together():
    r = rec(annotator="Alice", participant="Bob")
    r["labels"]["pause"] = "maybe"
    with pytest.raises(A.AnnotationError) as exc:
        A.validate_annotation(r)
    assert {"bad_annotator_token", "bad_participant_token", "label_values_must_be_present_or_absent"} <= set(exc.value.codes)


def test_batch_rejects_overlaps_and_duplicates_per_annotator_and_participant():
    ok = A.validate_batch([rec(start=0.0), rec(start=3.0), rec(ANN2, start=0.0), rec(participant=pt(2), start=0.0)])
    assert len(ok) == 4
    with pytest.raises(A.AnnotationError, match="overlapping_or_duplicate_windows"):
        A.validate_batch([rec(start=0.0), rec(start=0.0)])
    with pytest.raises(A.AnnotationError, match="overlapping_or_duplicate_windows"):
        A.validate_batch([rec(start=0.0), rec(start=1.0)])
    with pytest.raises(A.AnnotationError):
        A.validate_batch([rec(), {"name": "x"}])


def test_input_is_not_mutated_and_output_is_a_copy():
    src = rec(labels={"pause": "present"})
    snapshot = copy.deepcopy(src)
    out = A.validate_annotation(src)
    out["labels"]["pause"] = "absent"
    assert src == snapshot


# --- agreement: hand-computed cases -----------------------------------------------------------------------------

def _lists(a, b, c, d):
    """2x2 counts: a both-present, b only first, c only second, d both-absent."""
    first = ["present"] * a + ["present"] * b + ["absent"] * c + ["absent"] * d
    second = ["present"] * a + ["absent"] * b + ["present"] * c + ["absent"] * d
    return first, second


def test_kappa_classic_two_by_two():
    first, second = _lists(20, 5, 10, 15)  # n=50; po=.70; pe=(25*30+25*20)/2500=.50
    k = G.cohen_kappa(first, second)
    assert k["n"] == 50 and k["observed"] == pytest.approx(0.70) and k["expected"] == pytest.approx(0.50)
    assert k["kappa"] == pytest.approx(0.40) and k["reason"] is None


def test_kappa_perfect_and_perfectly_opposed_with_variation():
    assert G.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0])["kappa"] == 1.0
    assert G.cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1])["kappa"] == -1.0
    assert G.cohen_kappa(["a", "b", "c", "a"], ["a", "b", "c", "a"])["kappa"] == 1.0  # three categories
    k = G.cohen_kappa(["a", "a", "b", "b"], ["a", "b", "b", "b"])  # po=.75, pe=.5*.25+.5*.75=.5
    assert k["kappa"] == pytest.approx(0.5)


def test_kappa_degenerate_cases_are_none_with_a_reason_never_one():
    same = G.cohen_kappa(["absent"] * 10, ["absent"] * 10)
    assert same["kappa"] is None and same["reason"] == "no_variation" and same["observed"] == 1.0
    assert G.cohen_kappa([1], [1])["kappa"] is None and G.cohen_kappa([1], [1])["reason"] == "too_few_items"
    assert G.cohen_kappa([], [])["reason"] == "too_few_items"
    assert G.cohen_kappa([1, 0], [1])["reason"] == "length_mismatch"
    assert G.cohen_kappa(5, [1])["reason"] == "not_sequences"
    one_sided = G.cohen_kappa(["x"] * 6, ["x"] * 3 + ["y"] * 3)  # rater 1 constant, rater 2 not
    assert one_sided["kappa"] == pytest.approx(0.0)
    assert G.cohen_kappa([None, None, 1], [1, None, 1])["reason"] == "too_few_items"


def test_kappa_drops_missing_pairs_and_counts_them():
    k = G.cohen_kappa([1, 0, None, 1, 0, float("nan")], [1, 0, 1, None, 1, 0])
    assert k["n"] == 3 and k["dropped_missing"] == 3 and k["kappa"] == pytest.approx(0.4, abs=0.5) and k["observed"] == pytest.approx(2 / 3)


def test_alpha_hand_computed_cases():
    # two raters, units (a,a) (a,b) (b,b) (b,a): o_aa=2, o_bb=2, o_ab=o_ba=2; n=8; 1 - 7*4/32
    assert G.krippendorff_alpha_nominal([["a", "a"], ["a", "b"], ["b", "b"], ["b", "a"]])["alpha"] == pytest.approx(0.125)
    # one unit of three raters (a,a,b): 1 - 2*2/4
    assert G.krippendorff_alpha_nominal([["a", "a", "b"]])["alpha"] == pytest.approx(0.0)
    # add the unit (b,b,b): n=6, off-diagonal 2, expected 16 -> 1 - 5*2/16
    assert G.krippendorff_alpha_nominal([["a", "a", "b"], ["b", "b", "b"]])["alpha"] == pytest.approx(0.375)
    assert G.krippendorff_alpha_nominal([["a", "a"], ["b", "b"], ["a", "a"], ["b", "b"]])["alpha"] == 1.0


def test_alpha_missing_values_and_degenerate_cases():
    r = G.krippendorff_alpha_nominal([["a", "a"], ["a", None], ["b", "b"], [None, None], [None, "b"]])
    assert r["alpha"] == 1.0 and r["n_pairable"] == 4  # units with fewer than two values are not pairable
    r = G.krippendorff_alpha_nominal([["a", None, "a"], ["b", "b", None]])
    assert r["alpha"] == 1.0 and r["n_pairable"] == 4
    for units, reason in [([["a", "a"], ["a", "a"]], "no_variation"), ([["a"], ["b"]], "too_few_pairable_values"), ([], "too_few_pairable_values"),
                          ([[None, None]], "too_few_pairable_values"), (5, "not_sequences")]:
        out = G.krippendorff_alpha_nominal(units)
        assert out["alpha"] is None and out["reason"] == reason
    assert G.krippendorff_alpha_nominal([["a", "b"]] * 3)["alpha"] is not None


def test_alpha_close_to_kappa_for_two_raters_with_many_items():
    first, second = _lists(200, 50, 100, 150)
    a = G.krippendorff_alpha_nominal([[x, y] for x, y in zip(first, second)])["alpha"]
    k = G.cohen_kappa(first, second)["kappa"]
    assert abs(a - k) < 0.01


def test_label_agreement_small_n_withholds_interpretation_and_large_n_bands():
    first, second = _lists(5, 2, 3, 40)
    small = G.label_agreement(first, second)
    assert small["table"] == {"both_present": 5, "only_first": 2, "only_second": 3, "both_absent": 40} and small["positives_total"] == 15
    assert small["interpretation"] == "counts_only_fewer_than_20_positive_windows" and small["band"] is None
    assert small["kappa"] is None and small["kappa_reason"] == "fewer_than_20_positive_windows"  # withheld, not just un-banded
    assert small["raw_kappa_do_not_interpret"] is not None and small["raw_kappa_reason"] is None
    assert small["observed_agreement"] == pytest.approx(45 / 50) and small["pabak"] == pytest.approx(0.8) and small["bias_index"] == pytest.approx(0.02)
    assert small["prevalence_index"] == pytest.approx(35 / 50)
    big = G.label_agreement(*_lists(30, 4, 4, 30))
    assert big["interpretation"] == "interpretable" and big["band"] == "substantial_enough_to_use"
    mid = G.label_agreement(*_lists(25, 10, 10, 30))
    assert mid["band"] == "moderate_report_do_not_rely"
    low = G.label_agreement(*_lists(15, 20, 20, 20))
    assert low["band"] == "unreliable_drop_from_primary"
    none = G.label_agreement(["absent"] * 40, ["absent"] * 40)
    assert none["kappa"] is None and none["raw_kappa_reason"] == "no_variation" and none["kappa_reason"] == "fewer_than_20_positive_windows" and none["interpretation"] == "counts_only_fewer_than_20_positive_windows"
    assert G.band(0.6) == "substantial_enough_to_use" and G.band(0.4) == "moderate_report_do_not_rely" and G.band(0.39) == "unreliable_drop_from_primary" and G.band(None) is None


def test_bootstrap_by_participant_is_deterministic_resamples_participants_and_skips_undefined():
    items = {pt(i): [("x", "x")] * 4 for i in range(6)}
    items[pt(9)] = [("x", "y")] * 4  # one participant holds every disagreement

    def acc(its):
        return sum(1 for a, b in its if a == b) / len(its)

    a = G.bootstrap_by_participant(items, acc)
    assert a == G.bootstrap_by_participant(items, acc)
    varied = {pt(i): [(1, 1)] * (6 - i) + [(1, 0)] * i for i in range(6)}  # a continuous statistic, so seeds differ visibly
    assert G.bootstrap_by_participant(varied, acc, 40) != G.bootstrap_by_participant(varied, acc, 40, seed=7)  # few draws: the percentiles move with the seed
    assert G.N_DRAWS == 1000 and G.SEED == 505
    assert a["estimate"] == pytest.approx(24 / 28) and a["lo"] < a["estimate"] <= a["hi"] == 1.0 and a["n_valid"] == 1000 and a["n_participants"] == 7
    assert a["lo"] < 0.8  # clip-level resampling could not move this far: whole participants drop out together
    one = G.bootstrap_by_participant({pt(1): [(1, 1)]}, acc)
    assert one["reason"] == "fewer_than_2_participants" and one["lo"] is None
    never = G.bootstrap_by_participant({pt(1): [1], pt(2): [2]}, lambda its: None)
    assert never["reason"] == "statistic_undefined_in_every_draw" and never["n_valid"] == 0
    kappa = G.bootstrap_by_participant({pt(i): _pairs(i) for i in range(8)}, lambda its: G.cohen_kappa([x for x, _ in its], [y for _, y in its])["kappa"], 200)
    assert kappa["lo"] <= kappa["estimate"] <= kappa["hi"]


def _pairs(i):
    return [(1, 1), (0, 0), (1, 0 if i % 2 else 1), (0, 0)]


# --- window alignment ---------------------------------------------------------------------------------------------

def _blocks(*starts, **kw):
    return [A.validate_annotation(rec(start=s, **kw)) for s in starts]


def test_alignment_inherits_the_block_holding_at_least_half_the_window():
    blocks = _blocks(0.0, 3.0)
    fired = frozenset()
    windows = [(0.0, 3.0, fired, False), (1.0, 4.0, fired, False), (1.5, 4.5, fired, False), (2.0, 5.0, fired, False),
               (4.0, 7.0, fired, False), (5.5, 8.5, fired, False)]
    got = [(w[0], i) for w, i in W.align(windows, blocks)]
    assert got == [(0.0, 0), (1.0, 0), (1.5, 0), (2.0, 1), (4.0, 1)]  # 1.5/1.5 tie -> earlier block; (5.5..) has 0.5 s overlap -> dropped
    assert W.align(windows, []) == []


def test_truth_mapping_and_sustained_pause_look_away():
    blocks = A.validate_batch([rec(start=0.0, labels={"fidget": "present"}), rec(start=3.0, labels={"pause": "present", "look_away": "present"}),
                               rec(start=6.0, labels={"pause": "present", "look_away": "present"}), rec(start=9.0, labels={"pause": "present", "look_away": "present"}),
                               rec(start=15.0, labels={"pause": "present", "look_away": "present"}), rec(start=18.0, labels={"asks_help": "present"}),
                               rec(start=21.0, unscorable=True, labels={"fidget": "present"})])
    t = lambda target, i: W._truth(target, blocks, i)  # noqa: E731
    assert t("state:nervous-like", 0) is True and t("state:unsure-like", 0) is False and t("raw:fidget", 0) is True and t("raw:pause", 0) is False
    assert t("state:unsure-like", 1) is True and t("state:nervous-like", 1) is True  # look_away maps to both
    assert [t("state:needs-help-like", i) for i in (0, 1, 2, 3, 4, 5)] == [False, True, True, True, False, True]  # block 4 stands alone (gap before it)
    assert t("raw:fidget", 6) is None and t("state:needs-help-like", 6) is None  # unscorable is excluded


def test_state_map_is_registered_with_the_protocol_mapping():
    assert W.STATE_MAP["nervous-like"] == ("fidget", "look_away") and W.STATE_MAP["unsure-like"] == ("pause", "look_away")
    assert W.STATE_MAP["needs-help-like"][0] == "asks_help" and "sustained" in W.STATE_MAP["needs-help-like"][1]
    assert "50%" in W.ALIGNMENT_RULE and "hypothesis" in W.__doc__


# --- end to end on synthetic streams and manufactured annotations -----------------------------------------------------

def _cohort(n_unsure, n_calm, flip=0):
    streams, anns = {}, []
    for i in range(n_unsure + n_calm):
        unsure = i < n_unsure
        p = pt(i + 1)
        streams[p] = syn.generate("unsure-like" if unsure else "calm", seed=500 + i, duration=9.0)
        if i == 0:  # a face dropout in the middle of the first participant's stream, so some windows must abstain
            streams[p] = syn.build(9.0, 8.0, 500, uns=lambda t: 1.0, drop=lambda t, u: 4.0 <= t < 7.0)
        for a in (ANN1, ANN2):
            for b in range(3):
                labels = {"look_away": "present", "pause": "present"} if unsure else {}
                if a == ANN2 and flip and (i + b) % flip == 0:
                    labels = {} if unsure else {"look_away": "present"}
                anns.append(rec(a, p, 3.0 * b, labels))
    return streams, anns


@pytest.fixture(scope="module")
def small():
    streams, anns = _cohort(6, 6)
    return W.score_pilot(SyntheticAdapter(), "synthetic", streams, anns, n_draws=60)


@pytest.fixture(scope="module")
def evaluable():
    streams, anns = _cohort(12, 3, flip=5)
    return W.score_pilot(SyntheticAdapter(), "synthetic", streams, anns, n_draws=60)


def test_scoring_structure_counts_and_exploratory_flag(small):
    r = small
    assert r["exploratory"] is True and set(r["splits"]) == {"ALL"} and len(r["splits"]["ALL"]) == 12
    assert r["counts"] == {"participants": 12, "participants_scored": 12, "annotators": 2, "blocks": 72, "unscorable_blocks": 0}
    assert set(r["scoring"]) == set(W.TARGETS) and r["views"] == [ANN1, ANN2, "consensus"]
    assert r["banner"] == "SYNTHETIC / NOT REAL MODEL PERFORMANCE" and r["caveat"] == eval_report.CAVEAT
    assert r["final_opened_once_enforced"] is False and r["protocol_hash"] is None
    assert r["manifest"]["decision"]["result"] == "allowed" and r["config_sha256"] == hb1.CONFIG_SHA256
    json.dumps(r, allow_nan=False)


def test_hb1_matches_manufactured_annotations_by_construction_and_abstentions_are_counted(small):
    d = small["scoring"]["state:unsure-like"][ANN1]["ALL"]
    assert d["n_windows"] > 0 and d["n_positive_windows"] > 0 and d["n_participants_with_positive"] == 6
    full = d["hb1"]["full_cohort"]
    assert d["n_abstained"] == full["confusion"]["abstained"] > 0  # the first window of each stream is still filling or short
    conf = full["confusion"]
    assert conf["tp"] + conf["fn"] + conf["tn"] + conf["fp"] == d["n_windows"]  # full cohort: an abstention is wrong on both classes
    assert conf["fn"] + conf["fp"] >= d["n_abstained"]  # every abstention is a miss or a false alarm in the full cohort
    ans = d["hb1"]["answered_only"]["confusion"]
    assert ans["abstained"] == d["n_abstained"] and ans["tp"] + ans["fn"] + ans["tn"] + ans["fp"] == d["n_windows"] - d["n_abstained"]
    assert full["balanced_accuracy"] is not None and full["abstention_rate"] == pytest.approx(d["n_abstained"] / d["n_windows"])
    nc = d["always_no_cue"]["full_cohort"]
    assert nc["recall"] == 0.0 and nc["balanced_accuracy"] == 0.5 and nc["abstention_rate"] == 0.0


def test_labels_with_too_few_participants_are_not_evaluable_and_the_rule_is_visible(small):
    for view in small["scoring"]["state:unsure-like"].values():
        d = view["ALL"]
        assert d["evaluable"] is False and d["not_evaluable_reason"] == "not_evaluable_at_this_n"  # 6 positive participants < 10
    md = W.render_pilot_markdown(W.build_pilot_report(small))
    assert "| false |" in md and ">= 10 participants" in md


def test_enough_participants_and_windows_make_a_target_evaluable(evaluable):
    d = evaluable["scoring"]["state:unsure-like"][ANN1]["ALL"]
    assert d["n_participants_with_positive"] == 12 and d["n_positive_windows"] >= 30 and d["evaluable"] is True and d["not_evaluable_reason"] is None
    cons = evaluable["scoring"]["state:unsure-like"]["consensus"]["ALL"]
    assert 0 < cons["n_windows"] < evaluable["scoring"]["state:unsure-like"][ANN1]["ALL"]["n_windows"]  # disagreeing windows are left out of consensus
    ci = d["hb1"]["ci_balanced_accuracy_by_participant"]
    assert ci["n_participants"] == 15 and ci["lo"] is not None and ci["lo"] <= ci["estimate"] <= ci["hi"]


def test_agreement_between_manufactured_annotators(small, evaluable):
    perfect = small["agreement"]["look_away"]
    assert perfect["pair"]["table"] == {"both_present": 18, "only_first": 0, "only_second": 0, "both_absent": 18} and perfect["pair"]["kappa"] == 1.0
    assert perfect["alpha"]["alpha"] == 1.0 and perfect["pair"]["interpretation"] == "interpretable"
    assert small["agreement"]["fidget"]["pair"]["kappa"] is None and small["agreement"]["fidget"]["pair"]["raw_kappa_reason"] == "no_variation"
    assert small["agreement"]["fidget"]["pair"]["interpretation"] == "counts_only_fewer_than_20_positive_windows"
    imperfect = evaluable["agreement"]["look_away"]
    assert imperfect["pair"]["kappa"] < 1.0 and imperfect["pair"]["table"]["only_first"] > 0
    ci = imperfect["kappa_ci_by_participant"]
    assert ci["n_participants"] >= 12 and ci["lo"] <= ci["estimate"] <= ci["hi"]
    assert imperfect["alpha"]["alpha"] is not None and imperfect["alpha"]["alpha"] < 1.0


def test_unscorable_blocks_are_excluded_and_counted():
    streams, anns = _cohort(2, 1)
    for a in anns:
        if a["participant"] == pt(1) and a["window_start"] == 3.0:
            a["unscorable"] = True
    r = W.score_pilot(SyntheticAdapter(), "synthetic", streams, anns, n_draws=20)
    assert r["counts"]["unscorable_blocks"] == 2 and r["unscorable_by_annotator"] == {ANN1: 1, ANN2: 1}
    clean = W.score_pilot(SyntheticAdapter(), "synthetic", *_cohort(2, 1), n_draws=20)
    assert r["scoring"]["raw:pause"][ANN1]["ALL"]["n_windows"] < clean["scoring"]["raw:pause"][ANN1]["ALL"]["n_windows"]
    assert r["agreement"]["pause"]["pair"]["n"] == clean["agreement"]["pause"]["pair"]["n"] - 1


def test_missing_streams_and_single_annotator_do_not_crash():
    streams, anns = _cohort(2, 2)
    del streams[pt(1)]
    r = W.score_pilot(SyntheticAdapter(), "synthetic", streams, anns, n_draws=20)
    assert r["counts"]["participants"] == 4 and r["counts"]["participants_scored"] == 3
    solo = [a for a in anns if a["annotator"] == ANN1]
    r = W.score_pilot(SyntheticAdapter(), "synthetic", streams, solo, n_draws=20)
    assert r["views"] == [ANN1] and "pair" not in r["agreement"]["pause"] and r["agreement"]["pause"]["alpha"]["alpha"] is None
    json.dumps(r, allow_nan=False)


def test_participant_disjoint_split_reuses_the_eval_helper_when_n_is_large_enough():
    streams, anns = _cohort(15, 10)
    r = W.score_pilot(SyntheticAdapter(), "synthetic", streams, anns, n_draws=10)
    assert r["exploratory"] is False and set(r["splits"]) == {"DEV", "VAL", "FINAL"}
    from socialcue.eval import proxy_eval as pe

    pe.assert_actor_disjoint(r["splits"])
    assert sum(len(v) for v in r["splits"].values()) == 25 and W.actor_disjoint_split is pe.actor_disjoint_split
    assert set(r["scoring"]["raw:pause"][ANN1]) == {"DEV", "VAL", "FINAL"}


def test_scoring_is_deterministic_and_inputs_are_not_mutated():
    streams, anns = _cohort(3, 3)
    snap = copy.deepcopy(anns)
    a = W.score_pilot(SyntheticAdapter(), "synthetic", streams, anns, n_draws=30)
    assert anns == snap
    assert json.dumps(a, sort_keys=True) == json.dumps(W.score_pilot(SyntheticAdapter(), "synthetic", *_cohort(3, 3), n_draws=30), sort_keys=True)


def test_invalid_annotations_stop_scoring_before_anything_is_computed():
    streams, anns = _cohort(2, 1)
    bad = anns + [{**rec(), "name": "Alice"}]
    with pytest.raises(A.AnnotationError):
        W.score_pilot(SyntheticAdapter(), "synthetic", streams, bad)


# --- the pilot adapter cannot be scored ----------------------------------------------------------------------------------

def test_pilot_declaration_is_blocked_in_the_allowlist_and_needs_an_ethics_token_field():
    entry = rights.load_allowlist()[0]["pilot_consented"]
    assert entry["status"] == "blocked_pending_operator" and entry["requires_ethics_approval"] is True and entry["source_class"] == "consented_pilot"
    a = declarations.PilotDeclaration()
    assert registry._FACTORIES["pilot_consented"] is declarations.PilotDeclaration and a.ethics_approval_token is None
    with pytest.raises(RightsRefusal) as exc:
        authorize(a, "research_evaluation", "research_noncommercial")
    assert exc.value.code == "ETHICS_APPROVAL_MISSING"
    for bad in (None, True, 12345678, "", "short", "        ", ["approval-0001"]):
        class T(declarations.PilotDeclaration):
            ethics_approval_token = bad

        with pytest.raises(RightsRefusal) as exc:
            authorize(T(), "research_evaluation", "research_noncommercial")
        assert exc.value.code == "ETHICS_APPROVAL_MISSING", bad


def test_a_token_string_is_not_an_approval_the_allowlist_still_blocks():
    class Token(declarations.PilotDeclaration):
        ethics_approval_token = "approval-0001-made-up"
        blocked_reason = None

    with pytest.raises(RightsRefusal) as exc:
        authorize(Token(), "research_evaluation", "research_noncommercial")
    assert exc.value.code == "BLOCKED_NEEDS_OPERATOR_DECISION"
    assert "NOT an approval" in declarations.PilotDeclaration.__doc__ and "NOT an approval" in rights.load_allowlist()[0]["pilot_consented"]["note"]
    assert "EXISTS" in rights.__doc__ and "not an approval" in rights.__doc__
    with pytest.raises(RightsRefusal):
        Token().items("research_evaluation", "research_noncommercial")
    with pytest.raises(RightsRefusal):
        registry.open_adapter("pilot_consented", "research_evaluation", "research_noncommercial")


def test_use_and_scope_refusals_still_come_first_for_the_pilot():
    a = declarations.PilotDeclaration()
    for use, scope, code in [("commercial_training", "commercial_clean", "USE_NOT_ALLOWED"), ("research_evaluation", "commercial_clean", "SCOPE_MISMATCH"),
                             ("research_training", "research_noncommercial", "USE_NOT_ALLOWED")]:
        with pytest.raises(RightsRefusal) as exc:
            authorize(a, use, scope)
        assert exc.value.code == code


def test_score_pilot_cannot_be_pointed_at_the_pilot_adapter_or_at_non_adapters():
    streams, anns = _cohort(2, 1)
    with pytest.raises(RightsRefusal):
        W.score_pilot(declarations.PilotDeclaration(), "research_noncommercial", streams, anns)

    class Token(declarations.PilotDeclaration):
        ethics_approval_token = "approval-0001-made-up"

    with pytest.raises(RightsRefusal) as exc:
        W.score_pilot(Token(), "research_noncommercial", streams, anns)
    assert exc.value.code == "BLOCKED_NEEDS_OPERATOR_DECISION"
    for bad in ("C:/pilot/data", pathlib.Path("."), {"items": []}, None):
        with pytest.raises(TypeError):
            W.score_pilot(bad, "synthetic", streams, anns)


def test_other_allowlist_entries_are_unchanged_by_the_pilot_entry():
    entries = rights.load_allowlist()[0]
    assert {n for n, e in entries.items() if e["status"] == "granted"} == {"synthetic", "synthetic_commercial_clean"}
    assert entries["ravdess"]["status"] == entries["crema_d"]["status"] == "blocked_pending_operator"
    assert "requires_ethics_approval" not in entries["ravdess"]


# --- report ------------------------------------------------------------------------------------------------------------------

def test_report_always_carries_banner_caveat_claims_and_limitations(small, tmp_path):
    rep = W.build_pilot_report(small)
    assert rep["status"] == "SYNTHETIC / NOT REAL MODEL PERFORMANCE"
    md = W.render_pilot_markdown(rep)
    for text in (rep["status"], eval_report.CAVEAT, W.PILOT_LIMITATIONS, *W.PILOT_ALLOWED_CLAIMS, *W.PILOT_DISALLOWED_CLAIMS,
                 rep["manifest_sha256"], rep["hb1_config_sha256"], "Volunteer sample, scripted mock tasks, small N, single site."):
        assert text in md, text
    assert "Research prototype. On held-out actors of [dataset]" in W.PILOT_LIMITATIONS  # the verbatim section-5 block
    assert md.splitlines()[0] == "# SYNTHETIC / NOT REAL MODEL PERFORMANCE" and "Barrett" in md
    jp, mp = W.write_pilot_report(rep, str(tmp_path))
    assert json.loads(pathlib.Path(jp).read_text(encoding="utf-8"))["status"] == rep["status"] and pathlib.Path(mp).read_text(encoding="utf-8") == md
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pilot_report.json", "pilot_report.md"]
    with pytest.raises(FileExistsError):
        W.write_pilot_report(rep, str(tmp_path))
    assert rep == W.build_pilot_report(copy.deepcopy(small))


def test_non_synthetic_runs_are_labelled_pilot_draft_and_cannot_pose_as_synthetic(small):
    fake = copy.deepcopy(small)
    fake["adapter"]["source_class"] = "consented_pilot"
    fake["banner"] = W.PILOT_LABEL
    rep = W.build_pilot_report(fake)
    assert W.PILOT_LABEL == "PILOT-DRAFT" and rep["status"] == "PILOT-DRAFT"
    md = W.render_pilot_markdown(rep)
    assert eval_report.CAVEAT in md and md.splitlines()[0] == "# PILOT-DRAFT"
    for status in (eval_report.SYNTHETIC_LABEL, "whatever", "PILOT"):
        bad = copy.deepcopy(rep)
        bad["status"] = status
        bad["report_sha256"] = W.sha256_of({k: v for k, v in bad.items() if k != "report_sha256"})
        with pytest.raises(eval_report.ReportIncomplete):
            W.render_pilot_markdown(bad)


@pytest.mark.parametrize("key", W.REQUIRED)
def test_report_refuses_to_render_without_a_required_part(small, key, tmp_path):
    rep = W.build_pilot_report(small)
    rep[key] = ""
    with pytest.raises(eval_report.ReportIncomplete):
        W.render_pilot_markdown(rep)
    with pytest.raises(eval_report.ReportIncomplete):
        W.write_pilot_report(rep, str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_report_refuses_edited_text_hidden_limitations_or_edited_numbers(small):
    base = W.build_pilot_report(small)

    def rehash(r):
        r["report_sha256"] = W.sha256_of({k: v for k, v in r.items() if k != "report_sha256"})
        return r

    for mutate in (lambda r: r.update(caveat="softer"), lambda r: r.update(limitations="short"), lambda r: r.update(allowed_claims=["HB-1 reads nervousness."]),
                   lambda r: r.update(disallowed_claims=[]), lambda r: r["results"].update(final_opened_once_enforced=True),
                   lambda r: r["results"].pop("protocol_hash"), lambda r: r["results"].update(banner="OTHER"), lambda r: r.update(manifest_sha256="0" * 64)):
        bad = copy.deepcopy(base)
        mutate(bad)
        with pytest.raises(eval_report.ReportIncomplete):
            W.render_pilot_markdown(rehash(bad))
    edited = copy.deepcopy(base)
    edited["results"]["counts"]["participants"] = 999
    with pytest.raises(eval_report.ReportIncomplete):
        W.render_pilot_markdown(edited)  # hash mismatch without a rehash


def test_report_text_never_claims_a_feeling(small):
    md = W.render_pilot_markdown(W.build_pilot_report(small))
    assert not re.search(r"\b(is nervous|is unsure|feels?|feeling of|detects nervousness)\b", md, re.I)


# --- purity ------------------------------------------------------------------------------------------------------------------------

def test_pilot_modules_have_no_file_network_media_or_numpy_code():
    for path in sorted(PILOT_DIR.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        for needle in ("import numpy", "import cv2", "import socket", "requests", "urllib", "subprocess", "glob", "listdir", "pathlib", "os.environ",
                       "getenv", "imread", "VideoCapture", "np.load", "__import__", "eval(", "exec("):
            assert needle not in src, (path.name, needle)
        if path.name != "window_scoring.py":
            assert "open(" not in src, path.name
    ws = (PILOT_DIR / "window_scoring.py").read_text(encoding="utf-8")
    assert ws.count("open(") == 2 and ws.count(', "x"') == 2


def test_import_has_no_side_effects_and_loads_no_heavy_runtime():
    code = (
        "import sys, json, hashlib, math, copy, random, dataclasses, typing, builtins, io, os, socket, abc, functools, re\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import socialcue.state, socialcue.data.rights, socialcue.data.provenance, socialcue.data.registry, socialcue.eval.proxy_eval, socialcue.eval.report\n"
        "before = set(sys.modules)\n"
        "def boom(*a, **k):\n"
        "    raise RuntimeError('side effect at import')\n"
        "builtins.open = boom; io.open = boom; os.open = boom\n"
        "os.listdir = boom; os.scandir = boom; os.makedirs = boom; os.mkdir = boom\n"
        "socket.socket = boom; socket.create_connection = boom; socket.getaddrinfo = boom\n"
        "import socialcue.pilot, socialcue.pilot.annotations, socialcue.pilot.agreement, socialcue.pilot.window_scoring\n"
        "new = set(sys.modules) - before\n"
        "bad = [m for m in new if m.split('.')[0] in ('numpy','cv2','torch','onnxruntime','mediapipe','requests','urllib3','http','subprocess','tempfile','socket')]\n"
        "assert not bad, bad\n"
        "from socialcue.pilot import annotations as A, agreement as G\n"
        "assert G.cohen_kappa([1,0,1,0],[1,0,1,0])['kappa'] == 1.0\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "OK"


# --- review S16 fixes (EL-006 S16b) ---------------------------------------------------------------------------------------------

REAL_LOOKING = "PT-9f3c1b22"


def _rows(md, label):
    return [[c.strip() for c in line.strip().strip("|").split("|")] for line in md.splitlines() if line.startswith(f"| {label} |")]


def test_p1_kappa_and_alpha_are_withheld_below_twenty_positive_windows_everywhere():
    streams, anns = _cohort(1, 1)  # 2 participants, 3 blocks each: a handful of positive windows
    r = W.score_pilot(SyntheticAdapter(), "synthetic", streams, anns, n_draws=20)
    pause = r["agreement"]["pause"]
    assert pause["pair"]["positives_total"] < 20
    assert pause["pair"]["kappa"] is None and pause["pair"]["kappa_reason"] == "fewer_than_20_positive_windows"
    assert pause["pair"]["raw_kappa_do_not_interpret"] == 1.0  # the raw value is kept, under a name that says what it is
    assert pause["alpha"]["alpha"] is None and pause["alpha"]["reason"] == "fewer_than_20_positive_windows" and pause["alpha"]["raw_alpha_do_not_interpret"] == 1.0
    ci = pause["kappa_ci_by_participant"]
    assert ci["estimate"] is None and ci["lo"] is None and ci["reason"] == "fewer_than_20_positive_windows"
    for lab in A.LABELS:  # no label may show a number from fewer than 20 positives
        e = r["agreement"][lab]
        if e["pair"]["positives_total"] < 20:
            assert e["pair"]["kappa"] is None and e["alpha"]["alpha"] is None
    json.dumps(r, allow_nan=False)
    md = W.render_pilot_markdown(W.build_pilot_report(r))
    row = _rows(md, "pause")[0]
    assert row[5] == "n/a" and row[6] == "[n/a, n/a]" and row[7] == "counts_only_fewer_than_20_positive_windows", row
    assert "1.000" not in "|".join(row[5:7])


def test_p1_label_with_enough_positives_still_shows_its_numbers(small):
    md = W.render_pilot_markdown(W.build_pilot_report(small))
    row = _rows(md, "look_away")[0]
    assert row[5] == "1.000" and row[7] == "interpretable"
    assert small["agreement"]["look_away"]["alpha"]["alpha"] == 1.0 and "raw_alpha_do_not_interpret" not in small["agreement"]["look_away"]["alpha"]


def test_p1_gate_alpha_helper():
    raw = G.krippendorff_alpha_nominal([["present", "present"], ["absent", "absent"]])
    assert raw["alpha"] == 1.0
    gated = G.gate_alpha(raw, 4)
    assert gated["alpha"] is None and gated["reason"] == G.WITHHELD and gated["raw_alpha_do_not_interpret"] == 1.0 and raw["alpha"] == 1.0
    assert G.gate_alpha(raw, 20)["alpha"] == 1.0 and "raw_alpha_do_not_interpret" not in G.gate_alpha(raw, 20)


def test_p2_real_looking_participant_codes_are_refused_under_the_synthetic_adapter():
    streams = {REAL_LOOKING: syn.generate("calm", duration=9.0)}
    anns = [rec(ANN1, REAL_LOOKING, 0.0), rec(ANN2, REAL_LOOKING, 0.0)]
    with pytest.raises(W.PilotInputError) as exc:
        W.score_pilot(SyntheticAdapter(), "synthetic", streams, anns, n_draws=10)
    assert exc.value.codes == ["real_looking_participant_codes_under_synthetic_adapter"] and isinstance(exc.value, ValueError)
    ok_streams, ok_anns = _cohort(1, 1)
    mixed = dict(ok_streams, **streams)
    with pytest.raises(W.PilotInputError):
        W.score_pilot(SyntheticAdapter(), "synthetic", mixed, ok_anns, n_draws=10)  # a stray real-looking stream is enough
    with pytest.raises(W.PilotInputError):
        W.score_pilot(SyntheticAdapter(), "synthetic", ok_streams, ok_anns + anns, n_draws=10)  # so is a stray annotation
    assert A.is_synthetic_participant(pt(1)) and not A.is_synthetic_participant(REAL_LOOKING) and not A.is_synthetic_participant("PT-5eed12")
    assert not A.is_synthetic_participant(None) and not A.is_synthetic_participant("PT-5eed00001")


def test_p2_synthetic_codes_cannot_enter_a_non_synthetic_run(monkeypatch):
    class Token(declarations.PilotDeclaration):
        ethics_approval_token = "approval-0001-made-up"

    monkeypatch.setattr(W, "authorize", lambda *a, **k: None)  # pretend an operator unblocked it, to reach the code check
    streams, anns = _cohort(1, 1)
    with pytest.raises(W.PilotInputError) as exc:
        W.score_pilot(Token(), "research_noncommercial", streams, anns, n_draws=10)
    assert exc.value.codes == ["synthetic_participant_codes_under_non_synthetic_adapter"]
    real_streams = {REAL_LOOKING: syn.generate("calm", duration=9.0)}
    r = W.score_pilot(Token(), "research_noncommercial", real_streams, [rec(ANN1, REAL_LOOKING, 0.0), rec(ANN2, REAL_LOOKING, 0.0)], n_draws=10)
    assert r["banner"] == "PILOT-DRAFT"  # never the SYNTHETIC banner for real-looking participants


def test_p2_scored_data_is_hashed_and_bound_to_the_manifest():
    streams, anns = _cohort(2, 1)
    a = W.score_pilot(SyntheticAdapter(), "synthetic", streams, anns, n_draws=10)
    b = W.score_pilot(SyntheticAdapter(), "synthetic", *_cohort(2, 1), n_draws=10)
    assert a["data_sha256"] == b["data_sha256"] and re.fullmatch(r"[0-9a-f]{64}", a["data_sha256"])
    assert a["bound_manifest_sha256"] == W.sha256_of({"manifest": a["manifest"]["manifest_sha256"], "data": a["data_sha256"]})
    other_streams, other_anns = _cohort(2, 2)
    assert W.score_pilot(SyntheticAdapter(), "synthetic", other_streams, other_anns, n_draws=10)["data_sha256"] != a["data_sha256"]
    changed = [dict(x) for x in anns]
    changed[0] = {**changed[0], "labels": {**changed[0]["labels"], "fidget": "present"}}
    assert W.score_pilot(SyntheticAdapter(), "synthetic", streams, changed, n_draws=10)["data_sha256"] != a["data_sha256"]
    perturbed = {p: list(v) for p, v in streams.items()}
    first = next(iter(perturbed))
    perturbed[first][0] = hb1.FrameRecord(**{**perturbed[first][0].__dict__, "quality": 0.5})
    assert W.score_pilot(SyntheticAdapter(), "synthetic", perturbed, anns, n_draws=10)["data_sha256"] != a["data_sha256"]
    md = W.render_pilot_markdown(W.build_pilot_report(a))
    assert a["data_sha256"] in md and a["bound_manifest_sha256"] in md


def test_p2_report_refuses_a_data_hash_that_is_not_bound(small):
    rep = W.build_pilot_report(small)
    for mutate in (lambda r: r["results"].update(data_sha256="0" * 64), lambda r: r["results"].pop("data_sha256"),
                   lambda r: r["results"].update(bound_manifest_sha256="f" * 64)):
        bad = copy.deepcopy(rep)
        mutate(bad)
        bad["report_sha256"] = W.sha256_of({k: v for k, v in bad.items() if k != "report_sha256"})  # self-rehashed
        with pytest.raises(eval_report.ReportIncomplete):
            W.render_pilot_markdown(bad)


def test_p2_the_limit_is_stated_in_the_docstring_and_the_verbatim_limitations():
    doc = " ".join(W.__doc__.split())
    assert "ADAPTER's declarations only" in doc and "does not by itself bind the data" in doc and "reserved synthetic form" in doc
    assert "Data enters only through" not in doc
    assert "not the data handed to the scorer" in W.PILOT_LIMITATIONS and "withheld below 20 positive windows" in W.PILOT_LIMITATIONS
    assert W.PILOT_LIMITATIONS.startswith(eval_report.ALLOWED_CLAIMS[0])
    assert "not the data handed to the scorer" in W.render_pilot_markdown(W.build_pilot_report(W.score_pilot(SyntheticAdapter(), "synthetic", *_cohort(1, 1), n_draws=10)))


# P3 ---------------------------------------------------------------------------------------------------------------------------

def test_p3_agreement_helpers_never_raise_on_bad_types():
    assert G.cohen_kappa([[1], [2]], [[1], [2]])["reason"] == "unhashable_values"
    assert G.cohen_kappa([{"a": 1}, {"b": 2}], [{"a": 1}, {"b": 2}])["kappa"] is None
    for bad in (None, 5, object()):
        out = G.label_agreement(bad, bad)
        assert out["kappa"] is None and out["kappa_reason"] == "not_sequences" and out["n"] == 0 and out["interpretation"] == "undefined"
    assert G.label_agreement([1, 2], [1])["kappa_reason"] == "length_mismatch"
    assert G.label_agreement([[1], [2]], [[1], [2]])["n"] == 2  # unhashable values: summary still returned
    assert G.label_agreement(["present"], 5)["kappa_reason"] == "not_sequences"
    json.dumps(G.label_agreement(None, None), allow_nan=False)


def test_p3_bootstrap_never_raises_on_bad_arguments_or_a_raising_statistic():
    for items in (None, 5, [], "abc"):
        out = G.bootstrap_by_participant(items, lambda x: 1.0)
        assert out["reason"] == "invalid_arguments" and out["lo"] is None
    assert G.bootstrap_by_participant({"a": [1]}, None)["reason"] == "invalid_arguments"
    assert G.bootstrap_by_participant({"a": [1]}, "nope")["reason"] == "invalid_arguments"
    assert G.bootstrap_by_participant({"a": 5, "b": [1]}, lambda x: 1.0)["reason"] == "invalid_arguments"
    raising = G.bootstrap_by_participant({"a": [1], "b": [2]}, lambda x: 1 / 0)
    assert raising["reason"] == "statistic_undefined_in_every_draw" and raising["estimate"] is None and raising["n_valid"] == 0
    for junk in ("x", float("nan"), float("inf"), [1], True):
        out = G.bootstrap_by_participant({"a": [1], "b": [2]}, lambda x, j=junk: j)
        assert out["estimate"] is None and out["lo"] is None
    flaky = iter([1.0] + [None] * 5 + [0.5] * 2000)
    assert G.bootstrap_by_participant({"a": [1], "b": [2]}, lambda x: next(flaky), 20)["n_valid"] < 20


@pytest.mark.parametrize("bad", [None, 5, 3.5, "abc", b"abc", {"a": 1}, object()])
def test_p3_validate_batch_raises_one_documented_error_type(bad):
    with pytest.raises(A.AnnotationError) as exc:
        A.validate_batch(bad)
    assert exc.value.codes in (["not_a_list"], ["not_a_dict"]) or exc.value.codes[0].startswith("not_a")


def test_p3_validate_batch_still_accepts_lists_tuples_and_generators():
    records = [rec(start=0.0), rec(start=3.0)]
    assert len(A.validate_batch(records)) == 2 and len(A.validate_batch(tuple(records))) == 2 and len(A.validate_batch(r for r in records)) == 2
    assert A.validate_batch([]) == []


@pytest.mark.parametrize("streams,anns,codes", [
    (None, "ok", ["streams_must_be_a_dict"]), ([], "ok", ["streams_must_be_a_dict"]), ("abc", "ok", ["streams_must_be_a_dict"]), (5, "ok", ["streams_must_be_a_dict"]),
    ({"x": 5}, "ok", ["each_stream_must_be_a_list"]), ({"x": None}, "ok", ["each_stream_must_be_a_list"]), ({"x": "abc"}, "ok", ["each_stream_must_be_a_list"]),
    ("ok", None, ["annotations_must_be_a_list"]), ("ok", "abc", ["annotations_must_be_a_list"]), ("ok", {"a": 1}, ["annotations_must_be_a_list"]),
    (None, None, ["streams_must_be_a_dict", "annotations_must_be_a_list"]),
])
def test_p3_score_pilot_raises_pilot_input_error_for_bad_containers(streams, anns, codes):
    good_streams, good_anns = _cohort(1, 1)
    with pytest.raises(W.PilotInputError) as exc:
        W.score_pilot(SyntheticAdapter(), "synthetic", good_streams if streams == "ok" else streams, good_anns if anns == "ok" else anns, n_draws=5)
    assert exc.value.codes == codes and isinstance(exc.value, ValueError)


def test_p3_score_pilot_other_bad_inputs_use_documented_errors_only():
    streams, anns = _cohort(1, 1)
    for bad_anns in (5, 3.5):
        with pytest.raises((W.PilotInputError, A.AnnotationError)):
            W.score_pilot(SyntheticAdapter(), "synthetic", streams, bad_anns, n_draws=5)
    with pytest.raises(A.AnnotationError):
        W.score_pilot(SyntheticAdapter(), "synthetic", streams, [None], n_draws=5)
    junk = dict(streams)
    junk[pt(1)] = [None, {"t": 1.0}, "x"] + list(streams[pt(1)])
    assert W.score_pilot(SyntheticAdapter(), "synthetic", junk, anns, n_draws=5)["counts"]["participants"] == 2  # junk items inside a stream are ignored
    assert W.score_pilot(SyntheticAdapter(), "synthetic", {}, anns, n_draws=5)["counts"]["participants_scored"] == 0
    with pytest.raises(TypeError):
        W.score_pilot("not an adapter", "synthetic", streams, anns)  # the adapter check keeps its own type error


# P4 / P6 ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [1.0, True, "1", None, 2, 0, [1]])
def test_p4_schema_version_must_be_the_int_one(bad):
    r = rec()
    r["schema_version"] = bad
    with pytest.raises(A.AnnotationError, match="bad_schema_version"):
        A.validate_annotation(r)
    assert A.validate_annotation(rec())["schema_version"] == 1 and type(A.validate_annotation(rec())["schema_version"]) is int


def test_p6_block_starts_are_not_forced_onto_a_grid_and_the_note_says_so():
    assert "NOT forced onto a 3 s grid" in A.__doc__ and "no enforcement" in A.__doc__
    off = A.validate_batch([rec(start=0.5), rec(start=3.5)])
    assert [o["window_start"] for o in off] == [0.5, 3.5]
    streams, anns = _cohort(1, 1)
    shifted = [{**a, "window_start": a["window_start"] + 0.5, "window_end": a["window_end"] + 0.5} for a in anns]
    r = W.score_pilot(SyntheticAdapter(), "synthetic", streams, shifted, n_draws=5)
    assert r["scoring"]["raw:pause"][ANN1]["ALL"]["n_windows"] > 0
    with pytest.raises(A.AnnotationError, match="overlapping_or_duplicate_windows"):
        A.validate_batch([rec(start=0.5), rec(start=3.0)])


# --- review S17: withheld raw values are never serialised in a report (EL-006 S17b) ----------------------------------------------

def test_r2_withheld_raw_agreement_values_never_appear_in_any_report_output(tmp_path):
    streams, anns = _cohort(1, 1)
    r = W.score_pilot(SyntheticAdapter(), "synthetic", streams, anns, n_draws=10)
    snapshot = json.dumps(r, sort_keys=True)
    assert r["agreement"]["pause"]["pair"]["raw_kappa_do_not_interpret"] == 1.0 and r["agreement"]["pause"]["alpha"]["raw_alpha_do_not_interpret"] == 1.0
    rep = W.build_pilot_report(r)
    text = json.dumps(rep, sort_keys=True)
    for needle in ("raw_kappa", "raw_alpha", "do_not_interpret"):
        assert needle not in text, needle
    assert needle not in W.render_pilot_markdown(rep)
    jp, mp = W.write_pilot_report(rep, str(tmp_path))
    for path in (jp, mp):
        body = pathlib.Path(path).read_text(encoding="utf-8")
        assert "raw_kappa" not in body and "raw_alpha" not in body and "do_not_interpret" not in body
    assert json.dumps(r, sort_keys=True) == snapshot  # building a report does not touch the caller's results
    pair = rep["results"]["agreement"]["pause"]["pair"]
    assert pair["kappa"] is None and pair["kappa_reason"] == "fewer_than_20_positive_windows" and pair["interpretation"].startswith("counts_only")
    alpha = rep["results"]["agreement"]["pause"]["alpha"]
    assert alpha["alpha"] is None and alpha["reason"] == "fewer_than_20_positive_windows"


def test_r2_the_report_keeps_everything_else_and_stays_valid(small):
    rep = W.build_pilot_report(small)
    W.validate_pilot_report(rep)
    assert rep["results"]["agreement"]["look_away"]["pair"]["kappa"] == 1.0  # numbers with 20+ positives are not withheld
    assert rep["results"]["data_sha256"] == small["data_sha256"] and rep["manifest_sha256"] == small["manifest"]["manifest_sha256"]
    assert rep == W.build_pilot_report(copy.deepcopy(small))


def test_r2_validation_refuses_a_report_that_carries_withheld_values_even_after_a_rehash(small):
    rep = W.build_pilot_report(small)
    for path in (("agreement", "fidget", "pair"), ("agreement", "pause", "alpha")):
        bad = copy.deepcopy(rep)
        node = bad["results"]
        for k in path:
            node = node[k]
        node["raw_kappa_do_not_interpret" if path[-1] == "pair" else "raw_alpha_do_not_interpret"] = 1.0
        bad["report_sha256"] = W.sha256_of({k: v for k, v in bad.items() if k != "report_sha256"})
        with pytest.raises(eval_report.ReportIncomplete, match="withheld raw agreement"):
            W.render_pilot_markdown(bad)
        with pytest.raises(eval_report.ReportIncomplete):
            W.validate_pilot_report(bad)
    nested = copy.deepcopy(rep)
    nested["results"]["notes"].append({"x": [{"raw_alpha_anything": 1}]})
    nested["report_sha256"] = W.sha256_of({k: v for k, v in nested.items() if k != "report_sha256"})
    with pytest.raises(eval_report.ReportIncomplete):
        W.validate_pilot_report(nested)


def test_r2_strip_helper_is_a_pure_deep_copy():
    src = {"a": {"raw_kappa_do_not_interpret": 1, "raw_kappa_reason": None, "kappa": None, "list": [{"raw_alpha_do_not_interpret": 2, "ok": 1}]}, "raw_alpha_reason": "x"}
    snap = copy.deepcopy(src)
    out = W._strip_withheld(src)
    assert src == snap and out == {"a": {"kappa": None, "list": [{"ok": 1}]}}
    assert W._has_withheld(src) and not W._has_withheld(out) and not W._has_withheld({"rawkappa": 1}) and not W._has_withheld(5)



# S56: review S53 manufactured correctness regressions.
def test_s56_empty_pilot_intervals_age_out():
    from dataclasses import replace
    rows = syn.generate("nervous-like", duration=3.125)
    rows.append(replace(rows[-1], t=10.0, expression=None, box=None, landmarks=None, face_count=0))
    windows = W._hb_windows(rows)
    assert any(names and not abstained for _, end, names, abstained in windows if end == 3)
    assert all(abstained and not names for _, end, names, abstained in windows if end >= 6)


def test_s56_explicit_endpoint_is_right_closed_and_retains_dwell():
    from dataclasses import replace
    rows = syn.generate("unsure-like", duration=8.0)
    end = 6.0
    result = hb1.analyze(rows, evaluation_time=end)
    expected = hb1.analyze([r for r in rows if r.t <= end])
    assert result == expected
    support = [r for r in rows if end - 3 < r.t <= end]
    assert result["window"]["usable_frames"] == len(support)
    assert "needs-help-like" in {s["name"] for s in result["states"]}
    stale = hb1.analyze(rows, evaluation_time=20.0)
    assert stale["insufficient"]["flag"] and not stale["states"]


def _s56_angles(alternate, offset=0.0):
    from dataclasses import replace
    out = []
    for i, row in enumerate(syn.generate("calm", duration=3.0)):
        angle = offset + (0.001 if alternate and i % 2 else -0.001)
        dx, dy = 30 * math.cos(angle), 30 * math.sin(angle)
        landmarks = ((320-dx, 200-dy), (320+dx, 200+dy), (320,235), (300,265), (340,265))
        out.append(replace(row, landmarks=landmarks, expression={"fear": .8, "neutral": .2}, arousal=.1))
    return out


def test_s56_roll_branch_cut_is_small_not_maximal_nervousness():
    rows = _s56_angles(True, math.pi)
    assert hb1.extract_features(rows)["head_jitter"] < .002
    assert hb1.analyze(rows)["states"] == []


@pytest.mark.parametrize("offset", [math.pi, 2*math.pi, -2*math.pi])
def test_s56_equivalent_roll_rotations(offset):
    reference, rotated = _s56_angles(True), _s56_angles(True, offset)
    assert hb1.extract_features(rotated)["head_jitter"] == pytest.approx(hb1.extract_features(reference)["head_jitter"], abs=1e-12)
    assert hb1.analyze(rotated)["states"] == hb1.analyze(reference)["states"]


@pytest.mark.parametrize("third", ["unscorable", "partial"])
def test_s56_third_rater_coverage_does_not_delete_first_pair(third):
    anns = [rec(annotator=f"ANN-{i:06x}") for i in (1, 2)]
    streams = {pt(1): syn.generate("calm", duration=4)}
    baseline = W.score_pilot(SyntheticAdapter(), "synthetic", streams, anns, n_draws=1)
    extra = rec(annotator="ANN-000003", unscorable=True) if third == "unscorable" else rec(annotator="ANN-000003", start=3.0)
    result = W.score_pilot(SyntheticAdapter(), "synthetic", streams, anns + [extra], n_draws=1)
    for label in A.LABELS:
        assert result["agreement"][label]["pair"] == baseline["agreement"][label]["pair"]
        assert result["agreement"][label]["alpha"]["n_pairable"] == 2


@pytest.mark.parametrize("fps", [8, 9, 13, 16, 20])
def test_s56_ring_keeps_needs_help_reachable_and_five_second_boundary(fps):
    from socialcue_session import SocialCueSessions
    from dataclasses import replace
    rows = syn.generate("needs-help-like", fps=fps)
    store = SocialCueSessions(clock=lambda: 0)
    for row in rows:
        retained = store.add("s"*22, lambda now, row=row: row)
    full, ring = hb1.analyze(rows), hb1.analyze(retained)
    assert [s["name"] for s in ring["states"]] == [s["name"] for s in full["states"]]
    assert not ring["insufficient"]["flag"]
    assert len(retained) <= 64
    assert hb1.extract_features(retained)["dwell"] > 5.0
    # Uniform manufactured presence from zero includes the exact t=5 record.
    rows = syn.generate("unsure-like", fps=fps, duration=6.0)
    store = SocialCueSessions(clock=lambda: 0)
    for row in rows:
        retained = store.add("s"*22, lambda now, row=row: row)
        if row.t >= 5.0:
            full = hb1.analyze([r for r in rows if r.t <= row.t])
            ring = hb1.analyze(retained)
            full_help = "needs-help-like" in {s["name"] for s in full["states"]}
            ring_help = "needs-help-like" in {s["name"] for s in ring["states"]}
            # A retained bin representative may lag an arrival by <125 ms;
            # compare the exact boundary and the settled post-boundary result.
            if row.t == 5.0 or row.t >= 5.25:
                assert ring_help == full_help
            if row.t == 5.0: assert not full_help
            assert not ring_help or full_help
            assert row.t - retained[-1].t < .125 + 1e-9
        bins = [math.floor((r.t-retained.bin_origin)*8 + 1e-9) for r in retained]
        assert len(bins) == len(set(bins))


def test_s56_default_analysis_matches_head_on_444_case_grid():
    import types
    source = subprocess.run(["git", "show", "8863640:socialcue/state.py"], capture_output=True, text=True, encoding="utf-8")
    if source.returncode: pytest.skip("S56 baseline oracle requires git checkout with 8863640 history")
    old = types.ModuleType("s56_baseline_state")
    sys.modules[old.__name__] = old
    try:
        exec(compile(source.stdout, "8863640/socialcue/state.py", "exec"), old.__dict__)
        cases = [(scenario, strength, seed, prefix) for scenario in syn.SCENARIOS
                 for strength in (0.0, .5, 1.0) for seed in (0, 505, 991)
                 for prefix in (1, 8, 16, 24, 32, 48, None)]
        cases += [("needs-help-like", 1.0, 505, prefix) for prefix in (40, 41, 42)]
        assert len(cases) == 444
        for scenario, strength, seed, prefix in cases:
            rows = syn.generate(scenario, strength=strength, seed=seed)
            rows = rows if prefix is None else rows[:prefix]
            expected = old.analyze([old.FrameRecord(**vars(r)) for r in rows])
            assert hb1.analyze(rows) == hb1.analyze(rows, evaluation_time=None) == expected, (scenario, strength, seed, prefix)
        assert hb1.CONFIG_SHA256 == old.CONFIG_SHA256 == "4940a4cbbb1df3c2af27407e78546e47812c5134b8fcc2aec0daad9e2874b6b3"
    finally:
        sys.modules.pop(old.__name__, None)


def test_s56_alpha_keeps_shared_units_with_missing_third_ratings(monkeypatch):
    annotations = [rec(annotator=annotator, participant=pt(p), start=start,
                       labels={"fidget": "present" if start == 0 else "absent"})
                   for p in (1, 2) for start in (0.0, 3.0) for annotator in (ANN1, ANN2)]
    streams = {pt(p): syn.generate("calm", duration=6) for p in (1, 2)}
    baseline = W.score_pilot(SyntheticAdapter(), "synthetic", streams, annotations, n_draws=2)
    extra = [rec(annotator="ANN-0000c3", participant=pt(p), labels={"fidget": "present"}) for p in (1, 2)]
    extra.append(rec(annotator="ANN-0000c3", participant=pt(3)))  # unpairable third-only participant
    units_seen = []
    original = G.krippendorff_alpha_nominal
    def inspect(units):
        units_seen.extend(units)
        return original(units)
    monkeypatch.setattr(G, "krippendorff_alpha_nominal", inspect)
    result = W.score_pilot(SyntheticAdapter(), "synthetic", streams, annotations + extra, n_draws=2)
    for label in A.LABELS:
        assert result["agreement"][label]["pair"] == baseline["agreement"][label]["pair"]
        assert result["agreement"][label]["kappa_ci_by_participant"] == baseline["agreement"][label]["kappa_ci_by_participant"]
        assert result["agreement"][label]["alpha"]["n_pairable"] == 10
    assert any(len(unit) == 3 and unit[-1] is None for unit in units_seen)
    assert any(len(unit) == 3 and unit[-1] is not None for unit in units_seen)


def test_s56_same_bin_bad_evidence_is_not_hidden_by_later_good_frame():
    from dataclasses import replace
    from socialcue_session import SocialCueSessions
    store = SocialCueSessions(clock=lambda: 0)
    base = syn.generate("calm")[0]
    store.add("s"*22, lambda now: base)
    store.add("s"*22, lambda now: replace(base, t=.01, face_count=2))
    retained = store.add("s"*22, lambda now: replace(base, t=.02))
    assert len(retained) == 1 and retained[0].face_count == 2
    assert "multiple_faces" in hb1.analyze(retained)["insufficient"]["reasons"]


# --- S56b takeover additions (review S53; manufactured data only) --------------------------------------------------------------

def _unusable(row, t):
    from dataclasses import replace
    return replace(row, t=t, expression=None, box=None, landmarks=None, face_count=0)


def test_s56b_same_bin_unusable_frame_does_not_erase_the_usable_credit():
    """Failing-first repro: a no-face tick in the same 125 ms bin used to replace the usable record, so the ring said no_face."""
    from dataclasses import replace
    from socialcue_session import SocialCueSessions
    base = syn.generate("needs-help-like", fps=8.0)
    rows = []
    for r in base:
        rows += [replace(r, t=r.t + 0.01), _unusable(r, r.t + 0.07)]  # both inside one 125 ms bin
    store = SocialCueSessions(clock=lambda: 0)
    for row in rows:
        retained = store.add("s" * 22, lambda now, row=row: row)
    full, ring = hb1.analyze(rows), hb1.analyze(retained)
    assert not full["insufficient"]["flag"] and "needs-help-like" in {x["name"] for x in full["states"]}
    assert not ring["insufficient"]["flag"] and [x["name"] for x in ring["states"]] == [x["name"] for x in full["states"]]
    assert all(hb1._usable(r) for r in retained) and len(retained) <= 64


def test_s56b_bin_representative_priority_is_multiface_then_usable_then_earliest():
    from dataclasses import replace
    from socialcue_session import SocialCueSessions
    base = syn.generate("calm")[0]
    store = SocialCueSessions(clock=lambda: 0)
    store.add("s" * 22, lambda now: _unusable(base, 0.0))
    kept = store.add("s" * 22, lambda now: replace(base, t=0.05))
    assert len(kept) == 1 and hb1._usable(kept[0]) and kept[0].t == 0.05  # usable beats an earlier unusable tick
    kept = store.add("s" * 22, lambda now: replace(base, t=0.06))
    assert len(kept) == 1 and kept[0].t == 0.05  # equal rank: the earliest stays
    kept = store.add("s" * 22, lambda now: replace(base, t=0.10, face_count=3))
    assert len(kept) == 1 and kept[0].face_count == 3  # multi-face evidence is never hidden
    kept = store.add("s" * 22, lambda now: replace(base, t=0.11))
    assert kept[0].face_count == 3
    assert "multiple_faces" in hb1.analyze(kept)["insufficient"]["reasons"]


@pytest.mark.parametrize("fps", [30, 60, 240])
def test_s56b_high_request_rates_stay_bounded_one_record_per_bin(fps):
    from socialcue_session import SocialCueSessions
    rows = syn.generate("unsure-like", fps=fps, duration=12.0)
    store = SocialCueSessions(clock=lambda: 0)
    peak = 0
    for row in rows:
        retained = store.add("s" * 22, lambda now, row=row: row)
        peak = max(peak, len(retained))
    bins = [math.floor((r.t - retained.bin_origin) * 8 + 1e-9) for r in retained]
    assert peak <= 64 and len(bins) == len(set(bins))
    assert retained[-1].t - retained[0].t < 8.0 + 1e-9  # at most the latest eight seconds
    assert hb1.extract_features(retained)["dwell"] > 5.0 and not hb1.analyze(retained)["insufficient"]["flag"]
    if fps <= 60:  # at extreme rates the synthetic label cycle (a function of the frame index) aliases with 1-in-N binning
        full, ring = hb1.analyze(rows), hb1.analyze(retained)
        assert [x["name"] for x in ring["states"]] == [x["name"] for x in full["states"]]


def test_s56b_low_request_rate_history_is_trimmed_to_eight_seconds_not_sixty_four_frames():
    from socialcue_session import SocialCueSessions
    store = SocialCueSessions(clock=lambda: 0)
    base = syn.generate("calm")[0]
    from dataclasses import replace
    for k in range(40):  # 2 fps for 20 s
        retained = store.add("s" * 22, lambda now, k=k: replace(base, t=k * 0.5))
    assert len(retained) <= 17 and retained[-1].t - retained[0].t < 8.0 and retained[-1].t == 19.5


def test_s56b_endpoint_convention_is_left_open_right_closed_and_independent_of_later_records():
    from dataclasses import replace
    rows = syn.generate("nervous-like", duration=6.0)
    for end in (3.0, 4.0, 4.5):
        later = [r for r in rows if r.t <= end + 1.0]  # records after the endpoint must not matter
        assert hb1.analyze(later, evaluation_time=end) == hb1.analyze([r for r in rows if r.t <= end], evaluation_time=end)
    # a record exactly W before the endpoint is outside (t-W, t]; one exactly at the endpoint is inside
    probe = [replace(rows[0], t=0.0), replace(rows[1], t=3.0)]
    res = hb1.analyze(probe, evaluation_time=3.0)
    assert res["window"]["usable_frames"] == 1  # only the record AT the endpoint; the one exactly W earlier is excluded
    assert hb1.analyze(probe, evaluation_time=2.999)["window"]["usable_frames"] == 1  # only the record at 0.0 (3.0 is later than the endpoint)
    assert hb1.analyze(probe, evaluation_time=0.0)["window"]["usable_frames"] == 1  # the record at the endpoint 0.0 is included
    assert hb1.analyze(probe, evaluation_time=3.0001)["window"]["usable_frames"] == 1  # 0.0 has aged out (3.0001 - 3 > 0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), "3", True, [3.0], object()])
def test_s56b_non_finite_or_non_numeric_evaluation_time_is_refused_loudly(bad):
    with pytest.raises(ValueError):
        hb1.analyze(syn.generate("calm"), evaluation_time=bad)
    assert hb1.analyze(syn.generate("calm"), evaluation_time=None) == hb1.analyze(syn.generate("calm"))


def test_s56b_long_gap_windows_abstain_inside_the_gap_and_recover_after_it():
    from dataclasses import replace
    first = syn.generate("unsure-like", duration=4.0)
    second = [replace(r, t=r.t + 12.0) for r in syn.generate("unsure-like", duration=4.0)]
    windows = {round(t1): (names, abstained) for _, t1, names, abstained in W._hb_windows(first + second)}
    assert not windows[3][1] and "unsure-like" in windows[3][0]
    for gap_end in (7, 8, 9, 10, 11):  # (4,12) holds no record: the old 3 s evidence has aged out
        assert windows[gap_end][1] and not windows[gap_end][0], gap_end
    assert windows[15][1] is False and "unsure-like" in windows[15][0]  # support rebuilt after the gap


def test_s56b_sparse_stream_one_record_per_window_does_not_borrow_stale_support():
    from dataclasses import replace
    base = syn.generate("nervous-like", duration=1.0)[0]
    rows = [replace(base, t=float(k) * 2.5) for k in range(6)]  # one record every 2.5 s
    windows = W._hb_windows(rows)
    assert windows and all(abstained and not names for _, _, names, abstained in windows)


def test_s56b_roll_unwrap_more_cases():
    def feat(angles):
        out = []
        for row, a in zip(syn.generate("calm", duration=3.0), angles):
            dx, dy = 30 * math.cos(a), 30 * math.sin(a)
            from dataclasses import replace
            out.append(replace(row, landmarks=((320 - dx, 200 - dy), (320 + dx, 200 + dy), (320, 235), (300, 265), (340, 265)),
                               expression={"fear": .8, "neutral": .2}, arousal=.1))
        return hb1.extract_features(out)["head_jitter"]
    n = len(syn.generate("calm", duration=3.0))
    cut = [math.pi - 0.002 * (i % 5) if i % 3 else -math.pi + 0.002 * (i % 4) for i in range(n)]  # clustered around the branch cut
    assert feat(cut) < 0.01
    outlier_first = [math.pi - 0.001] + [-math.pi + 0.001 * (i % 3) for i in range(1, n)]
    assert feat(outlier_first) < 0.01
    assert feat([0.0] * n) == 0.0
    # the eye-line is DIRECTED (eye_R -> eye_L): samples that really alternate between opposite orientations stay large
    flipping = [0.0 if i % 2 else math.pi for i in range(n)]
    assert feat(flipping) > 1.0
    assert "directed" in (hb1._roll_sd.__doc__ or "")


def test_s56b_third_rater_on_other_participants_does_not_touch_the_first_pair():
    anns = [rec(annotator=a, participant=pt(p), start=s, labels={"pause": "present" if s == 0 else "absent"})
            for p in (1, 2) for s in (0.0, 3.0) for a in (ANN1, ANN2)]
    streams = {pt(p): syn.generate("calm", duration=6) for p in (1, 2, 3)}
    base = W.score_pilot(SyntheticAdapter(), "synthetic", {k: streams[k] for k in (pt(1), pt(2))}, anns, n_draws=3)
    extra = [rec(annotator="ANN-0000c3", participant=pt(3), labels={"pause": "present"}), rec(annotator="ANN-0000c3", participant=pt(1), start=3.0)]
    out = W.score_pilot(SyntheticAdapter(), "synthetic", streams, anns + extra, n_draws=3)
    for lab in A.LABELS:
        assert out["agreement"][lab]["pair"] == base["agreement"][lab]["pair"]
        assert out["agreement"][lab]["kappa_ci_by_participant"] == base["agreement"][lab]["kappa_ci_by_participant"]
    json.dumps(out, allow_nan=False)


# --- S58: bounded pilot window scheduling (review S57 N1) -------------------------------------------------------------------------

def _dense_reference(records, cfg=None):
    """The pre-S58 scheduler, verbatim: one endpoint per hop from first+W to the last record."""
    recs = sorted((r for r in records if isinstance(r, hb1.FrameRecord) and isinstance(r.t, (int, float)) and math.isfinite(r.t)), key=lambda r: r.t)
    if not recs:
        return []
    out, t1, last = [], recs[0].t + W.HB_WINDOW, recs[-1].t
    while t1 <= last + 1e-9:
        res = hb1.analyze([r for r in recs if r.t <= t1], cfg, evaluation_time=t1)
        out.append((t1 - W.HB_WINDOW, t1, frozenset(x["name"] for x in res["states"]), bool(res["insufficient"]["flag"])))
        t1 += W.HB_HOP
    return out


def _timed(fn, limit):
    """Run fn in a daemon thread; (finished, seconds, value). A runaway scheduler fails the test instead of hanging it."""
    import threading
    import time
    box = {}
    t0 = time.monotonic()
    th = threading.Thread(target=lambda: box.setdefault("v", fn()), daemon=True)
    th.start()
    th.join(limit)
    return (not th.is_alive()), time.monotonic() - t0, box.get("v")


def test_s58_one_far_future_record_finishes_in_well_under_a_second():
    from dataclasses import replace
    rows = syn.generate("nervous-like", duration=3.125)
    rows.append(replace(rows[-1], t=1_000_000.0))
    done, secs, out = _timed(lambda: W._hb_windows(rows), 5.0)
    assert done and secs < 1.0, secs  # before S58: 999,998 windows, about 85 s
    assert out is not None and len(out) < 50
    ends = [w[1] for w in out]
    assert ends == sorted(ends) and len(set(ends)) == len(ends)
    assert any(e <= 7.0 for e in ends)  # the real segment keeps its evidence-supported windows
    assert all(e <= 1_000_000.0 + 1e-9 for e in ends)


def test_s58_sparse_rule_keeps_evidence_windows_and_annotated_reach_and_marks_the_rest_absent():
    from dataclasses import replace
    rows = syn.generate("unsure-like", duration=4.0) + [replace(r, t=r.t + 50_000.0) for r in syn.generate("unsure-like", duration=4.0)]
    blocks = [A.validate_annotation(rec(start=1000.0)), A.validate_annotation(rec(start=3.0))]
    done, secs, out = _timed(lambda: W._hb_windows(rows, None, blocks), 5.0)
    assert done and secs < 1.0 and out
    by_end = {round(w[1], 6): w for w in out}
    for k in range(3, 8):  # evidence in the first segment
        assert k in by_end, k
    for reach in (1001.5, 1002.5, 1003.5, 1004.5):  # the block at 1000 s sits inside the gap: scheduled, abstains
        assert any(abs(e - reach) < 1.0 for e in by_end), reach
    gap = [w for w in out if 1000.0 <= w[1] <= 1005.0]
    assert gap and all(w[3] and not w[2] for w in gap)
    assert any(e > 50_000.0 for e in by_end)  # the second segment
    assert not any(8.5 < e < 999.0 for e in by_end)  # nothing between: no evidence and no block there


def test_s58_ordinary_streams_are_byte_identical_to_the_dense_scheduler():
    cases = [(sc, st, seed, pre) for sc in syn.SCENARIOS for st in (0.0, 1.0) for seed in (0, 505) for pre in (24, 48, None)]
    for sc, st, seed, pre in cases:
        rows = syn.generate(sc, strength=st, seed=seed)
        rows = rows if pre is None else rows[:pre]
        ref = _dense_reference(rows)
        assert W._hb_windows(rows) == ref, (sc, st, seed, pre)
        assert W._hb_windows(rows, None, [A.validate_annotation(rec(start=0.0))]) == ref  # blocks never change the dense path
    assert W._hb_windows([]) == [] and _dense_reference([]) == []


def test_s58_scoring_results_are_identical_with_the_dense_scheduler(monkeypatch):
    for n_unsure, n_calm in ((2, 2), (1, 1)):
        streams, anns = _cohort(n_unsure, n_calm)
        new = W.score_pilot(SyntheticAdapter(), "synthetic", streams, anns, n_draws=10)
        monkeypatch.setattr(W, "_hb_windows", lambda records, cfg=None, blocks=None: _dense_reference(records, cfg))
        old = W.score_pilot(SyntheticAdapter(), "synthetic", streams, anns, n_draws=10)
        monkeypatch.undo()
        assert json.dumps(new, sort_keys=True) == json.dumps(old, sort_keys=True)
        assert sha256_of_json(new) == sha256_of_json(old)


def sha256_of_json(obj):
    import hashlib
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("ascii")).hexdigest()


def test_s58_sparse_output_is_exactly_the_dense_windows_that_can_matter(monkeypatch):
    from dataclasses import replace
    rows = syn.generate("nervous-like", duration=4.0) + [replace(r, t=r.t + 30.0) for r in syn.generate("unsure-like", duration=4.0)]
    blocks = [A.validate_annotation(rec(start=s0)) for s0 in (0.0, 3.0, 12.0, 30.0, 33.0)]
    dense = _dense_reference(rows)
    monkeypatch.setattr(W, "MAX_DENSE_ENDPOINTS", 5)  # force the sparse rule on a small stream
    sparse = W._hb_windows(rows, None, blocks)
    assert len(dense) > 5 and len(sparse) < len(dense)
    recs_t = [r.t for r in rows]
    mattering = [w for w in dense
                 if any(w[0] < t <= w[1] for t in recs_t)
                 or any(b["window_start"] + 1.5 - 1e-9 <= w[1] <= b["window_end"] + 1.5 + 1e-9 for b in blocks)]
    assert [(w[0], w[1], w[2], w[3]) for w in sparse] == [(w[0], w[1], w[2], w[3]) for w in mattering]
    # scoring uses only windows that align to a block; dropping the rest cannot change an alignment
    assert W.align(sparse, blocks) == W.align(dense, blocks)


def test_s58_the_scheduling_rule_is_documented():
    doc = " ".join(W._hb_windows.__doc__.split())
    assert "MAX_DENSE_ENDPOINTS" in doc and "evidence" in doc and "annotated" in doc and "bounded" in doc
    assert W.MAX_DENSE_ENDPOINTS >= 10_000


# --- S60 (review S59 finding 10): label_agreement is binary-only --------------------------------------------------------------------

def test_s60_f10_third_category_is_refused_not_folded_into_not_present():
    a = ["present"] * 10 + ["absent"] * 2
    b = ["present"] * 10 + ["unsure"] * 2
    out = G.label_agreement(a, b)
    assert out["kappa"] is None and out["kappa_reason"] == "binary_labels_only" and out["interpretation"] == "undefined"
    assert out["observed_agreement"] is None and out["pabak"] is None and out["table"]["both_absent"] == 0  # before: 1.0, 1.0, table both_absent 2
    assert out["n"] == 12 and "binary" in G.label_agreement.__doc__.lower()
    assert G.cohen_kappa(a, b)["kappa"] == pytest.approx(0.4545, abs=1e-3)  # the raw kappa function still treats every category
    two = G.label_agreement(["present", "absent"] * 10, ["present", "absent"] * 10)
    assert two["observed_agreement"] == 1.0 and two["kappa_reason"] != "binary_labels_only"
    assert G.label_agreement(["present"] * 12, ["present"] * 12)["kappa_reason"] != "binary_labels_only"
    assert G.label_agreement([1, 2, 3], [1, 2, 3])["kappa_reason"] == "binary_labels_only"


def test_s62_agreement_categories_come_from_complete_pairs_only():
    out = G.label_agreement(["present", "absent", "unsure"], [ "present", "absent", None])  # review S61 finding C
    assert out["kappa_reason"] != "binary_labels_only" and out["n"] == 2 and out["observed_agreement"] == 1.0
    assert G.label_agreement(["present", "absent", "unsure"], ["present", "absent", "absent"])["kappa_reason"] == "binary_labels_only"


@pytest.mark.parametrize("a,b,reason", [
    (["absent", "unsure"], ["unsure", "absent"], "binary_labels_only"),
    ([[0], [1]], [[1], [0]], "unhashable_values")])
def test_s65_binary_summary_refuses_ambiguous_or_unhashable_categories(a, b, reason):
    out = G.label_agreement(a, b)
    assert out["kappa_reason"] == reason
    assert out["n"] == 2 and out["observed_agreement"] is None and out["pabak"] is None
    assert out["table"]["both_absent"] == 0 and out["raw_kappa_do_not_interpret"] is None


def test_s65_roll_broad_spread_is_order_independent_and_continuous_at_review_boundary():
    import itertools
    orderings = ([0, 1, math.pi], [-1e-9, 1, math.pi], [math.pi, 1, -1e-9], [1, -1e-9, math.pi])
    values = [hb1._roll_sd(xs) for xs in orderings]
    assert max(values) - min(values) < 1e-8
    assert all(v == pytest.approx(values[0], abs=1e-8) for v in values)
    for broad in ([-1e-9, 1, math.pi], [-3., -.5, .8, 2.7]):
        assert len({hb1._roll_sd(xs) for xs in itertools.permutations(broad)}) == 1
    assert hb1._roll_sd([0, math.pi]) == math.pi / 2


def test_s65_ordinary_analyze_and_features_have_identical_444_case_hashes():
    import types, hashlib
    source = subprocess.run(["git", "show", "a2a7960:socialcue/state.py"], capture_output=True, text=True, encoding="utf-8")
    if source.returncode: pytest.skip("S65 compatibility oracle needs git checkout with a2a7960 history")
    old = types.ModuleType("s65_baseline_state"); sys.modules[old.__name__] = old
    try:
        exec(compile(source.stdout, "a2a7960/socialcue/state.py", "exec"), old.__dict__)
        cases = [(scenario, strength, seed, prefix) for scenario in syn.SCENARIOS
                 for strength in (0., .5, 1.) for seed in (0, 505, 991)
                 for prefix in (1, 8, 16, 24, 32, 48, None)]
        cases += [("needs-help-like", 1., 505, prefix) for prefix in (40, 41, 42)]
        assert len(cases) == 444
        expected, actual = [], []
        for scenario, strength, seed, prefix in cases:
            rows = syn.generate(scenario, strength=strength, seed=seed)
            rows = rows if prefix is None else rows[:prefix]
            old_rows = [old.FrameRecord(**vars(row)) for row in rows]
            expected.append({"analyze": old.analyze(old_rows), "features": old.extract_features(old_rows)})
            actual.append({"analyze": hb1.analyze(rows), "features": hb1.extract_features(rows)})
        dumps = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        old_hash, new_hash = (hashlib.sha256(dumps(value)).hexdigest() for value in (expected, actual))
        print("S65_ANALYZE_FEATURES_444_OLD_SHA256=" + old_hash)
        print("S65_ANALYZE_FEATURES_444_NEW_SHA256=" + new_hash)
        assert old_hash == new_hash and dumps(actual) == dumps(expected)
    finally:
        sys.modules.pop(old.__name__, None)
