"""synthetic_only=true; SYNTHETIC / NOT REAL MODEL PERFORMANCE; seed 505.

Reuses the existing perfect/FN case from fe33b1b. New tests add genuinely missing
all-missing/surprise semantics, a hand confusion, unequal actor clusters, nonzero
paired differences and an independent Python-count oracle over all 1000 draws.
"""
from collections import Counter
from types import SimpleNamespace
from pathlib import Path
import hashlib
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
from emolab.benchmark import runner
from test_cremad_development import test_fixed_metrics_perfect_missing_surprise

LABELS=("angry","disgust","fear","happy","sad","surprise","neutral")
TRUTHS=("angry","disgust","fear","happy","sad","neutral")


def item(identity,actor,truth,prediction,status="no_face"):
    return {"id":identity,"actor":actor,"label":truth,"status":status if prediction is None else "predicted",
            "probabilities":None if prediction is None else [float(label==prediction) for label in LABELS]}


def test_hand_derived_confusion_and_denominators():
    rows=[item("a",1,"angry","angry"),item("b",1,"angry","happy"),item("c",2,"angry",None),
          item("d",2,"happy","happy"),item("e",3,"happy","angry")]
    result=runner.development_metrics(rows)
    expected=[[1,0,0,1,0,0,0,1],[0]*8,[0]*8,[1,0,0,1,0,0,0,0],[0]*8,[0]*8]
    assert result["confusion"]==expected and result["n"]==5
    assert result["accuracy"]==pytest.approx(2/5) and result["coverage"]==pytest.approx(4/5)
    assert result["macro_f1"]==pytest.approx(.15)
    assert result["per_class"]["angry"]==pytest.approx({"support":3,"precision":.5,"recall":1/3,"f1":.4})
    assert result["per_class"]["happy"]["f1"]==.5
    assert result["per_class"]["fear"]["recall"] is None
    assert result["per_class"]["fear"]["f1"]==0
    assert runner._duration_ms(0.,.1)==100.


@pytest.mark.parametrize("status",["no_face","decode_failure"])
def test_all_missing_retains_every_truth_and_has_zero_coverage(status):
    result=runner.development_metrics([item(str(i),1,label,None,status) for i,label in enumerate(TRUTHS)])
    assert result["coverage"]==result["accuracy"]==result["macro_f1"]==0
    assert [row[-1] for row in result["confusion"]]==[1]*6
    assert all(result["per_class"][label]["support"]==1 and result["per_class"][label]["recall"]==0 for label in TRUTHS)


def test_surprise_remains_wrong_prediction_with_full_coverage():
    result=runner.development_metrics([item("s",1,"angry","surprise")])
    assert result["confusion"][0][5]==1 and result["confusion"][0][7]==0
    assert result["coverage"]==1 and result["accuracy"]==result["macro_f1"]==0
    assert result["per_class"]["surprise"]=={"support":0,"precision":0.,"recall":None,"f1":0.}
    with pytest.raises(runner.BenchmarkContractError,match="missing prediction status"):
        runner.development_metrics([item("failed",1,"angry",None,"failed")])


def test_duplicate_manifest_ids_rejected_before_any_prediction():
    # Uniqueness belongs to the execution boundary, not the pure count function:
    # metrics must allow duplicate rows internally to preserve bootstrap multiplicity.
    row={"id":"duplicate","actor":1,"label":"angry"}
    fake=SimpleNamespace(name="mini-xception")
    def forbidden(*args):raise AssertionError("duplicate input reached execution")
    with pytest.raises(runner.BenchmarkContractError,match="unique clip manifest"):
        runner.run_development_clips([fake],{"rows":[row,row],"protocol_sha256":"synthetic"},forbidden)


@pytest.mark.parametrize("field,value",[("id","different"),("actor",99),("label","happy")])
def test_paired_cohort_identity_mismatch_rejected(field,value):
    first=[item("a",1,"angry","angry")];second=[{**first[0],field:value}]
    with pytest.raises(runner.BenchmarkContractError,match="paired cohort mismatch"):
        runner.development_bootstrap({"A":first,"B":second},seed=505)


def slow_count_oracle(rows):
    """Independent named outcome counts; no production metrics/confusion helpers."""
    outcomes=[]
    for row in rows:
        vector=row["probabilities"]
        prediction=None if vector is None else LABELS[max(range(7),key=lambda i:vector[i])]
        outcomes.append((row["label"],prediction))
    scores=[];recalls={}
    for label in TRUTHS:
        actual=sum(truth==label for truth,pred in outcomes)
        predicted=sum(pred==label for truth,pred in outcomes)
        correct=sum(truth==label and pred==label for truth,pred in outcomes)
        scores.append(2*correct/(actual+predicted) if actual+predicted else 0.)
        recalls[label]=correct/actual if actual else None
    return sum(scores)/6,recalls


def test_all_1000_unequal_cluster_draws_against_independent_slow_oracle():
    # One actor dominates clip count; actor draws must still be equally likely.
    first=[item(f"large-{i}",1,"angry","angry") for i in range(9)]
    first += [item("small",2,"happy","angry"),item("missing",3,"fear",None)]
    second=[{**row,"probabilities":[float(label==row["label"]) for label in LABELS],"status":"predicted"} for row in first]
    assert runner.development_metrics(first)["accuracy"]==pytest.approx(9/11)
    draws=np.random.default_rng(505).choice([1,2,3],size=(1000,3),replace=True)
    actual=runner.development_bootstrap({"A":first,"B":second},seed=505)
    assert actual["replicates"]==1000 and actual["draw_shape"]==[1000,3]
    assert actual["draws_sha256"]==hashlib.sha256(draws.tobytes()).hexdigest()
    assert any(Counter(draw)[1]==3 for draw in draws) # all nine clips repeated three times
    oracle={"A":[],"B":[]}
    for draw in draws:
        for name,rows in (("A",first),("B",second)):
            repeated=[row for actor in draw for row in rows if row["actor"]==actor]
            assert len(repeated)==sum(9 if actor==1 else 1 for actor in draw)
            oracle[name].append(slow_count_oracle(repeated))
    for name,values in oracle.items():
        assert actual["macro_f1_ci"][name]==pytest.approx(np.percentile([v[0] for v in values],[2.5,97.5]))
        for label in TRUTHS:
            recalls=[v[1][label] for v in values];defined=[v for v in recalls if v is not None]
            assert actual["undefined_recall_draws"][name][label]==recalls.count(None)
            if defined:assert actual["recall_ci"][name][label]==pytest.approx(np.percentile(defined,[2.5,97.5]))
            else:assert actual["recall_ci"][name][label] is None
    deltas=[a[0]-b[0] for a,b in zip(oracle["A"],oracle["B"])]
    assert min(deltas)<0
    assert actual["paired_macro_f1_difference_ci"]["A minus B"]==pytest.approx(np.percentile(deltas,[2.5,97.5]))


def test_one_actor_bootstrap_has_no_artificial_independent_clip_variation():
    rows=[item("a",7,"angry","angry"),item("b",7,"happy",None)]
    result=runner.development_bootstrap({"A":rows},seed=505)
    assert result["actor_count"]==1 and result["replicates"]==1000
    assert result["macro_f1_ci"]["A"]==pytest.approx([1/6,1/6])
    assert result["undefined_recall_draws"]["A"]["fear"]==1000
