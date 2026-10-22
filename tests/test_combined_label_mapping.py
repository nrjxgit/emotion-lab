"""Synthetic named-logit checks through the explicit unauthenticated harness."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
from emolab.adapters.claude_combined import ClaudeCombinedAdapter,FER7,CombinedContractError,BlockedAssetError
from emolab.benchmark.runner import run_benchmark
from test_combined_asset_contract import make_bundle,forbid_runtimes_and_network

NATIVE=["anger","disgust","fear","happiness","neutral","sadness","surprise"]
CANONICAL=["angry","disgust","fear","happy","neutral","sad","surprise"]
BY_NAME=dict(zip(NATIVE,CANONICAL))


def make(tmp_path,monkeypatch,*,classes=None,logits=None,spec_changes=None):
    bundle=make_bundle(tmp_path,classes=classes,logits=logits)
    if spec_changes:
        record=bundle.manifest["sidecar"]
        spec=json.loads((tmp_path/record["path"]).read_bytes());spec.update(spec_changes)
        bundle.rewrite("sidecar",spec)
    model=bundle.build()
    return model,tmp_path/"sidecar.json"


@pytest.mark.parametrize("index",range(7),ids=NATIVE)
def test_each_native_sentinel_maps_to_its_named_class(tmp_path,monkeypatch,index):
    logits=np.zeros(7,np.float32);logits[index]=10
    model,_=make(tmp_path,monkeypatch,logits=logits)
    result=model.predict_proba(face_rgb=np.zeros((8,8,3),np.uint8))
    assert result.top1==CANONICAL[index] and list(result.probabilities)==FER7
    assert result.probabilities[CANONICAL[index]]>.99
    assert result.confidence==result.probabilities[result.top1]
    assert result.uncertainty==pytest.approx(1-result.confidence)


def test_native_and_canonical_probabilities_are_both_named_and_ordered(tmp_path,monkeypatch):
    model,path=make(tmp_path,monkeypatch,logits=np.arange(7))
    result=model.predict_proba(np.zeros((8,8,3),np.uint8));meta=result.metadata;provenance=meta.provenance
    assert model.native_labels==tuple(NATIVE) and meta.class_mapping==dict(enumerate(NATIVE))
    assert list(provenance["native_probabilities"])==NATIVE
    assert provenance["original_classes"]==NATIVE and provenance["native_to_canonical"]==BY_NAME
    assert provenance["unsupported_native_classes"]==[]
    assert list(result.probabilities)==FER7 and meta.supported_classes==frozenset(FER7)
    assert sum(result.probabilities.values())==pytest.approx(1)
    for native,canonical in BY_NAME.items():assert provenance["native_probabilities"][native]==result.probabilities[canonical]
    assert provenance["head_spec_sha256"]==hashlib.sha256(path.read_bytes()).hexdigest()
    assert Path(provenance["head_spec"])==path
    assert model.temperature==1.0820367210745798 and meta.calibrated is False


def test_actual_spec_order_drives_mapping_instead_of_a_hardcoded_order(tmp_path,monkeypatch):
    labels=list(reversed(NATIVE));logits=np.zeros(7,np.float32);logits[0]=10
    model,_=make(tmp_path,monkeypatch,classes=labels,logits=logits)
    result=model.predict_proba(np.zeros((8,8,3),np.uint8))
    assert result.top1=="surprise" and model.native_labels==tuple(labels)
    assert list(result.metadata.provenance["native_probabilities"])==labels


@pytest.mark.parametrize("classes",[NATIVE[:-1]+["contempt"],NATIVE[:-1]+["unsupported"],NATIVE[:-1]+["anger"],NATIVE[:-1]+["angry"],NATIVE[:-1],[],NATIVE[:-1]+[None]])
def test_unsupported_duplicate_or_missing_labels_refuse_without_loading_weights(tmp_path,monkeypatch,classes):
    with pytest.raises(CombinedContractError,match="head spec|native label|canonical"):
        make(tmp_path,monkeypatch,classes=classes)


def test_wrong_declared_head_width_is_refused(tmp_path,monkeypatch):
    model,_=make(tmp_path,monkeypatch,spec_changes={"spec":{"n_classes":8}})
    with pytest.raises(CombinedContractError,match="head spec"):
        model.predict_proba(np.zeros((8,8,3),np.uint8))


def test_legacy_paths_fail_with_migration_error_before_access():
    with pytest.raises(BlockedAssetError,match="migration"):
        ClaudeCombinedAdapter("legacy-backbone.onnx","legacy-head.pt",head_spec_path="legacy-sidecar.json")


@pytest.mark.parametrize("width",[6,8])
def test_head_output_width_must_match_the_spec(tmp_path,monkeypatch,width):
    model,_=make(tmp_path,monkeypatch,logits=np.zeros(width))
    with pytest.raises(CombinedContractError,match="head output"):
        model.predict_proba(np.zeros((8,8,3),np.uint8))


def test_existing_runner_contract_and_keyword_input_remain_compatible(tmp_path,monkeypatch):
    logits=np.zeros(7,np.float32);logits[4]=10
    model,_=make(tmp_path,monkeypatch,logits=logits)
    report=run_benchmark([model],[np.zeros((8,8,3),np.uint8)],[6],sample_ids=["synthetic:neutral"],requested_count=1,return_report=True,synthetic=True)
    assert report["complete"] and report["results"]["claude-combined"]["accuracy"]==1


def test_embedding_failure_is_not_replaced_by_a_prediction(tmp_path,monkeypatch):
    model,_=make(tmp_path,monkeypatch)
    def fail(x):raise RuntimeError("synthetic missing embedding")
    monkeypatch.setattr(model,"_get_embedding",fail)
    with pytest.raises(RuntimeError,match="missing embedding"):
        model.predict_proba(np.zeros((8,8,3),np.uint8))
    report=run_benchmark([model],[np.zeros((8,8,3),np.uint8)],[0],sample_ids=["synthetic:missing"],requested_count=1,return_report=True,synthetic=True)
    assert not report["complete"] and "accuracy" not in report["results"]["claude-combined"]
