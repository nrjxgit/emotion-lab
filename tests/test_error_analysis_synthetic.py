"""synthetic_only=true; SYNTHETIC / NOT REAL MODEL PERFORMANCE; seed 505."""
from copy import deepcopy
import builtins
import hashlib
from html import escape
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training import error_analysis as analysis


def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()


@pytest.fixture(autouse=True)
def no_models_network_or_scan(monkeypatch):
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.startswith(("emolab.adapters","backend","torch","tensorflow","onnxruntime","transformers")):
            raise AssertionError("model import forbidden")
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",guarded)
    import socket
    def denied(*args,**kwargs):raise AssertionError("network/directory scan forbidden")
    monkeypatch.setattr(socket.socket,"connect",denied)
    monkeypatch.setattr(Path,"iterdir",denied);monkeypatch.setattr(Path,"glob",denied);monkeypatch.setattr(Path,"rglob",denied)


def fixture(tmp_path):
    root=tmp_path/"SYNTHETIC-analysis";root.mkdir()
    clips=[{"id":"SYNTHETIC-clip1","actor":"SYNTHETIC-actorA","label":"angry"},
           {"id":"SYNTHETIC-clip2","actor":"SYNTHETIC-actorA","label":"happy"},
           {"id":"SYNTHETIC-clip3","actor":"SYNTHETIC-actorB","label":"fear"},
           {"id":"SYNTHETIC-clip4","actor":"SYNTHETIC-actorB","label":"sad"}]
    models={"SYNTHETIC-modelA":list(analysis.LABELS),"SYNTHETIC-modelB":list(analysis.LABELS)}
    manifest={"schema":"synthetic-per-clip-manifest/1","synthetic_only":True,"exposure":analysis.MARKER,"seed":505,"complete":True,
              "protocol_sha256":sha(b"SYNTHETIC frozen protocol"),"models":models,"clips":clips,
              "class_support":{label:sum(c["label"]==label for c in clips) for label in analysis.TRUTHS}}
    bundle=SimpleNamespace(root=root,manifest=manifest,rows=[])
    statuses=["predicted","predicted","no_face","failed","predicted","decode_failure","no_face","predicted"]
    vectors=[[1.,0,0,0,0,0,0],[.3,0,0,.2,0,.5,0],None,None,[.3,0,0,.4,0,0,.3],None,None,[1/7]*7]
    for (model,support),offset in zip(models.items(),(0,4)):
        for index,clip in enumerate(clips):
            vector=vectors[offset+index]
            bundle.rows.append({"schema":"synthetic-per-clip/1","synthetic_only":True,"exposure":analysis.MARKER,"seed":505,"complete":True,
                "manifest_sha256":sha(canonical(manifest)),"protocol_sha256":manifest["protocol_sha256"],"model":model,**clip,
                "supported_classes":list(support),"status":statuses[offset+index],"probabilities":vector,
                "uncertainty":None if vector is None else 1-max(vector)})
    def refresh():
        bundle.manifest_pin=sha(canonical(bundle.manifest))
        raw=b"".join(json.dumps(row,allow_nan=True).encode()+b"\n" for row in bundle.rows)
        (root/"per-clip.jsonl").write_bytes(raw);bundle.file_pin=sha(raw)
        return raw
    def read(**kwargs):
        return analysis.read_analysis(root,"per-clip.jsonl",manifest=bundle.manifest,expected_manifest_sha256=bundle.manifest_pin,expected_file_sha256=bundle.file_pin,**kwargs)
    bundle.refresh=refresh;bundle.read=read;refresh()
    return bundle


def test_hand_counts_actors_classes_status_and_paired_denominators(tmp_path):
    b=fixture(tmp_path);result=b.read()
    a=result["models"]["SYNTHETIC-modelA"];other=result["models"]["SYNTHETIC-modelB"]
    assert result["synthetic_only"] is True and result["exposure"]==analysis.MARKER and result["seed"]==505
    assert result["row_count"]==8 and result["cohort_size"]==4 and not result["analysis_complete"]
    assert a["denominator"]==4 and a["prediction_count"]==2 and a["coverage"]==.5
    assert a["status_counts"]=={"predicted":2,"no_face":1,"decode_failure":0,"failed":1}
    assert not a["model_complete"] and other["model_complete"]
    assert a["confusion"][0][0]==1 and a["confusion"][3][5]==1
    assert a["confusion"][2][7]==1 and a["confusion"][4][9]==1 # missing/failure distinct, no invented predictions
    assert sum(map(sum,a["confusion"]))==4
    assert a["by_actor"]["SYNTHETIC-actorA"]["coverage"]==1
    assert a["by_actor"]["SYNTHETIC-actorB"]["coverage"]==0
    assert a["by_class"]["disgust"]["denominator"]==0 and a["by_class"]["disgust"]["coverage"] is None
    assert a["uncertainty"]["counts"]=={"low_[0,1/3)":1,"medium_[1/3,2/3)":1,"high_[2/3,1]":0,"unavailable":2}
    assert other["uncertainty"]["counts"]["high_[2/3,1]"]==1
    pair=result["paired"][0]
    assert pair["cohort_denominator"]==4 and pair["both_predicted_denominator"]==1
    assert pair["prediction_disagreement_count"]==1 and pair["prediction_disagreement_rate"]==1
    assert pair["one_predicted_count"]==2 and pair["neither_predicted_count"]==1
    assert pair["outcome_disagreement_count"]==3 and pair["outcome_disagreement_rate"]==.75
    assert result["input_sha256"]==b.file_pin and result["manifest_sha256"]==b.manifest_pin


@pytest.mark.parametrize("kind",["schema","exposure","not_synthetic","seed","incomplete","missing_field","extra_field","protocol","manifest","model","actor","clip","label","support","nan","width","negative","mass","boolean","uncertainty_nan","uncertainty_wrong","failed_vector","status","bad_identity_type"])
def test_bad_rows_fail_closed(tmp_path,kind):
    b=fixture(tmp_path);row=b.rows[0]
    if kind=="schema":row["schema"]="other"
    elif kind=="exposure":row["exposure"]="real development"
    elif kind=="not_synthetic":row["synthetic_only"]=False
    elif kind=="seed":row["seed"]=506
    elif kind=="incomplete":row["complete"]=False
    elif kind=="missing_field":row.pop("uncertainty")
    elif kind=="extra_field":row["extra"]=True
    elif kind=="protocol":row["protocol_sha256"]="0"*64
    elif kind=="manifest":row["manifest_sha256"]="0"*64
    elif kind=="model":row["model"]="SYNTHETIC-unrequested"
    elif kind=="actor":row["actor"]="SYNTHETIC-wrong-actor"
    elif kind=="clip":row["id"]="SYNTHETIC-wrong-clip"
    elif kind=="label":row["label"]="surprise"
    elif kind=="support":row["supported_classes"]=["angry"]
    elif kind=="nan":row["probabilities"][0]=float("nan")
    elif kind=="width":row["probabilities"]=[1.]*6
    elif kind=="negative":row["probabilities"]=[-.1,0,0,1.1,0,0,0]
    elif kind=="mass":row["probabilities"]=[.5,0,0,0,0,0,0]
    elif kind=="boolean":row["probabilities"]=[True,0,0,0,0,0,0]
    elif kind=="uncertainty_nan":row["uncertainty"]=float("nan")
    elif kind=="uncertainty_wrong":row["uncertainty"]=-.1
    elif kind=="failed_vector":row["status"]="failed"
    elif kind=="status":row["status"]="unknown"
    else:row["id"]=["SYNTHETIC-not-a-string"]
    b.refresh()
    with pytest.raises(analysis.SyntheticAnalysisError):b.read()


@pytest.mark.parametrize("kind",["duplicate_manifest_id","duplicate_row","missing_row","support_count","model_missing","protocol_pin","unknown_manifest","incomplete_manifest","support_schema"])
def test_manifest_and_complete_cohort_binding(tmp_path,kind):
    b=fixture(tmp_path)
    if kind=="duplicate_manifest_id":b.manifest["clips"][1]["id"]=b.manifest["clips"][0]["id"]
    elif kind=="duplicate_row":b.rows.append(deepcopy(b.rows[0]))
    elif kind=="missing_row":b.rows.pop()
    elif kind=="support_count":b.manifest["class_support"]["angry"]+=1
    elif kind=="model_missing":b.manifest["models"].pop("SYNTHETIC-modelB")
    elif kind=="protocol_pin":b.manifest["protocol_sha256"]="not-a-hash"
    elif kind=="unknown_manifest":b.manifest["extra"]=True
    elif kind=="incomplete_manifest":b.manifest["complete"]=False
    else:b.manifest["models"]["SYNTHETIC-modelA"]=[{}]
    b.refresh()
    with pytest.raises(analysis.SyntheticAnalysisError):b.read()


def test_unsupported_probability_mass_is_rejected(tmp_path):
    b=fixture(tmp_path);b.manifest["models"]["SYNTHETIC-modelA"]=["happy"]
    pin=sha(canonical(b.manifest))
    for row in b.rows:
        row["manifest_sha256"]=pin
        if row["model"]=="SYNTHETIC-modelA":row["supported_classes"]=["happy"]
    b.refresh()
    with pytest.raises(analysis.SyntheticAnalysisError,match="unsupported class"):b.read()


def test_zero_prediction_disagreement_is_unavailable_not_zero(tmp_path):
    b=fixture(tmp_path)
    for row in b.rows:row.update(status="no_face",probabilities=None,uncertainty=None)
    b.refresh();result=b.read();pair=result["paired"][0]
    assert result["analysis_complete"] and pair["both_predicted_denominator"]==0
    assert pair["prediction_disagreement_rate"] is None and pair["neither_predicted_count"]==4
    for value in result["models"].values():
        assert value["coverage"]==0 and value["uncertainty"]["defined_denominator"]==0
        assert value["uncertainty"]["counts"]["unavailable"]==4


@pytest.mark.parametrize("kind",["file_pin","manifest_pin","byte_cap","row_cap","hard_limit","line_cap","no_newline","bad_json","duplicate_json_key"])
def test_bounded_snapshot_and_json_rejections(tmp_path,kind):
    b=fixture(tmp_path);path=b.root/"per-clip.jsonl";kwargs={}
    if kind=="file_pin":b.file_pin="0"*64
    elif kind=="manifest_pin":b.manifest_pin="0"*64
    elif kind=="byte_cap":kwargs["max_bytes"]=32
    elif kind=="row_cap":kwargs["max_rows"]=7
    elif kind=="hard_limit":kwargs["max_bytes"]=analysis.MAX_BYTES+1
    else:
        raw=path.read_bytes()
        if kind=="line_cap":raw=b" "*(analysis.MAX_LINE_BYTES+1)+b"\n"
        elif kind=="no_newline":raw=raw.rstrip(b"\n")
        elif kind=="bad_json":raw=b"{bad}\n"
        else:raw=b'{"schema":"a","schema":"b"}\n'
        path.write_bytes(raw);b.file_pin=sha(raw)
    with pytest.raises(analysis.SyntheticAnalysisError):b.read(**kwargs)


@pytest.mark.parametrize("name",["../per-clip.jsonl","sub/per-clip.jsonl","sub\\per-clip.jsonl","other.jsonl"])
def test_no_filename_or_path_following(tmp_path,name,monkeypatch):
    b=fixture(tmp_path)
    monkeypatch.setattr(Path,"open",Mock(side_effect=AssertionError("invalid path opened")))
    with pytest.raises(analysis.SyntheticAnalysisError):
        analysis.read_analysis(b.root,name,manifest=b.manifest,expected_manifest_sha256=b.manifest_pin,expected_file_sha256=b.file_pin)


@pytest.mark.parametrize("component",["file","root","ancestor"])
def test_reparse_denied_before_open(tmp_path,monkeypatch,component):
    b=fixture(tmp_path);path=b.root/"per-clip.jsonl";original=Path.lstat
    target={"file":path,"root":b.root,"ancestor":b.root.parent}[component]
    def lstat(p,*args,**kwargs):
        info=original(p,*args,**kwargs)
        return SimpleNamespace(st_mode=info.st_mode,st_file_attributes=0x400) if p==target else info
    monkeypatch.setattr(Path,"lstat",lstat)
    monkeypatch.setattr(Path,"open",Mock(side_effect=AssertionError("reparse opened")))
    with pytest.raises(analysis.SyntheticAnalysisError,match="reparse"):b.read()


def test_one_file_read_exact_bytes_no_directory_scanning(tmp_path,monkeypatch):
    b=fixture(tmp_path);calls=[];original=Path.open
    def opened(path,*args,**kwargs):
        calls.append((path,args[0] if args else "r"));return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",opened)
    result=b.read()
    assert calls==[(b.root/"per-clip.jsonl","rb")]
    assert result["input_sha256"]==b.file_pin


def test_untrusted_names_are_escaped_and_all_report_tables_are_labelled(tmp_path):
    b=fixture(tmp_path);old="SYNTHETIC-modelA";name='SYNTHETIC-<script>alert("x")</script>&'
    b.manifest["models"][name]=b.manifest["models"].pop(old)
    actor='SYNTHETIC-<img src=x onerror="bad">'
    b.manifest["clips"][0]["actor"]=actor
    pin=sha(canonical(b.manifest))
    for row in b.rows:
        if row["model"]==old:row["model"]=name
        if row["id"]=="SYNTHETIC-clip1":row["actor"]=actor
        row["manifest_sha256"]=pin
    b.refresh();result=b.read();text=analysis.render_html(result)
    assert name not in text and actor not in text
    assert escape(name,quote=True) in text and escape(actor,quote=True) in text
    assert '<script>' not in text and '<img' not in text
    assert text.count('<table ')==text.count('data-synthetic-only="true"')==text.count('<caption>')
    assert all(analysis.MARKER in caption.split('</caption>')[0] for caption in text.split('<caption>')[1:])
    assert all(value["synthetic_only"] is True and value["exposure"]==analysis.MARKER for value in result["models"].values())
    # Shareable artifacts contain only this manufactured cohort and escaped HTML.
    (tmp_path/"SYNTHETIC-report.html").write_text(text,encoding="utf-8")
    (tmp_path/"SYNTHETIC-report.json").write_text(json.dumps(result,allow_nan=False),encoding="utf-8")


@pytest.mark.parametrize("kind",["ordinary_root","missing_input"])
def test_only_declared_synthetic_root_and_explicit_existing_input(tmp_path,monkeypatch,kind):
    b=fixture(tmp_path)
    root=tmp_path/"ordinary-results" if kind=="ordinary_root" else tmp_path/"SYNTHETIC-missing"
    if kind=="missing_input":root.mkdir()
    monkeypatch.setattr(Path,"open",Mock(side_effect=AssertionError("invalid root/input opened")))
    with pytest.raises(analysis.SyntheticAnalysisError):
        analysis.read_analysis(root,"per-clip.jsonl",manifest=b.manifest,expected_manifest_sha256=b.manifest_pin,expected_file_sha256=b.file_pin)


def test_input_change_during_read_is_rejected(tmp_path,monkeypatch):
    b=fixture(tmp_path);path=b.root/"per-clip.jsonl";original=Path.open
    class Changing:
        def __init__(self,stream):self.stream=stream
        def __enter__(self):self.stream.__enter__();return self
        def __exit__(self,*args):return self.stream.__exit__(*args)
        def fileno(self):return self.stream.fileno()
        def read(self,count):
            data=self.stream.read(count)
            with original(path,"ab") as writer:writer.write(b"CHANGED")
            return data
    def opened(p,*args,**kwargs):
        stream=original(p,*args,**kwargs)
        return Changing(stream) if p==path and args and args[0]=="rb" else stream
    monkeypatch.setattr(Path,"open",opened)
    with pytest.raises(analysis.SyntheticAnalysisError,match="changed"):b.read()


def test_ties_choose_first_class_in_declared_label_order(tmp_path):
    b=fixture(tmp_path)
    b.rows[0].update(probabilities=[.5,0,0,.5,0,0,0],uncertainty=.5)
    b.refresh();result=b.read()
    a=result["models"]["SYNTHETIC-modelA"]
    assert analysis.LABELS[0]=="angry" and a["confusion"][0][0]==1
    assert a["confusion"][0][3]==0
    assert result["paired"][0]["prediction_disagreement_count"]==1


@pytest.mark.parametrize("edge,bucket",[(1/3,"medium_[1/3,2/3)"),(2/3,"high_[2/3,1]")])
def test_exact_uncertainty_bucket_edges(tmp_path,edge,bucket):
    b=fixture(tmp_path)
    # Input uncertainty is exactly the edge; probabilities agree within row tolerance.
    vector=[2/3,1/3,0,0,0,0,0] if edge==1/3 else [1/3,1/3,1/3,0,0,0,0]
    b.rows[0].update(probabilities=vector,uncertainty=edge)
    b.refresh();result=b.read()
    buckets=result["models"]["SYNTHETIC-modelA"]["by_class"]["angry"]["uncertainty"]["counts"]
    assert buckets[bucket]==1 and sum(buckets.values())==1


@pytest.mark.parametrize("kind",["not_dict","missing","extra","models_type","models_bound","identifier",
    "summary_missing","count_bool","count_negative","coverage_nan","coverage_wrong","complete_type",
    "status_shape","status_total","order","matrix_shape","matrix_count","class_support",
    "uncertainty_missing","uncertainty_bound","actors_type","actors_bound","actor_group",
    "class_group","support","pairs_type","pair_missing","pair_count","pair_rate","pin","seed"])
def test_render_html_rejects_malformed_caller_reports(tmp_path,kind):
    b=fixture(tmp_path);report=b.read();a=report["models"]["SYNTHETIC-modelA"];pair=report["paired"][0]
    if kind=="not_dict":report=[]
    elif kind=="missing":report.pop("models")
    elif kind=="extra":report["unexpected"]=1
    elif kind=="models_type":report["models"]=[]
    elif kind=="models_bound":report["models"]={}
    elif kind=="identifier":report["models"]["not-manufactured"]=report["models"].pop("SYNTHETIC-modelA")
    elif kind=="summary_missing":a.pop("confusion")
    elif kind=="count_bool":a["denominator"]=True
    elif kind=="count_negative":a["prediction_count"]=-1
    elif kind=="coverage_nan":a["coverage"]=float("nan")
    elif kind=="coverage_wrong":a["coverage"]=.99
    elif kind=="complete_type":a["model_complete"]=0
    elif kind=="status_shape":a["status_counts"].pop("failed")
    elif kind=="status_total":a["status_counts"]["predicted"]+=1
    elif kind=="order":a["truth_order"]=list(reversed(analysis.TRUTHS))
    elif kind=="matrix_shape":a["confusion"][0].pop()
    elif kind=="matrix_count":a["confusion"][0][0]=True
    elif kind=="class_support":a["class_support"]["angry"]=5
    elif kind=="uncertainty_missing":a["uncertainty"].pop("counts")
    elif kind=="uncertainty_bound":a["uncertainty"]["counts"]["unavailable"]=-1
    elif kind=="actors_type":a["by_actor"]=[]
    elif kind=="actors_bound":a["by_actor"]={}
    elif kind=="actor_group":next(iter(a["by_actor"].values()))["coverage"]=2
    elif kind=="class_group":a["by_class"].pop("fear")
    elif kind=="support":a["supported_classes"]=["other"]
    elif kind=="pairs_type":report["paired"]={}
    elif kind=="pair_missing":pair.pop("models")
    elif kind=="pair_count":pair["prediction_disagreement_count"]=5
    elif kind=="pair_rate":pair["outcome_disagreement_rate"]=float("inf")
    elif kind=="pin":report["input_sha256"]="bad"
    else:report["seed"]=True
    with pytest.raises(analysis.SyntheticAnalysisError):analysis.render_html(report)


def test_render_html_validates_without_mutation_and_uses_ascii_separator(tmp_path):
    b=fixture(tmp_path);report=b.read();before=deepcopy(report)
    html=analysis.render_html(report)
    assert report==before and analysis.MARKER+" - cohort/status/coverage" in html
    assert analysis.MARKER+" ? " not in html
    for row in b.rows:row.update(status="no_face",probabilities=None,uncertainty=None)
    b.refresh();assert "<!doctype html>" in analysis.render_html(b.read())
