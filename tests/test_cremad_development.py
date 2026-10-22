"""Manufactured clips/arrays only. Retained scratch, no real dataset or deletion."""
from pathlib import Path
import hashlib
import json
import os
import time
import sys
from unittest import mock
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training import cremad_development as d
from emolab.benchmark import runner as r
from emolab.inference import ModelMetadata,ModelResult

@pytest.fixture
def scratch(tmp_path):return tmp_path

@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    import socket
    def deny(*a,**k):raise AssertionError("network forbidden")
    monkeypatch.setattr(socket.socket,"connect",deny);monkeypatch.setattr(socket,"create_connection",deny)

def clip(scratch,name="1001_DFA_ANG_XX.flv",payload=b"SYNTHETIC NOT MEDIA"):
    p=scratch/name;p.write_bytes(payload);actor,label=d.parse_name(name)
    return p,{"id":name,"actor":actor,"label":label,"bytes":len(payload)}

@pytest.mark.parametrize("name",["0000_DFA_ANG_XX.flv","1001_BAD_ANG_XX.flv","1001_DFA_BAD_XX.flv","../1001_DFA_ANG_XX.flv","1001_DFA_ANG_XX.mp4","1001_DFA_ANG_LO.flv.extra"])
def test_admission_before_open(scratch,name):
    with mock.patch.object(Path,"open",side_effect=AssertionError("must not open")):
        with pytest.raises(d.BoundaryError):d.admitted_path(scratch,name)

def test_metadata_only_count_and_excluded_spy(scratch,monkeypatch):
    p,row=clip(scratch);(scratch/"0000_DFA_ANG_XX.flv").write_bytes(b"SYNTHETIC excluded")
    monkeypatch.setattr(d,"CANONICAL_SOURCE",scratch)
    support={c:int(c=="angry") for c in d.LABELS.values()}
    original_stat=os.stat;original_open=os.open
    excluded=scratch/"0000_DFA_ANG_XX.flv"
    def stat_spy(path,*args,**kwargs):
        assert Path(path)!=excluded,"excluded file stat forbidden"
        return original_stat(path,*args,**kwargs)
    def open_spy(path,*args,**kwargs):
        assert Path(path)!=excluded,"excluded file open forbidden"
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(os,"stat",stat_spy);monkeypatch.setattr(os,"open",open_spy)
    with mock.patch.object(Path,"open",side_effect=AssertionError("metadata may not open payload")):
        m=d.metadata_manifest(scratch,expected_count=1,expected_actors=(1001,),expected_support=support)
        assert m["count"]==1 and m["rows"]==[row]
        with pytest.raises(d.BoundaryError,match="discrepancy"):d.metadata_manifest(scratch)

def test_payload_binding_pointer_and_forgery(scratch):
    p,row=clip(scratch);bound=d.bind_payload(scratch,row);assert bound["sha256"]==hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(d.BoundaryError,match="forged"):d.bind_payload(scratch,{**row,"actor":1002})
    with pytest.raises(d.BoundaryError,match="hash mismatch"):d.bind_payload(scratch,{**row,"sha256":"0"*64})
    p.write_bytes(b"version https://git-lfs.github.com/spec/v1\nSYNTHETIC")
    with pytest.raises(d.BoundaryError,match="LFS"):d.bind_payload(scratch,row)

def test_link_escape(scratch):
    target=scratch/"SYNTHETIC-target";target.write_bytes(b"fixture")
    link=scratch/"1001_DFA_ANG_XX.flv"
    try:link.symlink_to(target)
    except OSError:pytest.skip("symlink privilege unavailable; reparse branch still static")
    with pytest.raises(d.BoundaryError,match="reparse"):d.admitted_path(scratch,link.name)

@pytest.mark.parametrize("n,expected",[(0,[]),(1,[0]),(2,[0]),(5,[1,2,3]),(9,[2,4,6])])
def test_frame_policy(n,expected):assert d.frame_indices(n)==expected

def test_sequential_fake_decode(scratch):
    p,row=clip(scratch);opened=[]
    class Reader:
        def __init__(self,path):self.i=0;self.released=False;opened.append(self)
        def isOpened(self):return True
        def read(self):
            if self.i==5:return False,None
            self.i+=1;return True,np.full((3,4,3),self.i-1,dtype=np.uint8)
        def release(self):self.released=True
    frames,n,bound=d.decode_selected(scratch,row,Reader)
    assert n==5 and [int(f[0,0,0]) for f in frames]==[1,2,3] and len(opened)==2
    assert all(r.released for r in opened) and bound["sha256"]

def test_box_tie_border_and_no_fallback():
    frame=np.arange(8*8*3,dtype=np.uint8).reshape(8,8,3)
    assert d.choose_crop(frame,[]) is None
    boxes=[dict(x=4,y=0,w=2,h=2),dict(x=0,y=0,w=2,h=2),dict(x=-9,y=0,w=1,h=1)]
    crop=d.choose_crop(frame,boxes);assert crop.shape==(3,3,3) and np.array_equal(crop,frame[:3,:3]) and not crop.flags.writeable

def row(i,label,pred=None,actor=1001):
    p=None if pred is None else [float(c==pred) for c in r.FER7]
    return {"id":str(i),"actor":actor,"label":label,"status":"no_face" if p is None else "predicted","probabilities":p,"observed_inference_ms":[100.] if p else []}

def test_fixed_metrics_perfect_missing_surprise():
    rows=[row(i,c,c) for i,c in enumerate(r.DEVELOPMENT_TRUTHS)];m=r.development_metrics(rows)
    assert m["macro_f1"]==1 and m["accuracy"]==1 and m["ece"] is None
    assert m["per_class"]["surprise"]["support"]==0 and m["per_class"]["surprise"]["recall"] is None
    rows[0]=row(0,"angry");m=r.development_metrics(rows)
    assert m["macro_f1"]==pytest.approx(5/6) and m["coverage"]==pytest.approx(5/6)
    assert sum(map(sum,m["confusion"]))==6 and m["confusion"][0][7]==1
    rows[0]=row(0,"angry","surprise");assert r.development_metrics(rows)["macro_f1"]==pytest.approx(5/6)

def test_toy_metrics_fixed_absent_classes():
    m=r.development_metrics([row(0,"angry","angry"),row(1,"angry","happy"),row(2,"happy")])
    assert m["accuracy"]==pytest.approx(1/3) and m["coverage"]==pytest.approx(2/3)
    assert m["per_class"]["angry"]["f1"]==pytest.approx(2/3) and m["macro_f1"]==pytest.approx((2/3)/6)
    assert m["per_class"]["fear"]["recall"] is None

def test_paired_bootstrap_replay_multiplicity():
    rows=[row(0,"angry","angry",1001),row(1,"happy",None,1002)]
    a=r.development_bootstrap({"mini-xception":rows,"vit":rows});b=r.development_bootstrap({"mini-xception":rows,"vit":rows})
    assert a==b and a["replicates"]==1000 and a["actor_count"]==2
    assert a["paired_macro_f1_difference_ci"]["mini-xception minus vit"]==[0.,0.]
    assert 0<a["undefined_recall_draws"]["vit"]["angry"]<1000
    # Both copies of a drawn cluster retain BOTH its clips; fixed denominator.
    assert r.development_metrics([rows[0],rows[0]])["macro_f1"]==pytest.approx(1/6)

class Adapter:
    def __init__(self,name="mini-xception",fail=False):self.name=name;self.fail=fail;self.hashes=[]
    def predict_proba(self,crop):
        assert not crop.flags.writeable
        if self.fail:raise RuntimeError("SYNTHETIC failure")
        self.hashes.append(hashlib.sha256(crop.tobytes()).hexdigest())
        meta=ModelMetadata(self.name,"fixture",(2,2,3),dict(enumerate(r.FER7)),False,frozenset(r.FER7),{"SYNTHETIC":"pinned"})
        return ModelResult(dict(zip(r.FER7,[1.,0,0,0,0,0,0])),"angry",1.,0.,10.,meta)

def test_clip_aggregation_timing_failures_and_fer_preserved():
    a,b=Adapter(),Adapter("vit");manifest={"rows":[{"id":"fixture","actor":1001,"label":"angry"}],"protocol_sha256":"synthetic"}
    ticks=iter([0.,.1,1.,1.1]);crop=np.zeros((2,2,3),np.uint8)
    report=r.run_development_clips([a,b],manifest,lambda row:([crop],{"decoded_frames":5}),clock=lambda:next(ticks))
    assert report["primary_roster_complete"] and a.hashes==b.hashes
    assert report["results"]["mini-xception"]["items"][0]["observed_inference_ms"]==pytest.approx([100.])
    failed=r.run_development_clips([Adapter(fail=True)],manifest,lambda row:([crop],{}));assert not failed["complete"] and failed["results"]["mini-xception"]["metrics"] is None
    empty=r.run_development_clips([Adapter()],manifest,lambda row:([],{"decoded_frames":0}));assert empty["results"]["mini-xception"]["items"][0]["status"]=="decode_failure"
    assert r.run_benchmark([Adapter()], [crop], [0])["mini-xception"]["accuracy"]==1
    for pair in [(1,0),(0,float("nan"))]:
        with pytest.raises(r.BenchmarkContractError):r._duration_ms(*pair)


def test_unreadable_decoder_is_explicit_failure(scratch):
    _,row=clip(scratch)
    class Bad:
        def __init__(self,path):pass
        def isOpened(self):return False
        def release(self):pass
    with pytest.raises(d.DecodeFailure):d.decode_selected(scratch,row,Bad)

def test_equal_frame_average_and_no_face_denominator():
    class Alternating(Adapter):
        def predict_proba(self,crop):
            result=super().predict_proba(crop)
            if len(self.hashes)%2==0:
                from dataclasses import replace
                return replace(result,probabilities=dict(zip(r.FER7,[0.,0,0,1.,0,0,0])))
            return result
    manifest={"rows":[{"id":"fixture","actor":1001,"label":"angry"}],"protocol_sha256":"synthetic"}
    crop=np.zeros((2,2,3),np.uint8)
    report=r.run_development_clips([Alternating()],manifest,lambda row:([crop,crop],{"decoded_frames":3}))
    item=report["results"]["mini-xception"]["items"][0]
    assert item["probabilities"]==[.5,0,0,.5,0,0,0] and item["face_frames"]==2
    none=r.run_development_clips([Adapter()],manifest,lambda row:([],{"decoded_frames":3}))
    assert none["results"]["mini-xception"]["metrics"]["accuracy"]==0 and none["results"]["mini-xception"]["items"][0]["status"]=="no_face"


def test_synthetic_reparse_flag_denied_before_open(scratch,monkeypatch):
    p,row=clip(scratch);original=Path.lstat
    class Flagged:
        st_file_attributes=0x400
        def __init__(self,value):self.value=value
        def __getattr__(self,key):return getattr(self.value,key)
    def lstat(path,*args,**kwargs):
        value=original(path,*args,**kwargs)
        return Flagged(value) if path==p else value
    monkeypatch.setattr(Path,"lstat",lstat)
    with mock.patch.object(Path,"open",side_effect=AssertionError("must not open reparse")):
        with pytest.raises(d.BoundaryError,match="reparse"):d.admitted_path(scratch,row["id"])


@pytest.mark.parametrize("kind",["duplicate","malformed","nonclip","lfs"])
def test_manifest_boundary_cases(scratch,monkeypatch,kind):
    p,_=clip(scratch);monkeypatch.setattr(d,"CANONICAL_SOURCE",scratch)
    support={c:int(c=="angry") for c in d.LABELS.values()}
    kwargs=dict(expected_count=1,expected_actors=(1001,),expected_support=support)
    if kind=="duplicate":
        original=Path.iterdir
        monkeypatch.setattr(Path,"iterdir",lambda path:iter([p,p]) if path==scratch else original(path))
        with pytest.raises(d.BoundaryError,match="duplicate clip ID"):d.metadata_manifest(scratch,**kwargs)
    elif kind=="malformed":
        (scratch/"1001_BAD_ANG_XX.flv").write_bytes(b"SYNTHETIC")
        with pytest.raises(d.BoundaryError,match="unknown clip token"):d.metadata_manifest(scratch,**kwargs)
    elif kind=="nonclip":
        (scratch/"README").write_text("SYNTHETIC",encoding="ascii")
        with pytest.raises(d.BoundaryError,match="BLOCKED: non-clip directory entry discrepancy"):d.metadata_manifest(scratch,**kwargs)
    else:
        p.write_bytes(b"version https://git-lfs.github.com/spec/v1\nSYNTHETIC")
        # Metadata must NOT read bytes; the admitted row then fails at binding.
        manifest=d.metadata_manifest(scratch,**kwargs)
        with pytest.raises(d.BoundaryError,match="LFS pointer"):d.bind_payload(scratch,manifest["rows"][0])


def test_multiclip_actor_bootstrap_exact_draws_and_grouping(monkeypatch):
    rows=[row(0,"angry","angry",1001),row(1,"angry","happy",1001),
          row(2,"happy","happy",1002),row(3,"happy",None,1002),row(4,"fear","fear",1002)]
    draws=np.random.default_rng(505).choice([1001,1002],size=(1000,2),replace=True)
    seen=[];original=r.development_metrics
    def spy(items):seen.append(len(items));return original(items)
    monkeypatch.setattr(r,"development_metrics",spy)
    result=r.development_bootstrap({"mini-xception":rows,"vit":rows})
    assert result["replicates"]==1000 and result["draw_shape"]==[1000,2]
    assert result["draws_sha256"]==hashlib.sha256(draws.tobytes()).hexdigest()
    assert seen==[2,3,2,3] # validate/cache whole actor clusters just once/model
    oracle=[original([item for actor in draw for item in rows if item["actor"]==actor]) for draw in draws]
    values=[metric["macro_f1"] for metric in oracle]
    for label in r.DEVELOPMENT_TRUTHS:
        recalls=[metric["per_class"][label]["recall"] for metric in oracle]
        defined=[value for value in recalls if value is not None]
        expected=np.percentile(defined,[2.5,97.5]) if defined else None
        for model in ("mini-xception","vit"):
            assert result["undefined_recall_draws"][model][label]==recalls.count(None)
            if expected is None:assert result["recall_ci"][model][label] is None
            else:assert result["recall_ci"][model][label]==pytest.approx(expected)
    assert result["macro_f1_ci"]["vit"]==pytest.approx(np.percentile(values,[2.5,97.5]))
    assert result["paired_macro_f1_difference_ci"]["mini-xception minus vit"]==[0.,0.]


def test_bootstrap_fake_cohort_timing_guard():
    # Manufactured 46 clusters x 82 clips, two models: near full cohort scale.
    rows=[row(f"{a}-{i}",r.DEVELOPMENT_TRUTHS[i%6],r.FER7[i%7],a) for a in d.ACTORS for i in range(82)]
    start=time.perf_counter();result=r.development_bootstrap({"mini-xception":rows,"vit":rows})
    assert time.perf_counter()-start<10 # cached implementation; former per-draw loop is minutes
    assert result["draw_shape"]==[1000,46] and result["replicates"]==1000


def test_pilot_has_no_scores_or_vectors(monkeypatch):
    manifest={"rows":[{"id":"SYNTHETIC","actor":1001,"label":"angry"}],"protocol_sha256":"synthetic"}
    def forbidden(*args):raise AssertionError("pilot may not compute scores")
    monkeypatch.setattr(r,"development_metrics",forbidden)
    report=r.run_development_clips([Adapter()],manifest,lambda row:([np.zeros((2,2,3),np.uint8)],{"decoded_frames":3}),timing_only=True)
    model=report["results"]["mini-xception"];item=model["items"][0]
    assert report["complete"] and model["metrics"] is None
    assert "probabilities" not in item and item["status"]=="processed" and item["observed_inference_ms"]
    assert not any(key in json.dumps(report) for key in ('"macro_f1"','"accuracy"','"confusion"','"probabilities"'))
