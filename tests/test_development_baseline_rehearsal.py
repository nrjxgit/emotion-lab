"""SYNTHETIC end-to-end rehearsal; TEST SUBSTITUTION / NO HUMAN AUTHORITY.

This file is also the test-only inert Job child. It cannot use real asset/data
paths; no production CLI/config flag selects it. All arrays and clip bytes are
manufactured under an explicitly labelled temporary root. No real codec/model.
Fixture labels are stamped by this harness, not proof of production propagation.
COMPLETE marker AND checkpoint are required; checkpoint alone is not authority.
Source pins need independent review/rebinding; these tests grant no authority.
"""
from contextlib import contextmanager
from copy import deepcopy
import builtins
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch
import uuid
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training import run_development_baseline as entry
from training import cremad_development as development
from emolab.benchmark import runner
from emolab.inference import ModelMetadata,ModelResult

MARKER="SYNTHETIC / NOT REAL MODEL PERFORMANCE"
AUTH="TEST SUBSTITUTION / NO HUMAN AUTHORITY"
REAL_SUPERVISE=entry.supervise
REAL_WRITE=entry.write_json
REAL_VALIDATE=entry.validate_config
EVIDENCE=[]


def stamp(value):
    if isinstance(value,dict):return {**{k:stamp(v) for k,v in value.items()},"synthetic_only":True,"exposure":MARKER}
    if isinstance(value,list):return [stamp(v) for v in value]
    return value


def synthetic_path(value,root):
    # Lexical refusal BEFORE stat/hash/open. No real roster path is ever followed.
    path=Path(value)
    if not path.is_absolute() or ".." in path.parts or not path.is_relative_to(root):
        raise ValueError("real/outside paths refused by test-only rehearsal")
    return development.safe_path(path)


@contextmanager
def guards(root):
    counts={"forbidden_io":0,"network":0,"model_import":0}
    original_open=io.open;original_builtin=builtins.open;original_import=builtins.__import__
    forbidden_roots=[ROOT/"models",ROOT/"eval_data_cremad",ROOT/"guarded",ROOT/"docs/emotion-lab/latest",Path("C:/emolab"),Path.home()/".codex"]
    def check(file,mode):
        if isinstance(file,int):return
        path=Path(file).absolute()
        outside=not path.is_relative_to(root)
        blocked=any(path.is_relative_to(prefix) for prefix in forbidden_roots)
        payload=path.suffix.lower() in {".flv",".npz",".hdf5",".h5",".safetensors",".onnx",".mp4",".wav",".jpg",".png"}
        if blocked or (outside and (payload or any(x in str(mode) for x in "wax+"))):
            counts["forbidden_io"]+=1;raise AssertionError("real payload/protected IO forbidden")
    def opened(file,mode="r",*args,**kw):check(file,mode);return original_open(file,mode,*args,**kw)
    def builtin_opened(file,mode="r",*args,**kw):check(file,mode);return original_builtin(file,mode,*args,**kw)
    def deny(*args,**kwargs):counts["network"]+=1;raise AssertionError("network forbidden")
    def imported(name,*args,**kwargs):
        if name.startswith(("torch","tensorflow","transformers","emolab.adapters","backend.app.face.detector","backend.app.face.fer_model")):
            counts["model_import"]+=1;raise AssertionError("real factory/model import forbidden")
        return original_import(name,*args,**kwargs)
    with patch.object(io,"open",opened),patch.object(builtins,"open",builtin_opened),patch.object(builtins,"__import__",imported),patch.object(socket.socket,"connect",deny),patch.object(socket,"create_connection",deny):
        yield counts


def synthetic_validate(config,root):
    if config.get("synthetic_only") is not True or config.get("authority_label")!=AUTH:
        raise ValueError("test-only explicit synthetic declaration required")
    if config.get("roster")!=entry.ROSTER or config.get("protocol_sha256")!=development.PROTOCOL_SHA256 or config.get("phase") not in entry.CAPS:
        raise ValueError("synthetic protocol/roster/phase mismatch")
    if config["phase"]=="full" and config.get("pilot_accepted") is not True:
        raise ValueError("test substitution pilot acceptance absent")
    records=[]
    for key in ("rights_disposition","source_review","command_run_order"):
        ref=config.get(key)
        if not isinstance(ref,dict) or not ref.get("path") or not ref.get("sha256"):
            raise ValueError("synthetic evidence gate missing: "+key)
        path=synthetic_path(ref["path"],root)
        if json.loads(path.read_bytes()).get("authority_label")!=AUTH:raise ValueError("test evidence must disclaim authority")
        records.append(ref)
    assets=config.get("assets",[])
    expected={root/"SYNTHETIC-adapter.txt",root/"SYNTHETIC-detector.txt"}
    if {synthetic_path(r["path"],root) for r in assets}!=expected:raise ValueError("missing synthetic asset pin")
    records+=assets
    for ref in records:
        if not isinstance(ref.get("sha256"),str) or len(ref["sha256"])!=64:raise ValueError("missing synthetic asset/evidence pin")
    # Actual runner source only, explicitly permitted source, never a frozen asset.
    if entry.sha_file(Path(entry.__file__))!=config.get("runtime_source_sha256"):raise ValueError("drifted source hash")
    return entry.verify_assets(records)  # real hash/safe-path validation of tiny manufactured bytes


def install_labels(mode="success"):
    entry.EXPOSURE=MARKER;runner.DEVELOPMENT_EXPOSURE=MARKER
    def writer(path,obj,*,exclusive=False):
        if mode=="missing_rows" and Path(path).name=="model-result.json" and Path(path).parent.name=="mini-xception":
            obj=deepcopy(obj);obj["results"]["mini-xception"]["items"].pop()
        return REAL_WRITE(path,stamp(obj),exclusive=exclusive)
    entry.write_json=writer


class FakeReader:
    def __init__(self,path):
        self.description=json.loads(Path(path).read_bytes());self.index=0
        assert self.description["synthetic_only"] is True
    def isOpened(self):return not self.description.get("decode_failure",False)
    def read(self):
        if self.index==5:return False,None
        value=self.description["class_index"]*20+self.index;self.index+=1
        return True,np.full((8,8,3),value,dtype=np.uint8)
    def release(self):pass


class FakeAdapter:
    def __init__(self,name,mode):self.name=name;self.mode=mode;self.calls=0
    def predict_proba(self,crop):
        self.calls+=1
        if self.mode=="adapter_crash" and self.calls==3:raise RuntimeError("SYNTHETIC original adapter diagnostic")
        index=min(6,int(crop[0,0,0])//20)
        meta=ModelMetadata(self.name,"SYNTHETIC",(8,8,3),dict(enumerate(runner.FER7)),False,frozenset(runner.FER7),{"authority_label":AUTH,"source":"SYNTHETIC no weights"})
        return ModelResult({name:float(i==index) for i,name in enumerate(runner.FER7)},runner.FER7[index],1.,0.,1.,meta)


def child(config_path,out,name):
    config=json.loads(config_path.read_bytes());root=Path(config["synthetic_root"])
    assert root.name.startswith("SYNTHETIC-")
    with guards(root) as counts:
        entry.require_worker_job()  # actual named Windows Job membership, not a token-only stub
        synthetic_path(config_path,root);synthetic_path(out,root)
        synthetic_validate(config,root);install_labels(config.get("case","success"))
        assert json.loads((out/"checkpoint.json").read_bytes())["status"]=="INCOMPLETE"
        mode=config.get("case","success")
        entry.write_json(out/"guard-counts.json",counts,exclusive=True)
        try:
            manifest=json.loads((out.parent/"manifest.json").read_bytes())
            if name=="mini-xception" and mode in ("worker_crash","cap","corrupt_progress"):
                if mode=="corrupt_progress":(out/"worker-progress.json").write_text('{"attempted_ids":[',encoding="ascii");return
                entry.write_progress(out/"worker-progress.json",{"attempted_ids":[manifest["rows"][0]["id"]]})
                if mode=="worker_crash":
                    entry.write_json(out/"worker-failure.json",{"status":"INCOMPLETE","error":"SYNTHETIC intentional worker crash"},exclusive=True);os._exit(23)
                time.sleep(60)
            if name=="report":
                if mode=="report_cap":time.sleep(60)
                reports={n:json.loads((out.parent/n/"model-result.json").read_bytes()) for n in ("mini-xception","vit")}
                entry.publish_report(out.parent,manifest,reports)
                return
            adapter=FakeAdapter(name,mode)
            adapter.predict_proba(np.zeros((8,8,3),np.uint8))  # manufactured warmup only
            def prepare(row):
                source=root/"SYNTHETIC-clips";bound=development.bind_payload(source,row)
                try:frames,n,bound=development.decode_selected(source,bound,FakeReader)
                except development.DecodeFailure:return [],{"decoded_frames":0,"selected_frames":0,"sha256":bound["sha256"],"decode_detection_crop_ms":0.}
                crops=[]
                for frame in frames:
                    # Fake detector boxes; real canonical crop implementation.
                    boxes=[] if row["actor"]==1003 and row["label"]=="fear" else [dict(x=1,y=1,w=4,h=4)]
                    if mode=="crop_mismatch" and name=="vit":frame=frame+np.uint8(1)
                    crop=development.choose_crop(frame,boxes)
                    if crop is not None:crops.append(crop)
                return crops,{"decoded_frames":n,"selected_frames":len(frames),"sha256":bound["sha256"],"decode_detection_crop_ms":0.}
            ticks=iter(i*.001 for i in range(1000))
            environment={"authority_label":AUTH,"timing":"SYNTHETIC counter, NOT measured inference latency","versions":"SYNTHETIC fake detector/adapter/decoder"}
            entry._worker_core(config,out,name,manifest,adapter,prepare,cold_ms=0.,environment=environment,clock=lambda:next(ticks))
        except BaseException as exc:
            entry.write_json(out/"worker-failure.json",{"status":"INCOMPLETE","error":type(exc).__name__+": "+str(exc)},exclusive=True)
            raise
        finally:
            entry.write_json(out/"guard-counts-final.json",counts,exclusive=True)
            assert not any(counts.values()),counts


@pytest.fixture(scope="session",autouse=True)
def evidence_file():
    yield
    target=os.environ.get("EL005_P3_EVIDENCE")
    if target:
        with Path(target).open("x",encoding="utf-8") as f:json.dump(stamp({"cases":EVIDENCE,"limitation":"Synthetic source/orchestration and real inert Windows Jobs only; NO real data/model/asset/codec or human authority."}),f,indent=2)


@pytest.fixture
def protected(tmp_path):
    root=tmp_path/"SYNTHETIC-P3";root.mkdir()
    with guards(root) as counts:
        yield root
        assert not any(counts.values()),counts


@pytest.fixture
def rehearsal(protected,monkeypatch):
    root=protected;source=root/"SYNTHETIC-clips";source.mkdir();outputs=root/"SYNTHETIC-runs";outputs.mkdir()
    for actor in development.ACTORS[:3]:
        for token,label in development.LABELS.items():
            payload={"synthetic_only":True,"class_index":runner.FER7.index(label),"decode_failure":actor==1003 and label=="disgust"}
            (source/f"{actor}_DFA_{token}_XX.flv").write_text(json.dumps(payload),encoding="ascii")
    config={"synthetic_only":True,"exposure":MARKER,"authority_label":AUTH,"synthetic_root":str(root),"phase":"full","pilot_accepted":True,"protocol_sha256":development.PROTOCOL_SHA256,"roster":deepcopy(entry.ROSTER),"runtime_source_sha256":entry.sha_file(Path(entry.__file__)),"assets":[]}
    for key in ("rights_disposition","source_review","command_run_order"):
        path=root/("SYNTHETIC-"+key+".json");path.write_text(json.dumps({"authority_label":AUTH,"gate":key}),encoding="ascii")
        config[key]={"path":str(path),"sha256":entry.sha_file(path)}
    for name in ("SYNTHETIC-adapter.txt","SYNTHETIC-detector.txt"):
        path=root/name;path.write_text(AUTH,encoding="ascii");config["assets"].append({"path":str(path),"sha256":entry.sha_file(path)})
    monkeypatch.setattr(entry,"validate_config",lambda cfg:synthetic_validate(cfg,root))
    monkeypatch.setattr(entry,"EXPOSURE",MARKER);monkeypatch.setattr(runner,"DEVELOPMENT_EXPOSURE",MARKER)
    monkeypatch.setattr(development,"CANONICAL_SOURCE",source)
    def labelled(path,obj,*,exclusive=False):return REAL_WRITE(path,stamp(obj),exclusive=exclusive)
    monkeypatch.setattr(entry,"write_json",labelled)
    jobs=[];requested_caps=[]
    def supervised(argv,out,**kwargs):
        name=Path(out).name;mode=current.get("case","success")
        requested=(kwargs["wall_seconds"],kwargs["cpu_seconds"])
        expected=(600,600) if name=="report-worker" else entry.CAPS[current["phase"]]
        assert requested==expected
        requested_caps.append((current["phase"],name,requested))
        assert json.loads((Path(out).parent/"checkpoint.json").read_bytes())["status"]=="INCOMPLETE"
        wall=3. if (mode=="cap" and name=="mini-xception") or (mode=="report_cap" and name=="report-worker") else 20
        result=REAL_SUPERVISE(argv,out,wall_seconds=wall,cpu_seconds=10,memory_bytes=384*1024**2,stop_grace=3,ids=kwargs.get("ids",()))
        jobs.append(result);assert result.get("active_processes")==0,result
        return result
    monkeypatch.setattr(entry,"supervise",supervised)
    current={}
    def metadata():return development.metadata_manifest(source,expected_count=18,expected_actors=development.ACTORS[:3],expected_support={c:3 for c in development.LABELS.values()})
    def command(cfg,out,name):return [sys.executable,"-B",str(Path(__file__).resolve()),"--synthetic-child",str(cfg),str(out),name]
    def dispatch(changes=None,*,factory=None):
        current.clear();current.update(deepcopy(config));current.update(changes or {})
        path=root/("SYNTHETIC-config-"+uuid.uuid4().hex+".json");REAL_WRITE(path,current,exclusive=True)
        previous=set(outputs.iterdir())
        try:
            result=entry._orchestrate(current,path,outputs,manifest_factory=factory or metadata,command_builder=command)
        except BaseException as exc:
            created=set(outputs.iterdir())-previous
            status="INCOMPLETE" if created else "BLOCKED"
            if not created:labelled(root/("SYNTHETIC-BLOCKED-"+uuid.uuid4().hex+".json"),{"status":"BLOCKED","error":str(exc)},exclusive=True)
            EVIDENCE.append({"case":current.get("case","gate/collision"),"status":status,"diagnostic":str(exc),"outputs":[str(p) for p in created],"jobs":deepcopy(jobs)})
            raise
        created=set(outputs.iterdir())-previous;assert len(created)==1
        out=created.pop();EVIDENCE.append({"case":current.get("case","success"),"phase":current["phase"],"status":json.loads((out/"checkpoint.json").read_bytes())["status"],"out":str(out),"jobs":deepcopy(jobs)})
        return out
    return SimpleNamespace(root=root,source=source,outputs=outputs,config=config,dispatch=dispatch,jobs=jobs,metadata=metadata,requested_caps=requested_caps)


def check_receipts(out):
    entries=(out/"receipts.sha256").read_text(encoding="utf-8").splitlines()
    listed=set()
    for line in entries:
        digest,name=line.split("  ",1);listed.add(name)
        assert entry.sha_file(out/name)==digest
    expected={str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p.name not in ("receipts.sha256","COMPLETE")}
    assert listed==expected


def test_prepare_pilot_full_report_job_and_complete_last(rehearsal,monkeypatch):
    original=Path.open;markers=[]
    def opened(path,mode="r",*args,**kwargs):
        if path.name=="COMPLETE" and mode=="x":
            check_receipts(path.parent)
            assert (path.parent/"bootstrap.json").is_file() and all(j["active_processes"]==0 for j in rehearsal.jobs)
            assert json.loads((path.parent/"checkpoint.json").read_bytes())["status"]=="COMPLETE"
            markers.append(path)
        return original(path,mode,*args,**kwargs)
    monkeypatch.setattr(Path,"open",opened)
    assert entry.CAPS=={"pilot":(600,300),"full":(3600,2700)}
    pilot=rehearsal.dispatch({"phase":"pilot","pilot_accepted":False})
    assert re.fullmatch(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{32}",pilot.name)
    assert json.loads((pilot/"manifest.json").read_bytes())["count"]==12
    assert not (pilot/"COMPLETE").exists() and not (pilot/"metrics.json").exists()
    for name in ("mini-xception","vit"):
        model=json.loads((pilot/name/"model-result.json").read_bytes())["results"][name]
        assert model["metrics"] is None and all("probabilities" not in item for item in model["items"])
    full=rehearsal.dispatch()
    assert full!=pilot and len(markers)==1 and markers[0]==full/"COMPLETE"
    metrics=json.loads((full/"metrics.json").read_bytes());bootstrap=json.loads((full/"bootstrap.json").read_bytes())
    assert metrics["vit"]["n"]==18 and metrics["vit"]["coverage"]==pytest.approx(16/18)
    assert bootstrap["replicates"]==1000 and bootstrap["actor_count"]==3
    assert bootstrap["paired_macro_f1_difference_ci"]["mini-xception minus vit"]==[0.,0.]
    rows=[json.loads(line) for line in (full/"per-clip.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows)==36 and {r["status"] for r in rows}=={"predicted","no_face","decode_failure"}
    assert all(row["synthetic_only"] and row["exposure"]==MARKER for row in rows)
    assert all(j["status"]=="COMPLETE" and j["active_processes"]==0 for j in rehearsal.jobs)
    check_receipts(full)
    assert rehearsal.requested_caps==[("pilot",n,(600,300)) for n in ("mini-xception","vit")]+[("full",n,(3600,2700)) for n in ("mini-xception","vit")]+[("full","report-worker",(600,600))]


@pytest.mark.parametrize("case",["worker_crash","cap","corrupt_progress","adapter_crash","crop_mismatch","missing_rows","report_cap"])
def test_actual_job_failures_preserve_incomplete_and_no_complete(rehearsal,case):
    with pytest.raises((RuntimeError,ValueError)):
        rehearsal.dispatch({"case":case})
    out=next(rehearsal.outputs.iterdir());assert not (out/"COMPLETE").exists()
    assert json.loads((out/"checkpoint.json").read_bytes())["status"]=="INCOMPLETE"
    assert rehearsal.jobs and all(j["active_processes"]==0 for j in rehearsal.jobs)
    if case=="worker_crash":
        assert rehearsal.jobs[0]["attempted_ids"] and len(rehearsal.jobs[0]["unprocessed_ids"])==17
        assert "intentional worker crash" in rehearsal.jobs[0]["worker_failure"]["error"]
    if case=="corrupt_progress":assert rehearsal.jobs[0]["error"]=="invalid worker progress"
    if case in ("cap","report_cap"):assert any(j["status"]=="INCOMPLETE" and "cap" in j["reason"] for j in rehearsal.jobs)
    if case=="adapter_crash":
        assert "original adapter diagnostic" in rehearsal.jobs[0]["worker_failure"]["error"]
        ids=[r["id"] for r in json.loads((out/"manifest.json").read_bytes())["rows"]]
        assert rehearsal.jobs[0]["attempted_ids"]==ids[:1]
        assert rehearsal.jobs[0]["unprocessed_ids"]==ids[1:]
        assert len(rehearsal.jobs)==1
    if case=="missing_rows":
        assert "cannot truncate manifest" in json.loads((out/"checkpoint.json").read_bytes())["error"]
        assert len(rehearsal.jobs)==1
        manifest=json.loads((out/"manifest.json").read_bytes())
        assert manifest["count"]==len(manifest["rows"])==18
        assert not (out/"vit").exists()
    if case=="crop_mismatch":assert "paired source/crop identity mismatch" in rehearsal.jobs[-1]["worker_failure"]["error"]


@pytest.mark.parametrize("kind",["rights_disposition","source_review","command_run_order","asset_missing","asset_digest_missing","asset_digest_bad","source_drift","pilot_not_accepted"])
def test_rehearsal_gate_blocks_before_output_or_child(rehearsal,kind):
    change={kind:None} if kind in ("rights_disposition","source_review","command_run_order") else {"assets":rehearsal.config["assets"][:-1]} if kind=="asset_missing" else {"runtime_source_sha256":"0"*64} if kind=="source_drift" else {"pilot_accepted":False}
    if kind in ("asset_digest_missing","asset_digest_bad"):
        assets=deepcopy(rehearsal.config["assets"])
        if kind=="asset_digest_missing":assets[0].pop("sha256")
        else:assets[0]["sha256"]="0"*64
        change={"assets":assets}
    with pytest.raises(ValueError):rehearsal.dispatch(change)
    assert not list(rehearsal.outputs.iterdir()) and not rehearsal.jobs


@pytest.mark.parametrize("name",["0000_DFA_ANG_XX.flv","malformed.flv"])
def test_excluded_or_malformed_before_array_preparation(protected,name):
    prepare=Mock(side_effect=AssertionError("array preparation reached"))
    manifest={"protocol_sha256":development.PROTOCOL_SHA256,"rows":[{"id":name,"actor":1001,"label":"angry"}]}
    with pytest.raises(ValueError):entry._worker_core({"phase":"full"},protected,"mini-xception",manifest,FakeAdapter("mini-xception","success"),prepare,cold_ms=0,environment={})
    prepare.assert_not_called()


def test_duplicate_ids_rejected_before_preparation(protected):
    row={"id":"1001_DFA_ANG_XX.flv","actor":1001,"label":"angry"};prepare=Mock()
    with pytest.raises(runner.BenchmarkContractError,match="unique clip"):
        entry._worker_core({"phase":"full"},protected,"mini-xception",{"protocol_sha256":development.PROTOCOL_SHA256,"rows":[row,row]},FakeAdapter("mini-xception","success"),prepare,cold_ms=0,environment={})
    prepare.assert_not_called()


def test_real_asset_roster_and_source_paths_refused_by_rehearsal(protected):
    for path in (*entry.required_asset_paths(),development.CANONICAL_SOURCE):
        with pytest.raises(ValueError,match="real/outside paths refused"):synthetic_path(path,protected)


def test_production_entrypoints_cannot_use_test_authority(protected,monkeypatch):
    path=protected/"SYNTHETIC-config.json";path.write_text(json.dumps({"synthetic_only":True}),encoding="ascii")
    forbidden=Mock(side_effect=AssertionError("post-gate execution reached"))
    monkeypatch.setattr(entry,"_orchestrate",forbidden)
    with pytest.raises(ValueError,match="later explicit rights"):entry.main(["--config",str(path)])
    monkeypatch.undo()  # restore only this test's explicit substitutions; filesystem guards are context-managed
    monkeypatch.setattr(entry,"exclusive_output",forbidden)
    with pytest.raises(ValueError,match="not production authority"):entry.main(["--config",str(path),"--execute-authorized"])
    monkeypatch.setattr(entry,"require_worker_job",lambda:None)  # test gate order only, no actual production worker launched
    monkeypatch.setattr(entry,"publish_report",forbidden)
    with pytest.raises(ValueError,match="not production authority"):entry.main(["--config",str(path),"--worker","report","--out",str(protected)])
    forbidden.assert_not_called()


@pytest.mark.parametrize("gate",["rights_disposition","source_review","command_run_order"])
def test_production_requires_each_evidence_field_before_roster(gate,monkeypatch):
    config={"protocol_sha256":development.PROTOCOL_SHA256,"roster":entry.ROSTER,"phase":"pilot"}
    for key in ("rights_disposition","source_review","command_run_order"):config[key]={"path":AUTH,"sha256":"a"*64}
    config[gate]=None
    monkeypatch.setattr(entry,"verify_assets",lambda records:{})  # schema-only test; never validate manufactured permission as real
    roster=Mock(side_effect=AssertionError("roster reached without all evidence"));monkeypatch.setattr(entry,"required_asset_paths",roster)
    with pytest.raises(ValueError,match="external authority evidence missing"):REAL_VALIDATE(config)
    roster.assert_not_called()



def production_shape():
    # Schema-only manufactured references; never permission or real asset access.
    config={"protocol_sha256":development.PROTOCOL_SHA256,"roster":deepcopy(entry.ROSTER),"phase":"pilot","assets":[]}
    for gate in ("rights_disposition","source_review","command_run_order"):
        config[gate]={"path":AUTH,"sha256":"a"*64}
    return config


@pytest.mark.parametrize("kind",["protocol","pilot","asset_roster"])
def test_real_validator_rejects_schema_drift_without_asset_io(monkeypatch,kind):
    config=production_shape();verify=Mock(return_value={})
    monkeypatch.setattr(entry,"verify_assets",verify)
    roster=Mock(return_value={Path("SYNTHETIC-required.txt")})
    monkeypatch.setattr(entry,"required_asset_paths",roster)
    if kind=="protocol":config["protocol_sha256"]="0"*64;message="explicit frozen protocol"
    elif kind=="pilot":config.update(phase="full",pilot_accepted=False);message="reviewed feasible pilot"
    else:message="exact canonical"
    with pytest.raises(ValueError,match=message):REAL_VALIDATE(config)
    assert verify.call_args_list==([] if kind=="protocol" else [(( [config[g]],),{}) for g in ("rights_disposition","source_review","command_run_order")])
    assert roster.call_count==(kind=="asset_roster")


def test_report_worker_rechecks_flag_free_evidence_before_publication(protected,monkeypatch):
    path=protected/"SYNTHETIC-report-config.json"
    config=production_shape();config.pop("rights_disposition")
    path.write_text(json.dumps(config),encoding="ascii")
    monkeypatch.setattr(entry,"require_worker_job",lambda:None)
    forbidden=Mock(side_effect=AssertionError("post-gate IO/publication reached"))
    monkeypatch.setattr(entry,"publish_report",forbidden)
    monkeypatch.setattr(entry,"verify_assets",forbidden)
    with pytest.raises(ValueError,match="external authority evidence missing"):
        entry.main(["--config",str(path),"--worker","report","--out",str(protected)])
    forbidden.assert_not_called()


def test_existing_output_refused_before_any_child(rehearsal,monkeypatch):
    monkeypatch.setattr(entry.time,"strftime",lambda *a:"20000101T000000Z")
    monkeypatch.setattr(entry.uuid,"uuid4",lambda:SimpleNamespace(hex="f"*32))
    existing=entry.exclusive_output(rehearsal.outputs/("20000101T000000Z-"+"f"*32))
    before=(existing/"checkpoint.json").read_bytes()
    with pytest.raises(FileExistsError):rehearsal.dispatch()
    assert (existing/"checkpoint.json").read_bytes()==before
    assert json.loads((existing/"checkpoint.json").read_bytes())["status"]=="INCOMPLETE" and not rehearsal.jobs


def test_interruption_at_final_marker_retains_incomplete(rehearsal,monkeypatch):
    original=Path.open
    def opened(path,mode="r",*args,**kwargs):
        if path.name=="COMPLETE" and mode=="x":raise KeyboardInterrupt("SYNTHETIC final-publication interruption")
        return original(path,mode,*args,**kwargs)
    monkeypatch.setattr(Path,"open",opened)
    with pytest.raises(KeyboardInterrupt):rehearsal.dispatch()
    out=next(rehearsal.outputs.iterdir())
    assert (out/"receipts.sha256").is_file() and not (out/"COMPLETE").exists()
    assert json.loads((out/"checkpoint.json").read_bytes())["status"]=="INCOMPLETE"
    assert all(j["active_processes"]==0 for j in rehearsal.jobs)


if __name__=="__main__":
    if len(sys.argv)!=5 or sys.argv[1]!="--synthetic-child":raise SystemExit("test-only synthetic child; no production invocation")
    child(Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4])
