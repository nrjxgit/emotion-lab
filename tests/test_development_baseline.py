"""Offline manufactured assets and inert Windows workers, no real baseline."""
from pathlib import Path
from types import SimpleNamespace
import contextlib
import ctypes
import hashlib
import json
import os
import sys
import time
from unittest.mock import Mock
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training import run_development_baseline as entry
from training.cremad_development import ACTORS,PROTOCOL_SHA256
from emolab.adapters.vit import ViTAdapter,FER7
@pytest.fixture
def scratch(tmp_path):return tmp_path
@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    def deny(*a,**k):raise AssertionError("unmocked network forbidden")
    monkeypatch.setattr(socket.socket,"connect",deny);monkeypatch.setattr(socket,"create_connection",deny)

def assets(scratch):
    (scratch/"config.json").write_text("{}",encoding="ascii")
    (scratch/"model.safetensors").write_bytes(b"SYNTHETIC NOT WEIGHTS")
    proc={"size":{"height":224,"width":224},"resample":2,"do_resize":True,"do_rescale":True,"rescale_factor":1/255,"do_normalize":True,"image_mean":[.5]*3,"image_std":[.5]*3}
    (scratch/"preprocessor_config.json").write_text(json.dumps(proc),encoding="ascii")
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in scratch.iterdir()}

def test_no_model_load_import_and_missing_local():
    a=ViTAdapter()
    with pytest.raises(ValueError,match="explicit pinned local"):a._ensure_loaded()
    assert a._model is None

def test_vit_offline_cpu_eval_inference_spies(scratch,monkeypatch):
    pins=assets(scratch);tensor=SimpleNamespace();tensor.to=Mock(return_value=tensor)
    proc=Mock(return_value={"pixels":tensor})
    model=Mock();model.to=Mock(return_value=model);model.eval=Mock(return_value=model);model.config=SimpleNamespace(id2label=dict(enumerate(FER7)));model.return_value=SimpleNamespace(logits=[object()])
    processor_loader=SimpleNamespace(from_pretrained=Mock(return_value=proc));model_loader=SimpleNamespace(from_pretrained=Mock(return_value=model))
    probability=SimpleNamespace();probability.detach=lambda:probability;probability.cpu=lambda:probability;probability.numpy=lambda:np.array([1.,0,0,0,0,0,0])
    torch=SimpleNamespace(inference_mode=Mock(side_effect=contextlib.nullcontext),softmax=Mock(return_value=probability))
    monkeypatch.setitem(sys.modules,"transformers",SimpleNamespace(AutoImageProcessor=processor_loader,AutoModelForImageClassification=model_loader));monkeypatch.setitem(sys.modules,"torch",torch)
    a=ViTAdapter(local_path=str(scratch),asset_sha256=pins);result=a.predict_proba(np.zeros((2,2,3),np.uint8))
    for loader in (processor_loader,model_loader):
        args,kwargs=loader.from_pretrained.call_args;assert Path(args[0]).is_absolute() and kwargs["local_files_only"] is True and kwargs["trust_remote_code"] is False
    assert model_loader.from_pretrained.call_args.kwargs["use_safetensors"] is True
    model.to.assert_called_once_with("cpu");model.eval.assert_called_once();tensor.to.assert_called_with("cpu");torch.inference_mode.assert_called_once()
    assert result.top1=="angry" and result.metadata.provenance["verified_assets"]==pins
    a.predict_proba(np.zeros((2,2,3),np.uint8));model_loader.from_pretrained.assert_called_once()

@pytest.mark.parametrize("kind",["missing","hash","processor","labels","nan","zero","duplicate"])
def test_vit_fails_closed(scratch,kind):
    if kind in ("labels","duplicate","nan","zero"):
        labels=FER7 if kind not in ("labels","duplicate") else (FER7[:-1]+["calm"] if kind=="labels" else ["angry"]*7)
        values=np.zeros(7) if kind=="zero" else np.full(7,np.nan)
        a=ViTAdapter(id2label=tuple(labels),forward_fn=lambda rgb:values)
        with pytest.raises(ValueError):a.predict_proba(np.zeros((2,2,3),np.uint8))
    else:
        pins=assets(scratch)
        if kind=="missing":pins.pop("config.json")
        elif kind=="hash":pins["model.safetensors"]="0"*64
        else:
            p=scratch/"preprocessor_config.json";cfg=json.loads(p.read_bytes());cfg["image_std"]=[1.]*3;p.write_text(json.dumps(cfg),encoding="ascii");pins[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        with pytest.raises(ValueError):ViTAdapter(local_path=str(scratch),asset_sha256=pins)._ensure_loaded()

def test_config_and_asset_failure_before_inference(scratch):
    with pytest.raises(ValueError):entry.validate_config({})
    p=scratch/"SYNTHETIC.txt";p.write_bytes(b"fixture")
    with pytest.raises(ValueError):entry.verify_assets([{"path":str(p),"sha256":"0"*64}])
    with pytest.raises(ValueError):entry.main(["--config",str(p)])  # malformed JSON, no model factory called

def test_output_collision_incomplete_and_pilot_order(scratch):
    out=entry.exclusive_output(scratch/"result");assert json.loads((out/"checkpoint.json").read_bytes())["status"]=="INCOMPLETE"
    with pytest.raises(FileExistsError):entry.exclusive_output(out)
    rows=[{"id":f"{actor}-{label}","actor":actor,"label":label} for actor in ACTORS[:2] for label in ("angry","disgust","fear","happy","neutral","sad")]
    assert entry.pilot_rows(rows)==rows and len(rows)==12
    with pytest.raises(RuntimeError):entry.require_worker_job()

def test_audit_rejects_partial_and_missing_pair():
    with pytest.raises((KeyError,ValueError)):entry.audit_results({}, {"rows":[]})

@pytest.mark.skipif(os.name!="nt",reason="native Windows Job test, never advisory fallback")
@pytest.mark.parametrize("mode",["sleep_tree","memory","cpu","success","orphan_exit"])
def test_native_windows_caps_and_no_orphan(scratch,mode):
    marker=scratch/"SYNTHETIC-descendant-pid.txt"
    if mode=="sleep_tree":
        code=f"import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(60)']);Path({str(marker)!r}).write_text(str(p.pid));time.sleep(60)";wall=1.;cpu=5;memory=128*1024**2
    elif mode=="orphan_exit":
        code="import subprocess,sys;subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(60)'])";wall=5;cpu=5;memory=128*1024**2
    elif mode=="memory":
        # Bounded 192 MiB target exceeds the 48 MiB limit by 144 MiB.
        code="import ctypes,sys\nk=ctypes.WinDLL('kernel32',use_last_error=True)\nk.VirtualAlloc.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_uint32,ctypes.c_uint32]\nk.VirtualAlloc.restype=ctypes.c_void_p\na=[]\nfor _ in range(192):\n p=k.VirtualAlloc(None,1024*1024,0x3000,4)\n if not p:sys.exit(23)\n a.append(p)"
        wall=5;cpu=5;memory=48*1024**2
    elif mode=="cpu":code="while True: pass";wall=5;cpu=1;memory=128*1024**2
    else:code="pass";wall=5;cpu=5;memory=128*1024**2
    result=entry.supervise([sys.executable,"-B","-c",code],scratch/"job",wall_seconds=wall,cpu_seconds=cpu,memory_bytes=memory,stop_grace=3,ids=["fixture-clip"])
    assert result.get("active_processes")==0,result
    assert result["status"]==("COMPLETE" if mode=="success" else "INCOMPLETE"),result
    assert result["elapsed_seconds"]<wall+4 and result["unprocessed_ids"]==["fixture-clip"]
    assert json.loads((scratch/"job/execution.json").read_bytes())["status"]==result["status"]
    if mode in ("memory","cpu"):
        assert result["elapsed_seconds"]<wall-1,result
        assert result["exit_code"]!=0,result
        # Job commit-time accounting may include a small rejected-allocation overshoot.
        # Without the memory limit the worker reaches 192 MiB and exits zero:
        # the nonzero-exit assertion or this limit + 8 MiB check must fail.
        tolerance=8*1024**2 if mode=="memory" else 0
        assert 0<result["peak_job_memory_bytes"]<=memory+tolerance,result
        if mode=="memory":assert result["reason"]=="worker/cap failure",result
        else:
            assert result["reason"] in ("aggregate CPU cap","worker/cap failure"),result
            assert .8<=result["job_user_seconds"]+result["job_kernel_seconds"]<cpu+1,result
    if mode=="sleep_tree":
        assert marker.is_file();pid=int(marker.read_text());k=ctypes.WinDLL("kernel32",use_last_error=True)
        k.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32];k.OpenProcess.restype=ctypes.c_void_p;k.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32];k.CloseHandle.argtypes=[ctypes.c_void_p]
        handle=k.OpenProcess(0x100000,False,pid)
        if handle:
            try:assert k.WaitForSingleObject(handle,0)==0
            finally:k.CloseHandle(handle)


def test_positive_paired_output_audit_and_collision(scratch):
    from training.cremad_development import EXPOSURE
    rows=[{"id":"SYNTHETIC", "actor":1001,"label":"angry","bytes":10,"sha256":"a"*64,"crop_sha256":[],"decoded_frames":0,"selected_frames":0,"probabilities":None,"status":"decode_failure","observed_inference_ms":[]}]
    digest=hashlib.sha256(json.dumps([(r["id"],r["actor"],r["label"],r["bytes"],r["sha256"]) for r in rows],separators=(",",":")).encode()).hexdigest()
    manifest={"rows":rows};reports={n:{"complete":True,"sample_manifest_sha256":digest,"environment":{"synthetic":True},"results":{n:{"complete":True,"unprocessed_ids":[],"items":rows}}} for n in ("mini-xception","vit")}
    entry.publish_report(scratch,manifest,reports)
    assert json.loads((scratch/"metrics.json").read_bytes())["vit"]["accuracy"]==0
    assert json.loads((scratch/"bootstrap.json").read_bytes())["replicates"]==1000
    assert (scratch/"per-clip.jsonl").read_text().count("\n")==2
    assert not (scratch/"COMPLETE").exists()
    with pytest.raises(FileExistsError):entry.publish_report(scratch,manifest,reports)
    reports["vit"]["sample_manifest_sha256"]="0"*64
    with pytest.raises(ValueError,match="manifest hash"):entry.audit_results(reports,manifest)


def test_environment_snapshot_no_secrets():
    if os.name!="nt":pytest.skip("Windows metadata only")
    result=entry.environment_snapshot()
    assert result["ram_bytes"]>0 and result["architecture"] and result["seed"]==505
    assert "package_versions" in result and "environment" not in result


def test_make_mini_uses_numpy_adapter_and_pinned_loader(monkeypatch):
    from backend.app.face import fer_model
    from emolab.adapters.mini_xception import MiniXceptionAdapter
    path=ROOT/"models/fer_mini_xception_h5.hdf5"
    weights=object();loader=Mock(return_value=weights)
    predictor=Mock(return_value=np.array([1.,0,0,0,0,0,0]))
    monkeypatch.setattr(fer_model,"load_fer",loader);monkeypatch.setattr(fer_model,"predict_fer",predictor)
    adapter=entry.make_adapter("mini-xception",{path:"a"*64})
    assert isinstance(adapter,MiniXceptionAdapter)
    result=adapter.predict_proba(np.zeros((2,2,3),np.uint8))
    loader.assert_called_once_with(str(path));predictor.assert_called_once()
    tensor,used_weights=predictor.call_args.args
    assert used_weights is weights and tensor.shape==(64,64,1) and np.all(tensor==-1)
    assert result.top1=="angry" and result.metadata.provenance["sha256"]=="a"*64
    for source in ("emolab/adapters/mini_xception.py","backend/app/face/fer_model.py","backend/app/face/minixception_numpy.py"):
        assert ROOT/source in entry.required_asset_paths()
    with pytest.raises(ValueError,match="verified mini"):entry.make_adapter("mini-xception",{})


def test_progress_replace_is_atomic(scratch,monkeypatch):
    path=scratch/"worker-progress.json"
    previous={"attempted_ids":[],"unprocessed_ids":["a","b"]}
    latest={"attempted_ids":["a"],"unprocessed_ids":["b"]}
    entry.write_progress(path,previous);replace=os.replace;calls=[]
    def spy(source,target):
        assert json.loads(path.read_bytes())==previous
        assert json.loads(Path(source).read_bytes())==latest
        calls.append((source,target));replace(source,target)
    monkeypatch.setattr(os,"replace",spy);entry.write_progress(path,latest)
    assert len(calls)==1 and json.loads(path.read_bytes())==latest


@pytest.mark.skipif(os.name!="nt",reason="native Windows Job required")
def test_truncated_progress_keeps_failure_receipt_and_ids(scratch):
    out=scratch/"job"
    code=f"from pathlib import Path;Path({str(out/'worker-progress.json')!r}).write_text('{{\"attempted_ids\":[')"
    result=entry.supervise([sys.executable,"-B","-c",code],out,wall_seconds=5,cpu_seconds=3,memory_bytes=128*1024**2,ids=["a","b"])
    assert result["status"]=="INCOMPLETE" and result["error"]=="invalid worker progress"
    assert result["attempted_ids"]==[] and result["unprocessed_ids"]==["a","b"]
    assert json.loads((out/"execution.json").read_bytes())==result
    assert json.loads((out/"checkpoint.json").read_bytes())==result


def test_preinference_worker_failure_persists_reason(scratch,monkeypatch):
    config=scratch/"config.json";config.write_text("{}",encoding="ascii")
    monkeypatch.setattr(entry,"require_worker_job",lambda:None)
    def fail(*args):raise ValueError("SYNTHETIC pre-inference pin mismatch")
    monkeypatch.setattr(entry,"worker",fail)
    with pytest.raises(ValueError,match="pre-inference"):
        entry.main(["--config",str(config),"--worker","mini-xception","--out",str(scratch)])
    reason=json.loads((scratch/"worker-failure.json").read_bytes())
    assert reason=={"status":"INCOMPLETE","error":"ValueError: SYNTHETIC pre-inference pin mismatch"}


def test_legacy_fer_vit_route_fails_closed_without_directory():
    from training.evaluate_models import _construct_adapter
    # Same default factory path used by the existing FER CLI.
    adapter=_construct_adapter("vit")
    with pytest.raises(ValueError,match="explicit pinned local ViT directory required"):
        adapter.predict_proba(np.zeros((2,2,3),np.uint8))


@pytest.mark.skipif(os.name!="nt",reason="native Windows Job required")
@pytest.mark.parametrize("content",['{', '[]'])
def test_malformed_worker_failure_preserves_receipt_and_ids(scratch,content):
    out=scratch/"job"
    code=f"from pathlib import Path;Path({str(out/'worker-failure.json')!r}).write_text({content!r})"
    result=entry.supervise([sys.executable,"-B","-c",code],out,wall_seconds=5,cpu_seconds=3,memory_bytes=128*1024**2,ids=["a","b"])
    assert result["status"]=="INCOMPLETE" and result["error"]=="invalid worker failure receipt"
    assert result["attempted_ids"]==[] and result["unprocessed_ids"]==["a","b"]
    assert json.loads((out/"execution.json").read_bytes())==result
    assert json.loads((out/"checkpoint.json").read_bytes())==result
