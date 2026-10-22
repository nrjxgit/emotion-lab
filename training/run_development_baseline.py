"""Explicit opt-in development baseline. No asset/data IO or loading at import.

Stage A is source/synthetic only. Real invocation requires separate R2/source
review/full authoritative pins/Command run authority; flags do not prove intent.
Windows Job limits are mandatory; no advisory/unlocked fallback.
"""
from __future__ import annotations
import argparse
import ctypes as c
from ctypes import wintypes as w
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from training.cremad_development import ACTORS, CANONICAL_SOURCE, EXPOSURE, PROTOCOL, PROTOCOL_SHA256, metadata_manifest, safe_path, sha_file

ROSTER={"mini-xception":"PRIMARY","vit":"PRIMARY","hse":"BLOCKED: local asset authority unavailable","combined":"BLOCKED: embedding/access identity unresolved"}
CAPS={"pilot":(600,300),"full":(3600,2700)}
MEMORY_CAP=4*1024**3

def write_json(path,obj,*,exclusive=False):
    with Path(path).open("x" if exclusive else "w",encoding="utf-8",newline="\n") as f:
        json.dump(obj,f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())

def write_progress(path, obj):
    """Commit a complete progress snapshot atomically on the same volume."""
    path=Path(path);temporary=path.with_name(path.name+".tmp")
    write_json(temporary,obj)
    os.replace(temporary,path)


def exclusive_output(path):
    path=safe_path(path);path.mkdir(exist_ok=False)
    write_json(path/"checkpoint.json",{"status":"INCOMPLETE","exposure":EXPOSURE},exclusive=True)
    return path

class _Basic(c.Structure):
    _fields_=[("process_time",c.c_int64),("job_time",c.c_int64),("flags",w.DWORD),("min_ws",c.c_size_t),("max_ws",c.c_size_t),("active_limit",w.DWORD),("affinity",c.c_size_t),("priority",w.DWORD),("scheduling",w.DWORD)]
class _IO(c.Structure):
    _fields_=[(x,c.c_uint64) for x in ("read_ops","write_ops","other_ops","read_bytes","write_bytes","other_bytes")]
class _Extended(c.Structure):
    _fields_=[("basic",_Basic),("io",_IO),("process_memory",c.c_size_t),("job_memory",c.c_size_t),("peak_process",c.c_size_t),("peak_job",c.c_size_t)]
class _Accounting(c.Structure):
    _fields_=[("user",c.c_int64),("kernel",c.c_int64),("period_user",c.c_int64),("period_kernel",c.c_int64),("faults",w.DWORD),("total",w.DWORD),("active",w.DWORD),("terminated",w.DWORD)]
class _Startup(c.Structure):
    _fields_=[("cb",w.DWORD),("reserved",w.LPWSTR),("desktop",w.LPWSTR),("title",w.LPWSTR)]+[(x,w.DWORD) for x in ("x","y","xsize","ysize","xchars","ychars","fill","flags")]+[("show",w.WORD),("reserved2_size",w.WORD),("reserved2",c.c_void_p)]+[(x,w.HANDLE) for x in ("stdin","stdout","stderr")]
class _Process(c.Structure):
    _fields_=[("process",w.HANDLE),("thread",w.HANDLE),("pid",w.DWORD),("tid",w.DWORD)]

def supervise(argv,out,*,wall_seconds,cpu_seconds,memory_bytes=MEMORY_CAP,stop_grace=10,ids=()):
    """Create SUSPENDED, assign capped Job BEFORE resuming, contain descendants.

    Job memory/CPU enforcement is kernel-backed; wall cap is parent monotonic.
    Kill-on-close and TerminateJobObject, <=10 s grace, no tree fallback/retry.
    Synthetic tests may pass smaller limits/independent inert Python argv.
    """
    if os.name!="nt":raise RuntimeError("verified Windows Job API required; STOP")
    if not (0<wall_seconds<=3600 and 0<cpu_seconds<=2700 and 0<memory_bytes<=MEMORY_CAP and 0<stop_grace<=10):raise ValueError("invalid execution cap")
    out=exclusive_output(out);start=time.monotonic();k=c.WinDLL("kernel32",use_last_error=True)
    k.CreateJobObjectW.argtypes=[c.c_void_p,w.LPCWSTR];k.CreateJobObjectW.restype=w.HANDLE
    k.SetInformationJobObject.argtypes=[w.HANDLE,c.c_int,c.c_void_p,w.DWORD];k.SetInformationJobObject.restype=w.BOOL
    k.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE];k.AssignProcessToJobObject.restype=w.BOOL
    k.CreateProcessW.argtypes=[w.LPCWSTR,w.LPWSTR,c.c_void_p,c.c_void_p,w.BOOL,w.DWORD,c.c_void_p,w.LPCWSTR,c.POINTER(_Startup),c.POINTER(_Process)];k.CreateProcessW.restype=w.BOOL
    k.ResumeThread.argtypes=[w.HANDLE];k.ResumeThread.restype=w.DWORD
    k.GetExitCodeProcess.argtypes=[w.HANDLE,c.POINTER(w.DWORD)];k.GetExitCodeProcess.restype=w.BOOL
    k.TerminateJobObject.argtypes=[w.HANDLE,w.UINT];k.TerminateJobObject.restype=w.BOOL
    k.TerminateProcess.argtypes=[w.HANDLE,w.UINT];k.TerminateProcess.restype=w.BOOL
    k.QueryInformationJobObject.argtypes=[w.HANDLE,c.c_int,c.c_void_p,w.DWORD,c.c_void_p];k.QueryInformationJobObject.restype=w.BOOL
    k.CloseHandle.argtypes=[w.HANDLE];k.CloseHandle.restype=w.BOOL
    def require(ok):
        if not ok:raise OSError(c.get_last_error(),"Windows containment API failed")
    job_token="EL003-O08-"+uuid.uuid4().hex
    job=k.CreateJobObjectW(None,job_token);require(job);pi=_Process();assigned=False
    result={"status":"INCOMPLETE","argv":list(argv),"wall_seconds":wall_seconds,"cpu_seconds":cpu_seconds,"memory_bytes":memory_bytes,"attempted_ids":[],"unprocessed_ids":list(ids),"exposure":EXPOSURE}
    try:
        limits=_Extended();limits.basic.flags=0x2000|0x200|0x4;limits.basic.job_time=int(cpu_seconds*10000000);limits.job_memory=memory_bytes
        require(k.SetInformationJobObject(job,9,c.byref(limits),c.sizeof(limits)))
        startup=_Startup();startup.cb=c.sizeof(startup)
        env=dict(os.environ);env["EL003_O08_JOB_TOKEN"]=job_token;env["CUDA_VISIBLE_DEVICES"]="-1"
        env.update({x:"1" for x in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","TF_NUM_INTRAOP_THREADS","TF_NUM_INTEROP_THREADS","PYTHONDONTWRITEBYTECODE")})
        envbuf=c.create_unicode_buffer("\0".join(f"{x}={v}" for x,v in sorted(env.items()))+"\0\0")
        cmd=c.create_unicode_buffer(subprocess.list2cmdline([str(x) for x in argv]))
        require(k.CreateProcessW(None,cmd,None,None,False,0x4|0x400|0x08000000,envbuf,None,c.byref(startup),c.byref(pi)))
        result["pid"]=pi.pid
        require(k.AssignProcessToJobObject(job,pi.process));assigned=True
        require(k.ResumeThread(pi.thread)!=0xFFFFFFFF)
        next_checkpoint=start+300
        while True:
            code=w.DWORD();require(k.GetExitCodeProcess(pi.process,c.byref(code)))
            if code.value!=259:
                result["exit_code"]=code.value;result["reason"]="exit" if code.value==0 else "worker/cap failure";break
            usage=_Accounting();require(k.QueryInformationJobObject(job,1,c.byref(usage),c.sizeof(usage),None))
            if (usage.user+usage.kernel)/1e7>=cpu_seconds:result["reason"]="aggregate CPU cap";break
            if time.monotonic()-start>=wall_seconds:result["reason"]="parent wall cap";break
            if time.monotonic()>=next_checkpoint:
                write_json(out/"supervisor-checkpoint.json",{**result,"elapsed":time.monotonic()-start});next_checkpoint+=300
            time.sleep(.02)
        # Root exit alone is insufficient: any remaining descendant is an error.
        info=_Accounting();require(k.QueryInformationJobObject(job,1,c.byref(info),c.sizeof(info),None))
        settle_deadline=time.monotonic()+min(.25,stop_grace)
        while info.active and result.get("exit_code")==0 and time.monotonic()<settle_deadline:
            time.sleep(.01);require(k.QueryInformationJobObject(job,1,c.byref(info),c.sizeof(info),None))
        if info.active and result.get("exit_code")==0:result["reason"]="orphan descendant rejected"
        require(k.TerminateJobObject(job,1));deadline=time.monotonic()+stop_grace
        while True:
            require(k.QueryInformationJobObject(job,1,c.byref(info),c.sizeof(info),None))
            if info.active==0:break
            if time.monotonic()>=deadline:raise RuntimeError("Job did not drain within stop grace")
            time.sleep(.02)
        peak=_Extended();require(k.QueryInformationJobObject(job,9,c.byref(peak),c.sizeof(peak),None))
        code=w.DWORD();require(k.GetExitCodeProcess(pi.process,c.byref(code)))
        result["exit_code"]=code.value
        result.update(active_processes=info.active,job_user_seconds=info.user/1e7,job_kernel_seconds=info.kernel/1e7,peak_job_memory_bytes=peak.peak_job)
        result["status"]="COMPLETE" if result.get("exit_code")==0 and result["reason"]=="exit" else "INCOMPLETE"
    except BaseException as exc:
        result["error"]=type(exc).__name__+": "+str(exc)
        if pi.process:
            if assigned:k.TerminateJobObject(job,1)
            else:k.TerminateProcess(pi.process,1)  # still suspended, never ran
        result["status"]="INCOMPLETE"
    finally:
        if pi.thread:k.CloseHandle(pi.thread)
        if pi.process:k.CloseHandle(pi.process)
        k.CloseHandle(job)
        result["elapsed_seconds"]=time.monotonic()-start
        progress=out/"worker-progress.json"
        if progress.is_file():
            try:
                data=json.loads(progress.read_bytes());attempted=data.get("attempted_ids",[])
                if not isinstance(attempted,list) or any(not isinstance(x,str) for x in attempted) or len(set(attempted))!=len(attempted) or not set(attempted)<=set(ids):
                    raise ValueError("invalid worker progress")
                result.update(attempted_ids=attempted,unprocessed_ids=[x for x in ids if x not in attempted])
            except (ValueError,AttributeError,TypeError):
                result["status"]="INCOMPLETE";result["error"]="invalid worker progress"
        failure=out/"worker-failure.json"
        if failure.is_file():
            try:
                data=json.loads(failure.read_bytes())
                if not isinstance(data,dict):raise ValueError("failure receipt must be an object")
                result["worker_failure"]=data
            except (ValueError,TypeError):
                result["status"]="INCOMPLETE";result["error"]="invalid worker failure receipt"
        write_json(out/"execution.json",result,exclusive=True)
        write_json(out/"checkpoint.json",result)
    return result

def verify_assets(records):
    if not records:raise ValueError("full authoritative asset manifest required")
    verified={}
    for record in records:
        path=safe_path(record["path"]);expected=record["sha256"]
        if len(expected)!=64 or any(x not in "0123456789abcdef" for x in expected):raise ValueError("full SHA256 required")
        if path in verified or sha_file(path)!=expected:raise ValueError("asset pin mismatch/duplicate")
        verified[path]=expected
    return verified

def pilot_rows(rows):
    from training.cremad_development import LABELS
    return [next(r for r in rows if r["actor"]==actor and r["label"]==label) for actor in ACTORS[:2] for label in sorted(LABELS.values())]

def validate_config(config):
    if config.get("synthetic_only") or config.get("design_only") or config.get("template_status"):
        raise ValueError("synthetic/dormant config is not production authority")
    if config.get("protocol_sha256")!=PROTOCOL_SHA256 or config.get("roster")!=ROSTER or config.get("phase") not in CAPS:
        raise ValueError("explicit frozen protocol/roster/phase required")
    for gate in ("rights_disposition","source_review","command_run_order"):
        item=config.get(gate)
        if not isinstance(item,dict) or not item.get("path") or not item.get("sha256"):raise ValueError("external authority evidence missing")
        verify_assets([item])
    if config["phase"]=="full" and config.get("pilot_accepted") is not True:raise ValueError("reviewed feasible pilot required before full")
    records=config.get("assets",[])
    if {Path(r["path"]) for r in records}!=required_asset_paths():raise ValueError("exact canonical source/model/processor asset roster required")
    return verify_assets(records)

def require_worker_job():
    if os.name!="nt":raise RuntimeError("Windows capped worker only")
    token=os.environ.get("EL003_O08_JOB_TOKEN","")
    if not token.startswith("EL003-O08-"):raise RuntimeError("no supervisor Job token")
    k=c.WinDLL("kernel32",use_last_error=True)
    k.OpenJobObjectW.argtypes=[w.DWORD,w.BOOL,w.LPCWSTR];k.OpenJobObjectW.restype=w.HANDLE
    k.GetCurrentProcess.restype=w.HANDLE
    k.IsProcessInJob.argtypes=[w.HANDLE,w.HANDLE,c.POINTER(w.BOOL)];k.IsProcessInJob.restype=w.BOOL
    k.CloseHandle.argtypes=[w.HANDLE]
    job=k.OpenJobObjectW(0x4,False,token)
    try:
        inside=w.BOOL()
        if not job or not k.IsProcessInJob(k.GetCurrentProcess(),job,c.byref(inside)) or not inside.value:raise RuntimeError("worker is not in its supervisor Job")
    finally:
        if job:k.CloseHandle(job)

def make_adapter(name,verified):
    import numpy as np
    from emolab.benchmark.runner import FER7
    root=Path(__file__).resolve().parents[1]
    if name=="vit":
        from emolab.adapters.vit import ViTAdapter,VIT_REVISION
        directory=root/"models/vit"/VIT_REVISION
        pins={p.name:digest for p,digest in verified.items() if p.parent==directory}
        return ViTAdapter(local_path=str(directory),asset_sha256=pins)
    if name!="mini-xception":raise ValueError("blocked roster model")
    from emolab.adapters.mini_xception import MiniXceptionAdapter
    from backend.app.face.fer_model import load_fer,predict_fer
    from dataclasses import replace
    path=root/"models/fer_mini_xception_h5.hdf5"
    if path not in verified:raise ValueError("verified mini weights required")
    class FrozenMini(MiniXceptionAdapter):
        def predict_proba(self,crop):
            result=super().predict_proba(crop)
            vector=np.asarray([result.probabilities[c] for c in FER7])
            if vector.shape!=(7,) or not np.isfinite(vector).all() or (vector<0).any() or not np.isclose(vector.sum(),1,atol=1e-6,rtol=0):raise ValueError("invalid mini output")
            metadata=replace(result.metadata,provenance={**result.metadata.provenance,"weights":str(path),"sha256":verified[path],"device":"cpu"})
            return replace(result,metadata=metadata)
    adapter=FrozenMini();adapter._w=load_fer(str(path));adapter._predict_fer=predict_fer
    return adapter


def required_asset_paths():
    root=Path(__file__).resolve().parents[1]
    return {root/p for p in ("models/fer_mini_xception_h5.hdf5","models/face_detection_yunet_2023mar.onnx",*("models/vit/747cf16692eea925b54b7b543cf436848128b68d/"+n for n in ("config.json","preprocessor_config.json","model.safetensors")),"backend/app/face/detector.py","backend/app/face/preprocessing.py","emolab/adapters/mini_xception.py","backend/app/face/fer_model.py","backend/app/face/minixception_numpy.py","emolab/adapters/vit.py","emolab/benchmark/runner.py","training/cremad_development.py","training/run_development_baseline.py")}

def environment_snapshot():
    import platform
    import importlib.metadata
    class Memory(c.Structure):
        _fields_=[("length",w.DWORD),("load",w.DWORD)]+[(name,c.c_uint64) for name in ("total_phys","available_phys","total_page","available_page","total_virtual","available_virtual","extended_virtual")]
    memory=Memory();memory.length=c.sizeof(memory)
    if os.name!="nt":raise RuntimeError("Windows memory metadata unavailable")
    k=c.WinDLL("kernel32",use_last_error=True);k.GlobalMemoryStatusEx.argtypes=[c.POINTER(Memory)]
    if not k.GlobalMemoryStatusEx(c.byref(memory)):raise OSError("memory metadata unavailable")
    versions={}
    for package in ("numpy","opencv-python","torch","transformers","tensorflow","safetensors"):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]=None
    return {"exposure":EXPOSURE,"cpu":platform.processor(),"architecture":platform.machine(),"os":platform.platform(),"ram_bytes":memory.total_phys,"package_versions":versions,"numerical_threads":1,"seed":505}

def worker(config,out,name):
    require_worker_job();verified=validate_config(config)
    import random
    import numpy as np
    import cv2
    random.seed(505);np.random.seed(505);cv2.setNumThreads(1)
    import torch
    torch.manual_seed(505);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    from training.cremad_development import decode_selected,choose_crop,bind_payload,DecodeFailure
    from emolab.benchmark.runner import run_development_clips
    manifest=json.loads((out.parent/"manifest.json").read_bytes())
    _validate_worker_manifest(manifest)
    detector_path=Path(__file__).resolve().parents[1]/"models/face_detection_yunet_2023mar.onnx"
    detector=cv2.FaceDetectorYN_create(str(detector_path),"",(320,240),.6,.3,5000)
    if detector is None:raise RuntimeError("detector missing; cannot masquerade as no-face")
    started=time.perf_counter();adapter=make_adapter(name,verified)
    # At most three manufactured warmups, outside measured dataset timing.
    warmup=np.zeros((64,64,3),dtype=np.uint8);adapter.predict_proba(warmup)
    cold_ms=(time.perf_counter()-started)*1000
    def prepare(row):
        decode_start=time.perf_counter();bound=bind_payload(CANONICAL_SOURCE,row)
        try:frames,n,bound=decode_selected(CANONICAL_SOURCE,bound)
        except DecodeFailure as exc:return [],{"decoded_frames":0,"selected_frames":0,"sha256":bound["sha256"],"decode_error":str(exc),"decode_detection_crop_ms":(time.perf_counter()-decode_start)*1000}
        crops=[]
        for frame in frames:
            h,wid=frame.shape[:2];detector.setInputSize((wid,h));_,faces=detector.detect(frame)
            boxes=[] if faces is None else [dict(zip(("x","y","w","h"),map(int,f[:4]))) for f in faces]
            crop=choose_crop(frame,boxes)
            if crop is not None:crops.append(crop)
        return crops,{"decoded_frames":n,"selected_frames":len(frames),"sha256":bound["sha256"],"decode_detection_crop_ms":(time.perf_counter()-decode_start)*1000}
    environment={**environment_snapshot(),"python":sys.version,"numpy":np.__version__,"opencv":cv2.__version__,"opencv_build":cv2.getBuildInformation(),"torch":torch.__version__,"cpu_count":os.cpu_count(),"threads":1,"seed":505,"cpu_only":True}
    return _worker_core(config,out,name,manifest,adapter,prepare,cold_ms=cold_ms,environment=environment)


def _validate_worker_manifest(manifest):
    if manifest["protocol_sha256"]!=PROTOCOL_SHA256:raise ValueError("worker manifest protocol drift")
    for row in manifest["rows"]:  # all rows validated before first decode
        from training.cremad_development import parse_name
        if parse_name(row["id"])!=(row["actor"],row["label"]):raise ValueError("manifest admission drift")


def _worker_core(config,out,name,manifest,adapter,prepare,*,cold_ms,environment,clock=time.perf_counter):
    """Post-gate core with prepared dependencies; no loader/codec/root override CLI.

    Production worker checks its Job/config before constructing dependencies.
    Tests inject manufactured adapters/preparation in a test-only child.
    """
    _validate_worker_manifest(manifest)
    from emolab.benchmark.runner import run_development_clips
    def progress(items,remaining):
        write_progress(out/"worker-progress.json",{"attempted_ids":[r["id"] for r in items],"unprocessed_ids":remaining})
    report=run_development_clips([adapter],manifest,prepare,clock=clock,progress=progress,timing_only=config["phase"]=="pilot")
    report["cold_construction_first_call_ms"]=cold_ms;report["warmup_calls"]=1
    report["sample_manifest_sha256"]=hashlib.sha256(json.dumps([(r["id"],r["actor"],r["label"],r["bytes"],r.get("sha256")) for r in report["results"][name]["items"]],separators=(",",":")).encode()).hexdigest()
    report["environment"]=environment
    write_json(out/"model-result.json",report,exclusive=True)
    if not report["complete"]:raise RuntimeError(report["results"][name].get("error") or "model incomplete; no retry")

def audit_results(reports,manifest):
    from emolab.benchmark.runner import development_metrics,development_bootstrap
    results={};reference=None
    for name in ("mini-xception","vit"):
        report=reports[name];model=report["results"][name]
        if not report["complete"] or not model["complete"] or model["unprocessed_ids"]:raise ValueError("incomplete primary model")
        rows=model["items"]
        if [(r["id"],r["actor"],r["label"]) for r in rows]!=[(r["id"],r["actor"],r["label"]) for r in manifest["rows"]]:raise ValueError("cohort audit mismatch")
        binding=[(r["id"],r["sha256"],r["crop_sha256"],r["decoded_frames"],r["selected_frames"]) for r in rows]
        if reference is not None and binding!=reference:raise ValueError("paired source/crop identity mismatch")
        digest=hashlib.sha256(json.dumps([(r["id"],r["actor"],r["label"],r["bytes"],r.get("sha256")) for r in rows],separators=(",",":")).encode()).hexdigest()
        if report.get("sample_manifest_sha256")!=digest or any(r.get("sha256")!=m.get("sha256") for r,m in zip(rows,manifest["rows"])):raise ValueError("payload-bound manifest hash mismatch")
        reference=binding;results[name]=rows
    return {"metrics":{name:development_metrics(rows) for name,rows in results.items()},"bootstrap":development_bootstrap(results)}

def publish_report(out,manifest,reports):
    audited=audit_results(reports,manifest)
    write_json(out/"metrics.json",audited["metrics"],exclusive=True);write_json(out/"bootstrap.json",audited["bootstrap"],exclusive=True)
    with (out/"per-clip.jsonl").open("x",encoding="utf-8") as f:
        for name in ("mini-xception","vit"):
            for row in reports[name]["results"][name]["items"]:f.write(json.dumps({"model":name,"exposure":EXPOSURE,**row},allow_nan=False)+"\n")
    write_json(out/"environment.json",{"models":{name:r["environment"] for name,r in reports.items()},"exposure":EXPOSURE},exclusive=True)
    (out/"report.md").write_text("# "+EXPOSURE+"\nNo ECE, ranking/default selection, promotion or independent-generalization claim. HSE/Combined BLOCKED.\n",encoding="utf-8")

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True);parser.add_argument("--execute-authorized",action="store_true")
    parser.add_argument("--worker",choices=("mini-xception","vit","report"));parser.add_argument("--out",type=Path)
    args=parser.parse_args(argv)
    if args.worker:
        try:
            require_worker_job()
            config=json.loads(args.config.read_bytes())
            if args.worker=="report":
                validate_config(config)  # report workers must recheck the same gates/pins
                manifest=json.loads((args.out.parent/"manifest.json").read_bytes())
                reports={n:json.loads((args.out.parent/n/"model-result.json").read_bytes()) for n in ("mini-xception","vit")}
                publish_report(args.out.parent,manifest,reports)
            else:worker(config,args.out,args.worker)
            return 0
        except BaseException as exc:
            if args.out is not None:
                write_json(safe_path(args.out)/"worker-failure.json",{"status":"INCOMPLETE","error":type(exc).__name__+": "+str(exc)},exclusive=True)
            raise
    config=json.loads(args.config.read_bytes())
    if not args.execute_authorized:raise ValueError("later explicit rights/source/pin/run authority required; no media opened")
    root=Path(__file__).resolve().parents[1]/"eval_results/development-cremad-o08"
    return _orchestrate(config,args.config,root)


def _worker_command(config_path,out,name):
    return [sys.executable,"-B","-m","training.run_development_baseline","--config",str(config_path.resolve()),"--worker",name,"--out",str(out)]


def _orchestrate(config,config_path,output_parent,*,manifest_factory=None,command_builder=None):
    """Private orchestration; production always supplies its fixed root/factories.

    Validation remains mandatory. Test-only module injection supplies labelled
    authority substitutes, tiny manufactured manifests and inert child commands;
    no config field or command-line switch can select these dependencies.
    """
    manifest_factory=metadata_manifest if manifest_factory is None else manifest_factory
    command_builder=_worker_command if command_builder is None else command_builder
    validate_config(config)
    # Parent root must already exist under the later dispatch; never pick another.
    out=exclusive_output(output_parent/(time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())+"-"+uuid.uuid4().hex))
    try:
        manifest=manifest_factory()
        if config["phase"]=="pilot":
            manifest["rows"]=pilot_rows(manifest["rows"]);manifest["count"]=len(manifest["rows"])
            manifest["actor_count"]=len({r["actor"] for r in manifest["rows"]})
            manifest["class_support"]={label:sum(r["label"]==label for r in manifest["rows"]) for label in manifest["class_support"]}
        write_json(out/"protocol.json",PROTOCOL,exclusive=True);write_json(out/"actor-allowlist.json",{"actors":list(ACTORS),"exposure":EXPOSURE},exclusive=True)
        write_json(out/"manifest.json",manifest,exclusive=True);write_json(out/"asset-manifest.json",{"assets":config["assets"],"exposure":EXPOSURE},exclusive=True)
        executions={};wall,cpu=CAPS[config["phase"]]
        for name in ("mini-xception","vit"):
            cmd=command_builder(config_path,out/name,name)
            executions[name]=supervise(cmd,out/name,wall_seconds=wall,cpu_seconds=cpu,ids=[r["id"] for r in manifest["rows"]])
            write_json(out/"execution.json",executions)
            if executions[name]["status"]!="COMPLETE":raise RuntimeError("model cap/error; roster incomplete, no automatic continuation")
            if name=="mini-xception":
                first=json.loads((out/name/"model-result.json").read_bytes())["results"][name]["items"]
                identity=lambda rows:[(r["id"],r["actor"],r["label"]) for r in rows]
                if identity(first)!=identity(manifest["rows"]):raise ValueError("first worker cohort mismatch; cannot truncate manifest")
                manifest["rows"]=[{**r,"sha256":item["sha256"]} for r,item in zip(manifest["rows"],first)]
                write_json(out/"manifest.json",manifest)
        if config["phase"]=="pilot":
            write_json(out/"checkpoint.json",{"status":"PILOT_TIMING_ONLY","exposure":EXPOSURE,"scores_withheld":True})
            return 0
        cmd=command_builder(config_path,out/"report-worker","report")
        result=supervise(cmd,out/"report-worker",wall_seconds=600,cpu_seconds=600)
        if result["status"]!="COMPLETE":raise RuntimeError("report/bootstrap cap/error")
        write_json(out/"checkpoint.json",{"status":"COMPLETE","exposure":EXPOSURE})
        files=sorted(p for p in out.rglob("*") if p.is_file())
        (out/"receipts.sha256").write_text("".join(sha_file(p)+"  "+str(p.relative_to(out))+"\n" for p in files),encoding="utf-8")
        (out/"COMPLETE").open("x").close()  # final marker, only after audit/report and all jobs drained
        return 0
    except BaseException as exc:
        write_json(out/"checkpoint.json",{"status":"INCOMPLETE","error":type(exc).__name__+": "+str(exc),"exposure":EXPOSURE})
        raise

if __name__=="__main__":raise SystemExit(main())
