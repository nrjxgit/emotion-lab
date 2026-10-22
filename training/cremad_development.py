"""Development-only CREMA-D boundary. Import performs no media/model access.

Metadata admission precedes every open. Payload hashing/decoding is ONLY for a
later explicitly authorized run, never metadata preflight or import.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SOURCE = ROOT / "eval_data_cremad/cremad_repo/VideoFlash"
ACTORS = (1001,1002,1003,1004,1008,1015,1018,1019,1022,1024,1025,1029,1030,1031,1032,1033,1034,1035,1039,1040,1041,1044,1047,1048,1049,1050,1052,1055,1056,1057,1058,1060,1062,1063,1064,1065,1066,1067,1070,1072,1078,1079,1080,1085,1088,1089)
LABELS = dict(ANG="angry", DIS="disgust", FEA="fear", HAP="happy", NEU="neutral", SAD="sad")
SENTENCES = frozenset("DFA IEO IOM ITH ITS IWL IWW MTI TAI TIE TSI WSI".split())
SUPPORT = {"angry":642,"disgust":642,"fear":642,"happy":642,"neutral":549,"sad":642}
FRAME_POLICY = "sequential decode; unique floor((N-1)*q), q=.25,.50,.75; equal usable-frame mean"
EXPOSURE = "DEVELOPMENT / RETROSPECTIVE; upstream CREMA-D exposure UNKNOWN"
PROTOCOL = {"unit":"clip", "actors":list(ACTORS), "frame_policy":FRAME_POLICY,
    "detector":{"confidence":.6,"nms":.3,"top_k":5000},"crop":{"margin":.25,"square":False},
    "output_order":["angry","disgust","fear","happy","sad","surprise","neutral"],
    "bootstrap_replicates":1000,"seed":505,"ece":None,"exposure":EXPOSURE,
    "designer_order_sha256":"fe074a5ed70a4a7aa525ee2971a8753e6d60d3ff54613c66b338c70688718786",
    "class_map":LABELS,"metrics":"fixed six-class macroF1; 6x8 confusion including missing; actor-paired percentile 95% CI",
    "preprocess":{"mini":"gray64 bilinear /255 then *2-1","vit":"RGB224 bilinear /255 mean/std .5"},
    "caps":{"pilot_wall":600,"pilot_cpu":300,"full_wall":3600,"full_cpu":2700,"memory":4294967296,"stop_grace":10},
    "tuning":False,"ece_reason":"no authorized validation split/probability endpoint"}
PROTOCOL_SHA256 = hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()

class BoundaryError(ValueError): pass
class DecodeFailure(RuntimeError): pass

def safe_path(path):
    p=Path(path)
    if not p.is_absolute() or str(p).startswith(("\\\\","//")) or ".." in p.parts or any(":" in x for x in p.parts[1:]):
        raise BoundaryError("absolute local non-ADS path required")
    for q in (p,*p.parents):
        if q.exists() or q.is_symlink():
            info=q.lstat()
            if q.is_symlink() or getattr(info,"st_file_attributes",0)&getattr(stat,"FILE_ATTRIBUTE_REPARSE_POINT",0):
                raise BoundaryError("reparse/link denied")
    return p.resolve()

def parse_name(name):
    m=re.fullmatch(r"([0-9]{4})_([A-Z]{3})_([A-Z]{3})_(LO|MD|HI|XX)\.flv",name)
    if not m: raise BoundaryError("malformed clip name")
    actor=int(m[1])
    if actor not in ACTORS: raise BoundaryError("actor outside development inclusion")
    if m[2] not in SENTENCES or m[3] not in LABELS: raise BoundaryError("unknown clip token")
    return actor,LABELS[m[3]]

def admitted_path(root,name):
    parse_name(name)  # MUST precede any path stat/open, even for forged rows.
    root=safe_path(root)
    p=safe_path(root/name)
    if p.parent!=root or not p.is_file(): raise BoundaryError("missing/nonlocal clip")
    return p

def metadata_manifest(root=CANONICAL_SOURCE, *, expected_count=3759, expected_actors=ACTORS, expected_support=SUPPORT):
    """Strict production defaults. Private tests supply tiny expected counts only.

    Enumerate names only; excluded actor entries are discarded before stat/open
    and never included in diagnostics. No clip bytes/hash/decode occurs here.
    """
    root=safe_path(root)
    if root!=safe_path(CANONICAL_SOURCE): raise BoundaryError("noncanonical source")
    rows=[];seen=set()
    for p in sorted(root.iterdir(),key=lambda p:p.name):
        prefix=p.name.split("_",1)[0]
        if prefix.isdigit() and int(prefix) not in ACTORS: continue
        if p.suffix!=".flv":raise BoundaryError("BLOCKED: non-clip directory entry discrepancy; no payload opened")
        actor,label=parse_name(p.name)
        if p.name.casefold() in seen: raise BoundaryError("duplicate clip ID")
        seen.add(p.name.casefold());p=admitted_path(root,p.name)
        rows.append({"id":p.name,"actor":actor,"label":label,"bytes":p.stat().st_size})
    counts={c:sum(x["label"]==c for x in rows) for c in LABELS.values()}
    if len(rows)!=expected_count or set(x["actor"] for x in rows)!=set(expected_actors) or counts!=expected_support:
        raise BoundaryError("cohort count/support discrepancy; no payload opened")
    return {"root":str(root),"rows":rows,"actor_count":len(expected_actors),"count":len(rows),
            "class_support":counts,"protocol_sha256":PROTOCOL_SHA256,"frame_policy":FRAME_POLICY,"exposure":EXPOSURE}

def bind_payload(root,row):
    p=admitted_path(root,row["id"]);actor,label=parse_name(row["id"])
    if row.get("actor")!=actor or row.get("label")!=label: raise BoundaryError("forged metadata row")
    before=p.stat();h=hashlib.sha256()
    with p.open("rb") as f:
        prefix=f.read(128)
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"): raise BoundaryError("LFS pointer; never fetch")
        h.update(prefix)
        for chunk in iter(lambda:f.read(1048576),b""):h.update(chunk)
    after=p.stat()
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    if identity(before)!=identity(after) or after.st_size!=row["bytes"]:raise BoundaryError("payload changed")
    if row.get("sha256") is not None and h.hexdigest()!=row["sha256"]:raise BoundaryError("payload hash mismatch")
    return {**row,"sha256":h.hexdigest(),"frame_policy":FRAME_POLICY,"protocol_sha256":PROTOCOL_SHA256}

def frame_indices(n):
    if n<=0:return []
    return sorted({math.floor((n-1)*q) for q in (.25,.5,.75)})

def decode_selected(root,row,reader_factory=None):
    bound=bind_payload(root,row);p=admitted_path(root,bound["id"])
    if reader_factory is None:
        import cv2
        reader_factory=cv2.VideoCapture
    def frames():
        cap=reader_factory(str(p))
        try:
            if not cap.isOpened():raise DecodeFailure("decoder unavailable/corrupt clip")
            while True:
                ok,frame=cap.read()
                if not ok:break
                yield frame
        finally:cap.release()
    n=sum(1 for _ in frames());indices=frame_indices(n);selected=[];count=0
    for i,frame in enumerate(frames()):
        count+=1
        if i in indices:selected.append(np.array(frame,copy=True))
    if count!=n or sha_file(p)!=bound["sha256"]:raise BoundaryError("decode/payload identity drift")
    return selected,n,bound

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()

def choose_crop(frame,boxes):
    """Largest intersecting integer box, fixed tie order, canonical crop math."""
    if frame.dtype!=np.uint8 or frame.ndim!=3 or frame.shape[2]!=3:raise BoundaryError("BGR uint8 frame required")
    h,w=frame.shape[:2];valid=[]
    for box in boxes:
        coords=[box[k] for k in ("x","y","w","h")]
        if not all(np.isfinite(x) and float(x).is_integer() for x in coords):continue
        x,y,bw,bh=map(int,coords)
        if bw>0 and bh>0 and x<w and y<h and x+bw>0 and y+bh>0:valid.append((x,y,bw,bh))
    if not valid:return None
    x,y,bw,bh=min(valid,key=lambda b:(-b[2]*b[3],*b))
    from backend.app.face.preprocessing import crop_pixels
    crop=crop_pixels(frame,dict(x=x,y=y,w=bw,h=bh),margin=.25,square=False)
    return np.frombuffer(crop.tobytes(),dtype=np.uint8).reshape(crop.shape)
