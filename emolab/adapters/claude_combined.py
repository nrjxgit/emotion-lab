"""Combined containment: real encoder DISABLED pending independently admitted R11.

No implicit assets, ONNX sessions, serialized modules or model-runtime imports.
The separate for_synthetic harness accepts manufactured tensor buffers/factories;
it NEVER authenticates a checkpoint. Instances serialize loading and prediction.
Preprocessing remains BGR->RGB, AREA224, /255, ImageNet mean/std, NCHW.
Combined CREMA-D/v0.3 remains in-sample; calibrated=False, no independence claim.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
from threading import RLock
import time
from types import MappingProxyType

import cv2
import numpy as np
from emolab.inference import FaceEmotionAdapter, ModelMetadata, ModelResult

FER7 = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
_NATIVE_TO_FER7 = {"anger":"angry", "angry":"angry", "disgust":"disgust", "fear":"fear",
                  "happiness":"happy", "happy":"happy", "neutral":"neutral",
                  "sadness":"sad", "sad":"sad", "surprise":"surprise"}
_IMNET_MEAN = np.array([.485,.456,.406],dtype=np.float32)
_IMNET_STD = np.array([.229,.224,.225],dtype=np.float32)
_COMBINED_TEMPERATURE = 1.0820367210745798
INPUT_CONTRACT = "combined-bgr-area224-imagenet-nchw/v1"
HEAD_SCHEMA = {"mu":[1280],"sd":[1280],"net.0.weight":[256,1280],
               "net.0.bias":[256],"net.3.weight":[7,256],"net.3.bias":[7]}
_ROLES = ("encoder","head","sidecar","linkage","temperature")
_MAX_BYTES = {"encoder":8*1024**2,"head":4*1024**2,"sidecar":65536,"linkage":65536,"temperature":65536}


class BlockedAssetError(RuntimeError):
    """No admitted real Combined asset route is available."""


class UnsupportedSafeEncoder(BlockedAssetError):
    """The authoritative safe encoder architecture/representation is unverified."""


class CombinedContractError(ValueError):
    """Manifest, immutable asset bytes or synthetic tensor contract mismatch."""


def _keys(value, expected):
    if not isinstance(value,dict) or set(value)!=set(expected):
        raise CombinedContractError("missing/unknown contract fields")


def _sha(value):
    if not isinstance(value,str) or not re.fullmatch("[0-9a-f]{64}",value):
        raise CombinedContractError("full lowercase SHA256 required")


def _canonical(value):
    try:return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    except (TypeError,ValueError) as exc:raise CombinedContractError("invalid manifest JSON") from exc


def _labels(labels):
    if not isinstance(labels,list) or len(labels)!=7 or any(not isinstance(x,str) or x not in _NATIVE_TO_FER7 for x in labels):
        raise CombinedContractError("head spec requires seven supported native labels")
    canonical=[_NATIVE_TO_FER7[x] for x in labels]
    if len(set(canonical))!=7:raise CombinedContractError("native labels must map uniquely to canonical FER7")
    return tuple(labels)


def _relative(path):
    if not isinstance(path,str) or not path or "\\" in path or ":" in path:
        raise CombinedContractError("relative slash-separated asset path required")
    p=Path(path)
    if p.is_absolute() or any(x in ("", ".", "..") for x in path.split("/")):
        raise CombinedContractError("asset path escape")
    return p


def _record(record,role,extra=()):
    _keys(record,{"path","format","sha256","bytes",*extra})
    _relative(record["path"]);_sha(record["sha256"])
    size=record["bytes"]
    if type(size) is not int or not 0<size<=_MAX_BYTES[role]:
        raise CombinedContractError("asset byte limit exceeded")
    expected="torch-state-dict" if role in ("encoder","head") else "json"
    if record["format"]!=expected:raise CombinedContractError("unsupported safe tensor/data format")


def _schema(schema):
    if not isinstance(schema,dict) or not schema:raise CombinedContractError("tensor schema required")
    for key,shape in schema.items():
        if not isinstance(key,str) or not isinstance(shape,list) or not shape or len(shape)>4 or any(type(n) is not int or n<=0 for n in shape):
            raise CombinedContractError("invalid tensor key/shape schema")
        if int(np.prod(shape,dtype=object))>2_000_000:raise CombinedContractError("oversize tensor schema")


def _validate_manifest(manifest):
    _keys(manifest,{"schema","approved_root","encoder","head","sidecar","linkage","temperature","input_contract","crop_id"})
    if manifest["schema"]!="combined-embedding-contract/1":raise CombinedContractError("unsupported manifest schema")
    root=manifest["approved_root"]
    if not isinstance(root,str) or not Path(root).is_absolute() or root.startswith(("\\\\","//")) or ".." in Path(root).parts or any(":" in x for x in Path(root).parts[1:]):
        raise CombinedContractError("absolute local approved root required")
    if manifest["input_contract"]!=INPUT_CONTRACT or not isinstance(manifest["crop_id"],str) or not manifest["crop_id"]:
        raise CombinedContractError("input/crop binding required")
    _record(manifest["encoder"],"encoder",("architecture","feature","tensor_schema"))
    encoder=manifest["encoder"];architecture=encoder["architecture"]
    _keys(architecture,{"name","version","source_sha256"});_sha(architecture["source_sha256"])
    if any(not isinstance(architecture[k],str) or not architecture[k] for k in ("name","version")):
        raise CombinedContractError("architecture identity/version required")
    feature=encoder["feature"];_keys(feature,{"node","pooling","shape"})
    if feature["shape"]!=[None,1280] or any(not isinstance(feature[k],str) or not feature[k] for k in ("node","pooling")):
        raise CombinedContractError("exact feature node/pooling and [batch,1280] required")
    _schema(encoder["tensor_schema"])
    _record(manifest["head"],"head",("tensor_schema","classes"));_schema(manifest["head"]["tensor_schema"])
    if manifest["head"]["tensor_schema"]!=HEAD_SCHEMA:raise CombinedContractError("exact head tensor schema including mu/sd required")
    _labels(manifest["head"]["classes"])
    _record(manifest["sidecar"],"sidecar",("classes",));_labels(manifest["sidecar"]["classes"])
    if manifest["sidecar"]["classes"]!=manifest["head"]["classes"]:raise CombinedContractError("sidecar/head classes mismatch")
    _record(manifest["linkage"],"linkage")
    temperature=manifest["temperature"];_keys(temperature,{"value","source","key_path"})
    if type(temperature["value"]) not in (float,int) or temperature["value"]!=_COMBINED_TEMPERATURE:
        raise CombinedContractError("fixed Combined temperature required")
    if not isinstance(temperature["key_path"],list) or not temperature["key_path"] or any(not isinstance(k,str) or not k for k in temperature["key_path"]):
        raise CombinedContractError("temperature source selector required")
    _record(temperature["source"],"temperature")
    paths=[_asset_record(manifest,role)["path"].casefold() for role in _ROLES]
    if len(set(paths))!=len(paths):raise CombinedContractError("duplicate asset path")
    return deepcopy(manifest)


def _asset_record(manifest,role):
    return manifest["temperature"]["source"] if role=="temperature" else manifest[role]


def _no_reparse(path):
    for part in (*reversed(path.parents),path):
        try:info=part.lstat()
        except OSError as exc:raise CombinedContractError("missing asset/root") from exc
        if stat.S_ISLNK(info.st_mode) or getattr(info,"st_file_attributes",0)&0x400:
            raise CombinedContractError("reparse/link escape denied")


def _read_asset(root,record,expected):
    """One bounded read, same bytes hashed and later passed to the loader."""
    path=root/_relative(record["path"]);_no_reparse(path)
    if not path.resolve().is_relative_to(root.resolve()):raise CombinedContractError("asset escaped root")
    before=path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size!=record["bytes"]:
        raise CombinedContractError("asset type/size mismatch")
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    with path.open("rb") as stream:
        if identity(os.fstat(stream.fileno()))!=identity(before):raise CombinedContractError("asset changed before read")
        raw=stream.read(record["bytes"]+1)
        if identity(os.fstat(stream.fileno()))!=identity(before):raise CombinedContractError("asset changed during read")
    _no_reparse(path)
    if identity(path.stat())!=identity(before):raise CombinedContractError("asset path changed")
    if len(raw)!=record["bytes"] or hashlib.sha256(raw).hexdigest()!=expected:
        raise CombinedContractError("asset hash/size mismatch")
    return raw


def _state(raw,loader,schema):
    # Synthetic loader has the torch.load signature; real deserialization DISABLED.
    state=loader(io.BytesIO(raw),weights_only=True,map_location="cpu")
    if not isinstance(state,dict) or set(state)!=set(schema):raise CombinedContractError("tensor keys mismatch")
    frozen={}
    for key,shape in schema.items():
        value=np.asarray(state[key])
        if list(value.shape)!=shape or value.dtype.kind!="f" or not np.isfinite(value).all():
            raise CombinedContractError("tensor shape/finite/float mismatch")
        frozen[key]=np.frombuffer(value.tobytes(),dtype=value.dtype).reshape(value.shape)
    if "sd" in frozen and not (frozen["sd"]>0).all():raise CombinedContractError("normalization sd must be positive")
    return MappingProxyType(frozen)


class ClaudeCombinedAdapter(FaceEmotionAdapter):
    name="claude-combined"
    version="0.4.1"
    input_shape=(224,224,3)
    native_labels=()
    source="Claude Code Emotion Lab"
    licence="project code; FER2013/CREMA-D research terms unresolved"
    calibrated=False
    temperature=_COMBINED_TEMPERATURE

    def __init__(self,backbone_path=None,head_path=None,*,head_spec_path=None,manifest=None):
        # Deliberately precedes Path/exists/stat/read and any model loader.
        if any(x is not None for x in (backbone_path,head_path,head_spec_path)):
            raise BlockedAssetError("migration required: legacy paths are disabled; approved manifest required")
        if manifest is None:raise BlockedAssetError("approved Combined manifest required before filesystem access")
        _validate_manifest(manifest)
        raise UnsupportedSafeEncoder("real encoder disabled pending R11, independent authority and reviewed safe architecture")

    @classmethod
    def for_synthetic(cls,manifest,*,expected_manifest_sha256,expected_asset_pins,
                      tensor_loader,encoder_factory,head_factory,inference_context=nullcontext):
        """Manufactured fixtures ONLY. Pins supplied independently by test, not trust.

        This method cannot enable the real constructor or attest real assets.
        All result provenance is explicitly SYNTHETIC / NOT AUTHENTICATED.
        """
        _sha(expected_manifest_sha256)
        raw=_canonical(manifest)
        if hashlib.sha256(raw).hexdigest()!=expected_manifest_sha256:raise CombinedContractError("manifest pin mismatch")
        m=_validate_manifest(manifest)
        if not m["encoder"]["architecture"]["name"].startswith("SYNTHETIC-") or not m["crop_id"].startswith("SYNTHETIC-"):
            raise CombinedContractError("synthetic identity required; not an asset admission hook")
        _keys(expected_asset_pins,_ROLES)
        for role in _ROLES:
            _sha(expected_asset_pins[role])
            if expected_asset_pins[role]!=_asset_record(m,role)["sha256"]:raise CombinedContractError("independent synthetic asset pin mismatch")
        self=object.__new__(cls)
        self._manifest=m;self._manifest_sha256=expected_manifest_sha256
        self._pins=dict(expected_asset_pins);self._tensor_loader=tensor_loader
        self._encoder_factory=encoder_factory;self._head_factory=head_factory
        self._inference_context=inference_context;self._lock=RLock();self._loaded=None;self._failed=False
        self.native_labels=tuple(m["sidecar"]["classes"])
        self._native_to_canonical={x:_NATIVE_TO_FER7[x] for x in self.native_labels}
        return self

    def _load(self):
        with self._lock:
            if self._failed:raise BlockedAssetError("synthetic initialization failed; new instance required")
            if self._loaded is not None:return self._loaded
            try:
                m=self._manifest;root=Path(m["approved_root"]);_no_reparse(root)
                if not root.is_dir():raise CombinedContractError("approved root must be a directory")
                buffers={role:_read_asset(root,_asset_record(m,role),self._pins[role]) for role in _ROLES}
                sidecar=json.loads(buffers["sidecar"])
                if sidecar.get("classes")!=list(self.native_labels) or sidecar.get("spec")!={"in_dim":1280,"hidden":256,"n_classes":7,"dropout":.3,"va":False}:
                    raise CombinedContractError("sidecar/head spec mismatch")
                linkage=json.loads(buffers["linkage"])
                expected={"encoder_sha256":self._pins["encoder"],"head_sha256":self._pins["head"],"feature":m["encoder"]["feature"],"input_contract":INPUT_CONTRACT,"crop_id":m["crop_id"]}
                if linkage!=expected:raise CombinedContractError("head-training linkage mismatch")
                temperature=json.loads(buffers["temperature"])
                for key in m["temperature"]["key_path"]:temperature=temperature[key]
                if type(temperature) not in (float,int) or temperature!=_COMBINED_TEMPERATURE:
                    raise CombinedContractError("temperature source mismatch")
                encoder_state=_state(buffers["encoder"],self._tensor_loader,m["encoder"]["tensor_schema"])
                head_state=_state(buffers["head"],self._tensor_loader,HEAD_SCHEMA)
                encoder=self._encoder_factory(deepcopy(m["encoder"]))
                encoder.load_state_dict(dict(encoder_state),strict=True)
                encoder=encoder.to("cpu").eval()
                # mu/sd are explicit constructor buffers; all remaining parameters strict.
                head=self._head_factory(head_state["mu"],head_state["sd"])
                head.load_state_dict({k:v for k,v in head_state.items() if k not in ("mu","sd")},strict=True)
                head=head.to("cpu").eval()
                self._loaded=(encoder,head)
                return self._loaded
            except BaseException:
                self._loaded=None;self._failed=True
                raise

    def _get_embedding(self,x):
        encoder,_=self._load()
        value=np.asarray(encoder.forward_features(x,feature=deepcopy(self._manifest["encoder"]["feature"])))
        if value.shape!=(1,1280) or value.dtype.kind!="f" or not np.isfinite(value).all():
            raise CombinedContractError("feature must be finite float [batch,1280]")
        return value

    def predict_proba(self,face_rgb):
        with self._lock:
            _,head=self._load();start=time.perf_counter()
            crop=np.asarray(face_rgb)
            if crop.dtype!=np.uint8 or crop.ndim!=3 or crop.shape[2]!=3 or min(crop.shape[:2])<=0:
                raise CombinedContractError("BGR uint8 crop required")
            rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
            resized=cv2.resize(rgb,(224,224),interpolation=cv2.INTER_AREA)
            x=((resized.astype(np.float32)/255.-_IMNET_MEAN)/_IMNET_STD).transpose(2,0,1)[None]
            with self._inference_context():
                embedding=self._get_embedding(x)
                logits=np.asarray(head(embedding))
            if logits.shape!=(1,7) or logits.dtype.kind!="f" or not np.isfinite(logits).all():
                raise CombinedContractError("head output must be finite float [batch,7]")
            logits=logits[0]/_COMBINED_TEMPERATURE;vector=np.exp(logits-logits.max());vector/=vector.sum()
            native=dict(zip(self.native_labels,map(float,vector)))
            mapped={self._native_to_canonical[k]:v for k,v in native.items()}
            probs={name:mapped[name] for name in FER7};top=max(probs,key=probs.get)
            m=self._manifest
            encoder={**deepcopy(m["encoder"]),"path":str(Path(m["approved_root"])/m["encoder"]["path"])}
            provenance={"evidence":"SYNTHETIC / NOT AUTHENTICATED; real execution disabled",
                "exposure":"Combined CREMA-D/v0.3 in-sample; no independence claim",
                "source":self.source,"encoder":encoder,"head":{**deepcopy(m["head"]),"path":str(Path(m["approved_root"])/m["head"]["path"])},
                "sidecar":{**deepcopy(m["sidecar"]),"path":str(Path(m["approved_root"])/m["sidecar"]["path"])},
                "manifest_sha256":self._manifest_sha256,"head_spec":str(Path(m["approved_root"])/m["sidecar"]["path"]),
                "head_spec_sha256":self._pins["sidecar"],"linkage_sha256":self._pins["linkage"],
                "input_contract":INPUT_CONTRACT,"crop_id":m["crop_id"],
                "preprocessing":{"color":"BGR->RGB","resize":"AREA224","scale":"/255","mean":_IMNET_MEAN.tolist(),"std":_IMNET_STD.tolist(),"layout":"NCHW"},
                "temperature":_COMBINED_TEMPERATURE,"temperature_source":deepcopy(m["temperature"]),
                "original_classes":list(self.native_labels),"native_probabilities":native,
                "native_to_canonical":dict(self._native_to_canonical),"unsupported_native_classes":[]}
            metadata=ModelMetadata(self.name,self.version,self.input_shape,dict(enumerate(self.native_labels)),False,frozenset(FER7),provenance)
            return ModelResult(probs,top,probs[top],1-probs[top],(time.perf_counter()-start)*1000,metadata)
