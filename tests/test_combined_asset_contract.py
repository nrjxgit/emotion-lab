"""Manufactured buffers and NumPy-only factories; no Torch/ONNX imports or assets."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import Mock
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from emolab.adapters import claude_combined as module
from emolab.adapters.claude_combined import (ClaudeCombinedAdapter,CombinedContractError,
    BlockedAssetError,UnsupportedSafeEncoder,HEAD_SCHEMA,INPUT_CONTRACT)

NATIVE=["anger","disgust","fear","happiness","neutral","sadness","surprise"]


def digest(raw):return hashlib.sha256(raw).hexdigest()

def manifest_digest(manifest):
    return digest(json.dumps(manifest,sort_keys=True,separators=(",",":"),allow_nan=False).encode())


@pytest.fixture(autouse=True)
def forbid_runtimes_and_network(monkeypatch):
    import builtins,socket
    assert "torch" not in sys.modules and "onnxruntime" not in sys.modules
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.split(".")[0] in ("torch","onnxruntime","safetensors"):
            raise AssertionError("real runtime import forbidden")
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",guarded)
    def denied(*args,**kwargs):raise AssertionError("network forbidden")
    monkeypatch.setattr(socket.socket,"connect",denied)
    monkeypatch.setattr(socket,"create_connection",denied)


class FakeEncoder:
    def __init__(self,bundle):self.bundle=bundle
    def load_state_dict(self,state,*,strict):
        assert strict is True and set(state)=={"sentinel"}
        assert not state["sentinel"].flags.writeable
        self.state=state;self.bundle.encoder_strict+=1
    def to(self,device):assert device=="cpu";self.bundle.cpu_calls+=1;return self
    def eval(self):self.bundle.eval_calls+=1;return self
    def forward_features(self,x,*,feature):
        assert feature==self.bundle.manifest["encoder"]["feature"]
        self.bundle.inputs.append(x.copy())
        return self.bundle.features.copy()


class FakeHead:
    def __init__(self,bundle,mu,sd):
        assert mu.shape==sd.shape==(1280,) and np.all(sd>0)
        assert not mu.flags.writeable and not sd.flags.writeable
        self.bundle=bundle
    def load_state_dict(self,state,*,strict):
        assert strict is True and set(state)==set(HEAD_SCHEMA)-{"mu","sd"}
        self.bundle.head_strict+=1
    def to(self,device):assert device=="cpu";self.bundle.cpu_calls+=1;return self
    def eval(self):self.bundle.eval_calls+=1;return self
    def __call__(self,features):
        self.bundle.seen_features.append(features.copy())
        return self.bundle.logits[None].copy()


def make_bundle(tmp_path,*,classes=None,logits=None):
    """All pins here are manufactured test expectations, never asset authority."""
    classes=list(NATIVE if classes is None else classes)
    bundle=SimpleNamespace(load_calls=[],inputs=[],seen_features=[],encoder_strict=0,head_strict=0,
        cpu_calls=0,eval_calls=0,inference_calls=0,features=np.full((1,1280),7.25,np.float32),
        logits=np.zeros(7,np.float32) if logits is None else np.asarray(logits,np.float32))
    encoder_raw=b"SYNTHETIC encoder tensor fixture";head_raw=b"SYNTHETIC head tensor fixture"
    def asset(role,raw,format="json"):
        path=tmp_path/(role+(".bin" if format=="torch-state-dict" else ".json"));path.write_bytes(raw)
        return {"path":path.name,"format":format,"sha256":digest(raw),"bytes":len(raw)}
    encoder=asset("encoder",encoder_raw,"torch-state-dict")
    encoder.update(architecture={"name":"SYNTHETIC-reviewed-fake","version":"fixture/v1","source_sha256":digest(b"SYNTHETIC architecture source")},
        feature={"node":"exact_forward_features","pooling":"manufactured_global_mean","shape":[None,1280]},tensor_schema={"sentinel":[1280]})
    head=asset("head",head_raw,"torch-state-dict");head.update(tensor_schema=deepcopy(HEAD_SCHEMA),classes=classes)
    sidecar=asset("sidecar",json.dumps({"spec":{"in_dim":1280,"hidden":256,"n_classes":7,"dropout":.3,"va":False},"classes":classes,"version":"SYNTHETIC"}).encode());sidecar["classes"]=classes
    linkage=asset("linkage",json.dumps({"encoder_sha256":encoder["sha256"],"head_sha256":head["sha256"],"feature":encoder["feature"],"input_contract":INPUT_CONTRACT,"crop_id":"SYNTHETIC-fixed-crop/v1"}).encode())
    temperature=asset("temperature",json.dumps({"combined":{"temperature":1.0820367210745798}}).encode())
    bundle.manifest={"schema":"combined-embedding-contract/1","approved_root":str(tmp_path),"encoder":encoder,"head":head,"sidecar":sidecar,"linkage":linkage,
        "temperature":{"value":1.0820367210745798,"source":temperature,"key_path":["combined","temperature"]},"input_contract":INPUT_CONTRACT,"crop_id":"SYNTHETIC-fixed-crop/v1"}
    bundle.pins={role:(bundle.manifest[role] if role!="temperature" else temperature)["sha256"] for role in ("encoder","head","sidecar","linkage","temperature")}
    bundle.encoder_state={"sentinel":np.full(1280,7.25,np.float32)}
    bundle.head_state={key:np.ones(shape,np.float32) if key=="sd" else np.zeros(shape,np.float32) for key,shape in HEAD_SCHEMA.items()}
    def loader(buffer,*,weights_only,map_location):
        assert isinstance(buffer,io.BytesIO) and weights_only is True and map_location=="cpu"
        raw=buffer.read();bundle.load_calls.append(raw)
        if raw==encoder_raw:return bundle.encoder_state
        assert raw==head_raw
        return bundle.head_state
    bundle.loader=Mock(side_effect=loader)
    bundle.encoder_factory=Mock(side_effect=lambda spec:FakeEncoder(bundle))
    bundle.head_factory=Mock(side_effect=lambda mu,sd:FakeHead(bundle,mu,sd))
    @contextmanager
    def inference():
        bundle.inference_calls+=1
        yield
    bundle.inference=inference
    def build(**overrides):
        options=dict(expected_manifest_sha256=manifest_digest(bundle.manifest),expected_asset_pins=deepcopy(bundle.pins),tensor_loader=bundle.loader,
                     encoder_factory=bundle.encoder_factory,head_factory=bundle.head_factory,inference_context=bundle.inference)
        options.update(overrides)
        return ClaudeCombinedAdapter.for_synthetic(bundle.manifest,**options)
    bundle.build=build
    def rewrite(role,value):
        record=bundle.manifest[role] if role!="temperature" else bundle.manifest[role]["source"]
        raw=json.dumps(value).encode();(tmp_path/record["path"]).write_bytes(raw)
        record.update(sha256=digest(raw),bytes=len(raw));bundle.pins[role]=digest(raw)
    bundle.rewrite=rewrite
    return bundle


def test_real_constructor_is_blocked_before_filesystem(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError("filesystem access before block")
    for name in ("exists","stat","lstat","open","read_bytes","resolve"):
        monkeypatch.setattr(Path,name,forbidden)
    with pytest.raises(BlockedAssetError,match="manifest required"):
        ClaudeCombinedAdapter()
    with pytest.raises(BlockedAssetError,match="migration"):
        ClaudeCombinedAdapter("legacy.onnx","legacy.pt",head_spec_path="legacy.json")


def test_valid_manifest_still_cannot_enable_real_encoder(tmp_path,monkeypatch):
    bundle=make_bundle(tmp_path)
    monkeypatch.setattr(Path,"open",Mock(side_effect=AssertionError("no real reads")))
    with pytest.raises(UnsupportedSafeEncoder,match="real encoder disabled pending R11"):
        ClaudeCombinedAdapter(manifest=bundle.manifest)
    bundle.loader.assert_not_called()


@pytest.mark.parametrize("kind",["unknown","missing","manifest_pin","asset_pin","oversize","escape","absolute","duplicate","bad_format","bad_feature","non_synthetic"])
def test_invalid_contract_never_opens_or_loads(tmp_path,monkeypatch,kind):
    b=make_bundle(tmp_path);options={}
    if kind=="unknown":b.manifest["extra"]=True
    elif kind=="missing":b.manifest.pop("linkage")
    elif kind=="manifest_pin":options["expected_manifest_sha256"]="0"*64
    elif kind=="asset_pin":b.manifest["encoder"]["sha256"]="0"*64
    elif kind=="oversize":b.manifest["encoder"]["bytes"]=9*1024**2
    elif kind=="escape":b.manifest["encoder"]["path"]="../escape.bin"
    elif kind=="absolute":b.manifest["encoder"]["path"]=str(tmp_path/"absolute.bin")
    elif kind=="duplicate":b.manifest["head"]["path"]=b.manifest["encoder"]["path"]
    elif kind=="bad_format":b.manifest["encoder"]["format"]="serialized-module"
    elif kind=="bad_feature":b.manifest["encoder"]["feature"]["shape"]=[None,8]
    else:b.manifest["encoder"]["architecture"]["name"]="not-a-synthetic-identity"
    monkeypatch.setattr(Path,"open",Mock(side_effect=AssertionError("invalid contract read")))
    with pytest.raises(CombinedContractError):b.build(**options)
    b.loader.assert_not_called();b.encoder_factory.assert_not_called();b.head_factory.assert_not_called()


@pytest.mark.parametrize("kind",["missing","hash","size","reparse","changed_path"])
def test_bad_asset_never_reaches_loader(tmp_path,monkeypatch,kind):
    b=make_bundle(tmp_path);path=tmp_path/b.manifest["encoder"]["path"]
    if kind=="missing":b.manifest["encoder"]["path"]="MISSING.bin"
    elif kind=="hash":path.write_bytes(b"x"*b.manifest["encoder"]["bytes"])
    elif kind=="size":path.write_bytes(b"wrong size")
    elif kind=="reparse":
        original=Path.lstat
        def lstat(p,*args,**kwargs):
            info=original(p,*args,**kwargs)
            return SimpleNamespace(st_mode=info.st_mode,st_file_attributes=0x400) if p==path else info
        monkeypatch.setattr(Path,"lstat",lstat)
    else:
        original=Path.open
        def changed(p,*args,**kwargs):
            if p==path and args and args[0]=="rb":
                with original(p,"wb") as f:f.write(b"y"*b.manifest["encoder"]["bytes"])
            return original(p,*args,**kwargs)
        monkeypatch.setattr(Path,"open",changed)
    model=b.build()
    with pytest.raises(CombinedContractError):model.predict_proba(np.zeros((3,3,3),np.uint8))
    assert model._loaded is None and model._failed
    b.loader.assert_not_called();b.encoder_factory.assert_not_called();b.head_factory.assert_not_called()


@pytest.mark.parametrize("kind",["sidecar","linkage","temperature","head_classes"])
def test_cross_asset_binding_fails_before_tensor_load(tmp_path,kind):
    b=make_bundle(tmp_path)
    if kind=="sidecar":b.rewrite("sidecar",{"classes":list(reversed(NATIVE)),"spec":{"n_classes":7}})
    elif kind=="linkage":b.rewrite("linkage",{"encoder_sha256":"0"*64})
    elif kind=="temperature":b.rewrite("temperature",{"combined":{"temperature":2.}})
    else:b.manifest["head"]["classes"]=list(reversed(NATIVE))
    with pytest.raises(CombinedContractError):b.build().predict_proba(np.zeros((3,3,3),np.uint8))
    b.loader.assert_not_called()


@pytest.mark.parametrize("kind",["missing_key","extra_key","shape","nan","sd_zero","sd_nan","encoder_key"])
def test_tensor_contract_and_failed_cache_are_strict(tmp_path,kind):
    b=make_bundle(tmp_path)
    if kind=="missing_key":b.head_state.pop("net.0.bias")
    elif kind=="extra_key":b.head_state["surprise.extra"]=np.zeros(1,np.float32)
    elif kind=="shape":b.head_state["mu"]=np.zeros(1,np.float32)
    elif kind=="nan":b.head_state["net.3.bias"][0]=np.nan
    elif kind=="sd_zero":b.head_state["sd"][0]=0
    elif kind=="sd_nan":b.head_state["sd"][0]=np.nan
    else:b.encoder_state={"wrong":np.zeros(1280,np.float32)}
    model=b.build()
    with pytest.raises(CombinedContractError):model._load()
    assert model._loaded is None and model._failed
    b.encoder_factory.assert_not_called();b.head_factory.assert_not_called()
    calls=b.loader.call_count
    with pytest.raises(BlockedAssetError,match="new instance"):model._load()
    assert b.loader.call_count==calls


def test_exact_buffers_cache_sentinel_area_temperature_and_provenance(tmp_path,monkeypatch):
    b=make_bundle(tmp_path,logits=np.arange(7,dtype=np.float32));model=b.build()
    counts={};original=Path.open
    def opened(path,*args,**kwargs):
        if args and args[0]=="rb":counts[path.name]=counts.get(path.name,0)+1
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",opened)
    crop=np.arange(9*11*3,dtype=np.uint8).reshape(9,11,3)
    first=model.predict_proba(crop);model.predict_proba(crop)
    assert counts=={"encoder.bin":1,"head.bin":1,"sidecar.json":1,"linkage.json":1,"temperature.json":1}
    assert b.load_calls==[b"SYNTHETIC encoder tensor fixture",b"SYNTHETIC head tensor fixture"]
    assert b.encoder_factory.call_count==b.head_factory.call_count==1
    assert b.encoder_strict==b.head_strict==1 and b.cpu_calls==b.eval_calls==2 and b.inference_calls==2
    assert np.array_equal(b.seen_features[0],np.full((1,1280),7.25,np.float32))
    rgb=module.cv2.cvtColor(crop,module.cv2.COLOR_BGR2RGB)
    expected=module.cv2.resize(rgb,(224,224),interpolation=module.cv2.INTER_AREA).astype(np.float32)/255.
    expected=((expected-module._IMNET_MEAN)/module._IMNET_STD).transpose(2,0,1)[None]
    assert np.array_equal(b.inputs[0],expected)
    expected_probs=np.exp(np.arange(7)/1.0820367210745798);expected_probs/=expected_probs.sum()
    assert list(first.metadata.provenance["native_probabilities"].values())==pytest.approx(expected_probs)
    provenance=first.metadata.provenance
    assert provenance["encoder"]["sha256"]==b.pins["encoder"]
    assert provenance["encoder"]["path"]==str(tmp_path/"encoder.bin")
    assert provenance["head"]["sha256"]==b.pins["head"] and provenance["sidecar"]["sha256"]==b.pins["sidecar"]
    assert provenance["manifest_sha256"]==manifest_digest(b.manifest)
    assert "SYNTHETIC / NOT AUTHENTICATED" in provenance["evidence"]
    assert "backbone" not in provenance and ".onnx" not in json.dumps(provenance)
    assert first.metadata.calibrated is False


def test_verified_buffer_survives_path_change_after_verification(tmp_path):
    b=make_bundle(tmp_path);original=b.loader.side_effect
    def loader(buffer,**kwargs):
        # All hashes passed; subsequent path bytes cannot replace admitted buffer.
        if not b.load_calls:(tmp_path/"encoder.bin").write_bytes(b"CHANGED AFTER VERIFIED READ")
        return original(buffer,**kwargs)
    b.loader.side_effect=loader
    b.build().predict_proba(np.zeros((2,2,3),np.uint8))
    assert b.load_calls[0]==b"SYNTHETIC encoder tensor fixture"


def test_separate_instances_never_share_loaded_models(tmp_path):
    a=tmp_path/"a";a.mkdir();b=tmp_path/"b";b.mkdir()
    first=make_bundle(a);second=make_bundle(b)
    m1=first.build();m2=second.build();m1._load();m2._load()
    assert m1._loaded[0] is not m2._loaded[0] and m1._loaded[1] is not m2._loaded[1]
    assert first.loader.call_count==second.loader.call_count==2
    assert m1._manifest_sha256!=m2._manifest_sha256


@pytest.mark.parametrize("kind",["feature_width","feature_nan","feature_int","head_nan","head_width"])
def test_bad_feature_or_logits_never_become_prediction(tmp_path,kind):
    b=make_bundle(tmp_path)
    if kind=="feature_width":b.features=np.zeros((1,8),np.float32)
    elif kind=="feature_nan":b.features[0,0]=np.nan
    elif kind=="feature_int":b.features=np.zeros((1,1280),int)
    elif kind=="head_nan":b.logits[0]=np.nan
    else:b.logits=np.zeros(8,np.float32)
    with pytest.raises(CombinedContractError):b.build().predict_proba(np.zeros((2,2,3),np.uint8))


def test_factory_failure_never_publishes_partial_cache(tmp_path):
    b=make_bundle(tmp_path)
    b.head_factory.side_effect=RuntimeError("SYNTHETIC head construction failure")
    model=b.build()
    with pytest.raises(RuntimeError,match="head construction failure"):model._load()
    assert model._loaded is None and model._failed
    assert b.encoder_factory.call_count==1 and b.head_factory.call_count==1
    with pytest.raises(BlockedAssetError,match="new instance"):model._load()
    assert b.encoder_factory.call_count==1


def test_manifest_snapshot_cannot_be_changed_by_caller(tmp_path):
    b=make_bundle(tmp_path);expected=b.pins["encoder"]
    model=b.build();b.manifest["encoder"]["sha256"]="0"*64
    result=model.predict_proba(np.zeros((2,2,3),np.uint8))
    assert result.metadata.provenance["encoder"]["sha256"]==expected
