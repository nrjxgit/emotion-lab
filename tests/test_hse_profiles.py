"""Manufactured HSE profile regression tests: no reference assets or real models.

JSON/YAML sources and temperatures are generated under tmp_path. Fake session
outputs exercise source-byte binding, legacy compatibility, probability/mass
semantics, mutation isolation and fail-closed profile resolution. Source-read
spies admit only these fixtures and the local adapter's legacy source text.
"""

import sys
import pathlib
import hashlib
import json
import copy
import pytest
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "test_hse_profiles_adapter",
    ROOT / "emolab/adapters/claude_hsemotion.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

HSEmotionAdapter = module.HSEmotionAdapter
_resolve_profile_temperature = module._resolve_profile_temperature
_file_sha256_bytes = module._file_sha256_bytes
_load_profile_source = module._load_profile_source
_ADAPTER_SOURCE_PATH = module._ADAPTER_SOURCE_PATH

# Assigned per test by the manufactured source fixture, never reference paths.
CALIBRATION_V3 = None
DEFAULT_YAML = None
LEGACY_T = module._HSE_LEGACY_TEMPERATURE
REGISTRY_T = 2.25
LIVE_T = 3.75


@pytest.fixture(autouse=True)
def synthetic_profile_sources(tmp_path, monkeypatch):
    registry=tmp_path/"SYNTHETIC-registry.json"
    live=tmp_path/"SYNTHETIC-live.yaml"
    registry.write_bytes(json.dumps({"hsemotion_b0_va":{"temperature":REGISTRY_T}},indent=3).encode()+b"\n")
    live.write_bytes(f"# MANUFACTURED\nface:\n  expression:\n    temperature: {LIVE_T}\n".encode())
    monkeypatch.setitem(globals(),"CALIBRATION_V3",registry)
    monkeypatch.setitem(globals(),"DEFAULT_YAML",live)
    for profile,path in (("claude-registry-temperature/v1",registry),("claude-live-temperature/v1",live)):
        monkeypatch.setitem(module._HSE_PROFILES,profile,{
            **module._HSE_PROFILES[profile],"source_path":str(path),
            "description":"SYNTHETIC source-temperature fixture; no real authenticity claim"})
    original_read=pathlib.Path.read_bytes
    original_open=pathlib.Path.open
    reads=[]
    def admitted(path):
        path=pathlib.Path(path).resolve()
        assert path==_ADAPTER_SOURCE_PATH or path.is_relative_to(tmp_path.resolve()), "unexpected source read"
    def read_bytes(path):
        admitted(path);raw=original_read(path);reads.append((path,raw));return raw
    def guarded_open(path,mode="r",*args,**kwargs):
        if "r" in mode or "+" in mode:admitted(path)
        return original_open(path,mode,*args,**kwargs)
    monkeypatch.setattr(pathlib.Path,"read_bytes",read_bytes)
    monkeypatch.setattr(pathlib.Path,"open",guarded_open)
    original_loader=module._load_profile_source
    def source_loader(path):
        assert pathlib.Path(path).resolve().is_relative_to(tmp_path.resolve())
        return original_loader(path)
    monkeypatch.setattr(module,"_load_profile_source",source_loader)
    import socket
    def deny(*args,**kwargs):raise AssertionError("network forbidden")
    monkeypatch.setattr(socket.socket,"connect",deny)
    monkeypatch.setattr(socket,"create_connection",deny)
    yield reads
    assert all(pathlib.Path(path).resolve()==_ADAPTER_SOURCE_PATH or pathlib.Path(path).resolve().is_relative_to(tmp_path.resolve()) for path,_ in reads)


SYNTHETIC_LOGITS = np.array(
    [0.5, -0.3, 0.8, -1.2, 1.5, -0.7, 0.2, 0.9], dtype=np.float64
)
SYNTHETIC_VA = np.array([0.1, -0.4], dtype=np.float64)
SYNTHETIC_ROW = np.concatenate([SYNTHETIC_LOGITS, SYNTHETIC_VA])


class FakeSession:
    """Injects a fixed ONNX output row; records the input tensor for inspection."""

    def __init__(self, row):
        self.row = np.asarray(row, dtype=np.float32)
        self.last_tensor = None

    def get_inputs(self):
        from types import SimpleNamespace

        return [SimpleNamespace(name="synthetic_input")]

    def run(self, _outputs, inputs):
        self.last_tensor = inputs["synthetic_input"].copy()
        return [self.row.reshape(1, -1).copy()]  # mimic ONNX [batch, 10] output


def make_adapter(
    profile=None, logits=SYNTHETIC_LOGITS, va=SYNTHETIC_VA, expected_source_sha256=None
):
    adapter = HSEmotionAdapter(
        model_path="SYNTHETIC_DO_NOT_OPEN",
        temperature_profile=profile,
        expected_source_sha256=expected_source_sha256,
    )
    adapter._session = FakeSession(np.concatenate([logits, va]))
    return adapter


def _adapter_file_hash():
    return _file_sha256_bytes(_ADAPTER_SOURCE_PATH.read_bytes())


# ---------------------------------------------------------------------------
# Legacy compatibility
# ---------------------------------------------------------------------------


def test_legacy_default_constructor_no_profile():
    """No profile -> legacy default temperature, same as before the profile change."""
    a = make_adapter(profile=None)
    assert a.temperature == LEGACY_T
    assert a.temperature_profile["profile"] == "opencode-legacy/v1"
    # Legacy profile identifies OpenCode's own source file, not None
    assert a.temperature_profile["source_path"] == str(_ADAPTER_SOURCE_PATH)
    assert a.temperature_profile["key_path"] == ["_HSE_LEGACY_TEMPERATURE"]
    assert a.temperature_profile["raw_sha256"] == _adapter_file_hash()
    assert a.temperature_profile["value"] == LEGACY_T
    assert a.calibrated is False


def test_legacy_keyword_explicit():
    """Explicit opencode-legacy/v1 keyword resolves to the same legacy value."""
    a = make_adapter(profile="opencode-legacy/v1")
    assert a.temperature == LEGACY_T
    assert a.temperature_profile["profile"] == "opencode-legacy/v1"
    assert a.temperature_profile["value"] == LEGACY_T
    assert a.temperature_profile["source_path"] == str(_ADAPTER_SOURCE_PATH)


def test_legacy_profile_identifies_opencode_source():
    """The legacy value is NOT an Emotion Lab value; it carries OpenCode provenance."""
    a = make_adapter(profile=None)
    prof = a.temperature_profile
    assert "emotion-lab-claude-reference" not in prof["source_path"]
    assert prof["source_path"].endswith("claude_hsemotion.py")
    assert prof["key_path"] == ["_HSE_LEGACY_TEMPERATURE"]
    assert prof["raw_sha256"] == _adapter_file_hash()
    assert "legacy" in prof["description"].lower()
    assert "Not present" in prof["description"]


def test_legacy_temperature_unchanged():
    """Legacy default temperature remains bitwise identical to pre-repair value."""
    assert LEGACY_T == 1.3407638433649585
    a = make_adapter(profile=None)
    assert a.temperature == 1.3407638433649585


def test_constructor_preserves_model_path_argument():
    """model_path argument is preserved alongside temperature_profile keyword."""
    a = make_adapter(profile=None)
    assert a.model_path == "SYNTHETIC_DO_NOT_OPEN"


def test_face_rgb_keyword_remains_supported():
    """predict_proba(face_rgb=...) keyword still works after profile change."""
    a = make_adapter(profile="opencode-legacy/v1")
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    r = a.predict_proba(face_rgb=img)
    assert set(r.probabilities) == {
        "angry",
        "disgust",
        "fear",
        "happy",
        "sad",
        "surprise",
        "neutral",
    }


# ---------------------------------------------------------------------------
# Profile source resolution
# ---------------------------------------------------------------------------


def test_registry_profile_exact_value_and_hash():
    """claude-registry-temperature/v1 resolves to generated JSON bytes + SHA256."""
    expected_sha = _file_sha256_bytes(CALIBRATION_V3.read_bytes())
    a = make_adapter(profile="claude-registry-temperature/v1")
    assert a.temperature == REGISTRY_T
    prof = a.temperature_profile
    assert prof["profile"] == "claude-registry-temperature/v1"
    assert prof["source_path"] == str(CALIBRATION_V3)
    assert prof["key_path"] == ["hsemotion_b0_va", "temperature"]
    assert prof["value"] == REGISTRY_T
    assert prof["raw_sha256"] == expected_sha


def test_live_profile_exact_value_and_hash():
    """claude-live-temperature/v1 resolves to generated YAML bytes + SHA256."""
    expected_sha = _file_sha256_bytes(DEFAULT_YAML.read_bytes())
    a = make_adapter(profile="claude-live-temperature/v1")
    assert a.temperature == LIVE_T
    prof = a.temperature_profile
    assert prof["profile"] == "claude-live-temperature/v1"
    assert prof["source_path"] == str(DEFAULT_YAML)
    assert prof["key_path"] == ["face", "expression", "temperature"]
    assert prof["value"] == LIVE_T
    assert prof["raw_sha256"] == expected_sha


def test_single_byte_snapshot_hash_and_parse(monkeypatch):
    """Mutate backing fixture after its read; hash and parse must use first bytes."""
    first=CALIBRATION_V3.read_bytes()
    second=b'{"hsemotion_b0_va":{"temperature":9.5}}'
    original=pathlib.Path.read_bytes;reads=[]
    def replacing_read(path):
        raw=original(path)
        if path==CALIBRATION_V3:
            reads.append(raw);path.write_bytes(second)
        return raw
    monkeypatch.setattr(pathlib.Path,"read_bytes",replacing_read)
    data,sha=module._load_profile_source(str(CALIBRATION_V3))
    assert reads==[first]
    assert sha==hashlib.sha256(first).hexdigest()
    assert data["hsemotion_b0_va"]["temperature"]==REGISTRY_T


@pytest.mark.parametrize("bad_value",[0.,-1.,float("nan"),float("inf"),float("-inf"),True,False,"not-numeric",None])
def test_resolver_invalid_temperature_fails(bad_value):
    CALIBRATION_V3.write_bytes(json.dumps({"hsemotion_b0_va":{"temperature":bad_value}}).encode())
    with pytest.raises(ValueError):make_adapter(profile="claude-registry-temperature/v1")


def test_boolean_temperature_rejected():
    CALIBRATION_V3.write_bytes(b'{"hsemotion_b0_va":{"temperature":true}}')
    with pytest.raises(ValueError,match="boolean"):
        _resolve_profile_temperature("claude-registry-temperature/v1")


@pytest.mark.parametrize("profile",["claude-registry-temperature/v1","claude-live-temperature/v1"])
def test_missing_source_fails_before_model_loading(profile,tmp_path,monkeypatch):
    monkeypatch.setitem(module._HSE_PROFILES,profile,{**module._HSE_PROFILES[profile],"source_path":str(tmp_path/"MISSING.json")})
    with pytest.raises(ValueError,match="missing|not found|exist"):
        make_adapter(profile=profile)


def test_bad_selector_fails_before_model_loading(monkeypatch):
    profile="claude-registry-temperature/v1"
    monkeypatch.setitem(module._HSE_PROFILES,profile,{**module._HSE_PROFILES[profile],"key_path":["absent","temperature"]})
    with pytest.raises(ValueError,match="key path"):
        make_adapter(profile=profile)


def test_unknown_profile_fails_explicitly():
    """Unknown profile name raises ValueError with no fallback."""
    try:
        make_adapter(profile="nonexistent-profile/v99")
    except ValueError as exc:
        assert "unknown hse temperature profile" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for unknown profile")


def test_invalid_profile_type_fails():
    """Non-string profile selector fails explicitly."""
    try:
        make_adapter(profile=123)
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError("expected ValueError or TypeError for non-string profile")


def test_empty_profile_fails():
    """Empty string profile fails explicitly."""
    try:
        make_adapter(profile="")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty profile")


# ---------------------------------------------------------------------------
# Deep immutability
# ---------------------------------------------------------------------------


def test_temperature_profile_deep_immutability():
    """temperature_profile returns an independent deep copy; nested key_path
    lists are not shared with instance state."""
    a = make_adapter(profile="claude-registry-temperature/v1")
    prof1 = a.temperature_profile
    prof2 = a.temperature_profile
    # Two calls return independent objects
    assert prof1 is not prof2
    assert prof1 == prof2
    # Mutating the returned copy does not affect the instance
    prof1["key_path"].append("tampered")
    assert a.temperature_profile["key_path"] == ["hsemotion_b0_va", "temperature"]
    assert prof2["key_path"] == ["hsemotion_b0_va", "temperature"]


def test_result_provenance_profile_mutation_isolation():
    """Mutating profile metadata in result provenance does not affect the
    adapter instance or other results."""
    a = make_adapter(profile="claude-registry-temperature/v1")
    r1 = a.predict_proba(np.zeros((4, 4, 3), np.uint8))
    tp = r1.metadata.provenance["temperature_profile"]
    # Mutate the provenance snapshot
    tp["value"] = 999.0
    tp["key_path"].append("tampered")
    # Instance state unchanged
    assert a.temperature_profile["value"] == REGISTRY_T
    assert a.temperature_profile["key_path"] == ["hsemotion_b0_va", "temperature"]
    # A fresh result is also unaffected
    r2 = a.predict_proba(np.zeros((4, 4, 3), np.uint8))
    assert r2.metadata.provenance["temperature_profile"]["value"] == REGISTRY_T
    assert r2.metadata.provenance["temperature_profile"]["key_path"] == [
        "hsemotion_b0_va",
        "temperature",
    ]


# ---------------------------------------------------------------------------
# Expected source fingerprint (fails closed before model loading)
# ---------------------------------------------------------------------------


def test_expected_source_fingerprint_match_succeeds():
    """A correct expected_source_sha256 allows construction."""
    expected = _file_sha256_bytes(CALIBRATION_V3.read_bytes())
    a = make_adapter(
        profile="claude-registry-temperature/v1",
        expected_source_sha256=expected,
    )
    assert a.temperature == REGISTRY_T


def test_expected_source_fingerprint_mismatch_fails_before_loading():
    """A wrong expected_source_sha256 fails before model loading; no fallback."""
    wrong_sha = "0" * 64
    try:
        make_adapter(
            profile="claude-registry-temperature/v1",
            expected_source_sha256=wrong_sha,
        )
    except ValueError as exc:
        assert "does not match resolved profile source hash" in str(exc)
        assert "failing closed" in str(exc)
    else:
        raise AssertionError("expected ValueError for fingerprint mismatch")


# ---------------------------------------------------------------------------
# Profile change affects probabilities only
# ---------------------------------------------------------------------------


def test_profile_change_alters_probabilities_not_logits_or_va():
    """Changing only profile changes expression probabilities, not input tensor,
    raw logits, V/A, or weights."""
    legacy = make_adapter(profile="opencode-legacy/v1")
    registry = make_adapter(profile="claude-registry-temperature/v1")
    live = make_adapter(profile="claude-live-temperature/v1")

    # Run inference on all so the fake session records the input tensor
    img = np.zeros((4, 4, 3), np.uint8)
    legacy.predict_proba(img)
    registry.predict_proba(img)
    live.predict_proba(img)

    # Same fake session -> same raw logits and V/A
    assert np.allclose(legacy._session.row, registry._session.row)
    assert np.allclose(legacy._session.row, live._session.row)
    assert np.allclose(legacy._session.last_tensor, registry._session.last_tensor)
    assert np.allclose(legacy._session.last_tensor, live._session.last_tensor)

    # Same weights path
    assert legacy.model_path == registry.model_path == live.model_path

    # Expression probabilities must differ across profiles (different temperatures)
    legacy_probs = np.array(list(legacy.predict_proba(img).probabilities.values()))
    registry_probs = np.array(list(registry.predict_proba(img).probabilities.values()))
    live_probs = np.array(list(live.predict_proba(img).probabilities.values()))

    assert not np.allclose(legacy_probs, registry_probs, atol=1e-9)
    assert not np.allclose(legacy_probs, live_probs, atol=1e-9)
    assert not np.allclose(registry_probs, live_probs, atol=1e-9)


def test_native8_and_conditional_fer7_mass_valid_across_profiles():
    """Native8 sums to 1, conditional FER7 sums to 1, retained + contempt = 1
    across all profiles."""
    for profile in (
        "opencode-legacy/v1",
        "claude-registry-temperature/v1",
        "claude-live-temperature/v1",
    ):
        a = make_adapter(profile=profile)
        r = a.predict_proba(np.zeros((4, 4, 3), np.uint8))

        # Conditional FER7 sums to 1 over supported classes
        fer7_sum = float(sum(r.probabilities.values()))
        assert abs(fer7_sum - 1.0) < 1e-9, f"{profile}: FER7 sum {fer7_sum}"

        # Native8 from provenance sums to 1
        native8 = r.metadata.provenance["native8_probabilities"]
        native8_sum = float(sum(native8.values()))
        assert abs(native8_sum - 1.0) < 1e-9, f"{profile}: native8 sum {native8_sum}"

        # Retained + contempt = 1
        retained = r.metadata.provenance["retained_mass"]
        contempt = r.metadata.provenance["contempt_mass"]
        assert abs(retained + contempt - 1.0) < 1e-9, (
            f"{profile}: mass {retained} + {contempt}"
        )

        # V/A preserved (float32 ONNX -> float64 conversion has tiny precision noise)
        assert np.isclose(r.metadata.provenance["valence"], 0.1)
        assert np.isclose(r.metadata.provenance["arousal"], -0.4)


def test_profile_metadata_in_result_provenance():
    """Profile identity, source, selector, SHA256 and value appear in result provenance."""
    expected_sha = _file_sha256_bytes(CALIBRATION_V3.read_bytes())
    a = make_adapter(profile="claude-registry-temperature/v1")
    r = a.predict_proba(np.zeros((4, 4, 3), np.uint8))
    prov = r.metadata.provenance
    assert "temperature_profile" in prov
    tp = prov["temperature_profile"]
    assert tp["profile"] == "claude-registry-temperature/v1"
    assert tp["source_path"] == str(CALIBRATION_V3)
    assert tp["key_path"] == ["hsemotion_b0_va", "temperature"]
    assert tp["value"] == REGISTRY_T
    assert tp["raw_sha256"] == expected_sha
    assert prov["temperature"] == REGISTRY_T


# ---------------------------------------------------------------------------
# Output consistency
# ---------------------------------------------------------------------------


def test_output_top1_and_confidence_consistency():
    """top1 matches argmax of probabilities; confidence == top probability."""
    a = make_adapter(profile="opencode-legacy/v1")
    r = a.predict_proba(np.zeros((4, 4, 3), np.uint8))
    probs = r.probabilities
    expected_top = max(probs, key=probs.get)
    assert r.top1 == expected_top
    assert r.confidence == probs[expected_top]
    assert abs(r.uncertainty - (1.0 - r.confidence)) < 1e-12


def test_profile_read_guard_rejects_nonfixture_source(tmp_path,monkeypatch):
    profile="claude-registry-temperature/v1"
    # A manufactured outside path is rejected before source existence/read checks.
    monkeypatch.setitem(module._HSE_PROFILES,profile,{**module._HSE_PROFILES[profile],"source_path":str(tmp_path.parent/"SYNTHETIC-outside.json")})
    with pytest.raises(AssertionError):make_adapter(profile=profile)
