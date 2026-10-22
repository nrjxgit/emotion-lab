"""Portable HSEmotion profile integrity regression tests.

Adapts the coordinator's independent review cases so they do not depend on
machine-specific Claude reference paths, real weights, or dataset assets.
Native source profiles (registry/live) are exercised against temporary
synthetic JSON/YAML files under tmp_path via monkeypatch of the module-level
_HSE_PROFILES registry.

No model weights, datasets, or inference runtimes are loaded.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Locate repository/module relative to this test file; never hardcode paths.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ADAPTER_PATH = REPO_ROOT / "emolab" / "adapters" / "claude_hsemotion.py"

spec = importlib.util.spec_from_file_location("hse_profile_integrity", ADAPTER_PATH)
hse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hse)

HSEmotionAdapter = hse.HSEmotionAdapter
HSE_PROFILES = hse._HSE_PROFILES
LEGACY_TEMP = hse._HSE_LEGACY_TEMPERATURE


class FakeSession:
    """Minimal fake ONNX session returning a fixed [1, 10] output row."""

    def __init__(self, row):
        self.row = np.asarray(row, dtype=np.float32).reshape(1, -1)

    def get_inputs(self):
        return [SimpleNamespace(name="synthetic_input")]

    def run(self, _outputs, inputs):
        return [self.row.copy()]


# ---------------------------------------------------------------------------
# 1. Numeric class-level temperature read stays the legacy default
# ---------------------------------------------------------------------------
def test_legacy_class_temperature_read_stays_numeric():
    assert HSEmotionAdapter.temperature == LEGACY_TEMP
    assert isinstance(HSEmotionAdapter.temperature, float)


# ---------------------------------------------------------------------------
# 2. Correct expected_source_sha256 accepted for default + all 3 profiles
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("profile", [None, "opencode-legacy/v1"])
def test_correct_pin_accepted_legacy_and_default(profile):
    # Legacy profile has no external source file; its hash is the adapter source.
    model = HSEmotionAdapter("SYNTHETIC_DO_NOT_OPEN", temperature_profile=profile)
    pinned = HSEmotionAdapter(
        "SYNTHETIC_DO_NOT_OPEN",
        temperature_profile=profile,
        expected_source_sha256=model.temperature_profile["raw_sha256"],
    )
    assert pinned.temperature_profile == model.temperature_profile
    assert pinned._session is None


@pytest.mark.parametrize(
    "profile", ["claude-registry-temperature/v1", "claude-live-temperature/v1"]
)
def test_correct_pin_accepted_native_profiles(profile, tmp_path, monkeypatch):
    # Monkeypatch native profiles to temp synthetic files (no Claude reference dep).
    spec_entry = dict(HSE_PROFILES[profile])
    if profile == "claude-registry-temperature/v1":
        content = json.dumps({"hsemotion_b0_va": {"temperature": 1.6427638433649585}})
        suffix = ".json"
    else:
        content = "face:\n  expression:\n    temperature: 1.3407\n"
        suffix = ".yaml"
    src = tmp_path / f"synthetic-profile{suffix}"
    src.write_text(content, encoding="utf-8")
    spec_entry["source_path"] = str(src)
    monkeypatch.setitem(HSE_PROFILES, profile, spec_entry)

    model = HSEmotionAdapter("SYNTHETIC_DO_NOT_OPEN", temperature_profile=profile)
    pinned = HSEmotionAdapter(
        "SYNTHETIC_DO_NOT_OPEN",
        temperature_profile=profile,
        expected_source_sha256=model.temperature_profile["raw_sha256"],
    )
    assert pinned.temperature_profile == model.temperature_profile
    assert pinned._session is None
    assert (
        model.temperature_profile["raw_sha256"]
        == hashlib.sha256(src.read_bytes()).hexdigest()
    )


# ---------------------------------------------------------------------------
# 3. Wrong expected_source_sha256 rejected for default and explicit legacy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("profile", [None, "opencode-legacy/v1"])
def test_wrong_pin_rejected_default_and_legacy(profile):
    with pytest.raises(ValueError, match="fingerprint|hash|sha256"):
        HSEmotionAdapter(
            "SYNTHETIC_DO_NOT_OPEN",
            temperature_profile=profile,
            expected_source_sha256="0" * 64,
        )


# ---------------------------------------------------------------------------
# 4. Post-construction temperature mutation rejected for all 3 profiles
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "profile",
    [
        "opencode-legacy/v1",
        "claude-registry-temperature/v1",
        "claude-live-temperature/v1",
    ],
)
def test_temperature_cannot_silently_diverge(profile, tmp_path, monkeypatch):
    if profile != "opencode-legacy/v1":
        spec_entry = dict(HSE_PROFILES[profile])
        if profile == "claude-registry-temperature/v1":
            content = json.dumps(
                {"hsemotion_b0_va": {"temperature": 1.6427638433649585}}
            )
            suffix = ".json"
        else:
            content = "face:\n  expression:\n    temperature: 1.3407\n"
            suffix = ".yaml"
        src = tmp_path / f"synthetic-profile{suffix}"
        src.write_text(content, encoding="utf-8")
        spec_entry["source_path"] = str(src)
        monkeypatch.setitem(HSE_PROFILES, profile, spec_entry)

    model = HSEmotionAdapter("SYNTHETIC_DO_NOT_OPEN", temperature_profile=profile)
    model._session = FakeSession([0, 1, 2, 3, 4, 5, 6, 7, 0.2, -0.3])

    with pytest.raises((ValueError, AttributeError)):
        model.temperature = model.temperature + 1
        model.predict_proba(np.zeros((2, 2, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# 5. Invalid temperature values fail via constructor (synthetic data only)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value",
    [0, -1, float("nan"), float("inf"), float("-inf"), True, "not-numeric", None],
)
def test_invalid_temperature_rejected_via_constructor(tmp_path, monkeypatch, value):
    src = tmp_path / "synthetic-profile.json"
    src.write_text(
        json.dumps({"hsemotion_b0_va": {"temperature": value}}), encoding="utf-8"
    )
    spec_entry = dict(HSE_PROFILES["claude-registry-temperature/v1"])
    spec_entry["source_path"] = str(src)
    monkeypatch.setitem(HSE_PROFILES, "claude-registry-temperature/v1", spec_entry)

    with pytest.raises(ValueError):
        HSEmotionAdapter(
            "SYNTHETIC_DO_NOT_OPEN",
            temperature_profile="claude-registry-temperature/v1",
        )


# ---------------------------------------------------------------------------
# 6. One-read snapshot: hash and value bind to same byte snapshot when file
#    bytes change immediately after the captured read.
# ---------------------------------------------------------------------------
def test_one_read_snapshot_hash_value_consistency(tmp_path, monkeypatch):
    src = tmp_path / "synthetic-profile.json"
    first = b'{"hsemotion_b0_va":{"temperature":2.0}}'
    second = b'{"hsemotion_b0_va":{"temperature":3.0}}'
    src.write_bytes(first)

    spec_entry = dict(HSE_PROFILES["claude-registry-temperature/v1"])
    spec_entry["source_path"] = str(src)
    monkeypatch.setitem(HSE_PROFILES, "claude-registry-temperature/v1", spec_entry)

    original_read_bytes = Path.read_bytes
    reads = []

    def replacing_read(self):
        if self == src:
            data = original_read_bytes(self)
            reads.append(data)
            self.write_bytes(second)
            return data
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", replacing_read)

    model = HSEmotionAdapter(
        "SYNTHETIC_DO_NOT_OPEN", temperature_profile="claude-registry-temperature/v1"
    )
    assert reads == [first]
    assert model.temperature == 2.0
    assert model.temperature_profile["raw_sha256"] == hashlib.sha256(first).hexdigest()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
