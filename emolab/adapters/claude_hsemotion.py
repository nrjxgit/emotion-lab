"""Claude Emotion Lab - HSEmotion adapter.

Wraps the oarriaga FER2013-trained EfficientNet-B0 (enet_b0_8_va_mtl.onnx, Apache-2.0),
the same frozen checkpoint used by the Claude Code Emotion Lab research platform.
Preprocessing mirrors oarriaga: RGB 224x224 bilinear, /255, ImageNet mean/std,
output 8 expression logits + valence + arousal via ONNX Runtime.

The ONNX head returns [batch, 10]: indices 0-7 are expression logits in HSE8 order
(anger, contempt, disgust, fear, happiness, neutral, sadness, surprise); index 8 is
valence, index 9 is arousal. Expression softmax is applied ONLY to indices 0-7;
valence and arousal are preserved separately and never enter expression probabilities.

Output contract (emolab.inference/1.0): probabilities over the 7 supported FER7 classes
(sum to 1, conditional on dropping contempt). The retained mass (1 - p(contempt)) and
contempt mass are disclosed in metadata.provenance. The adapter does NOT silently discard
contempt mass; callers can reconstruct the unrenormalised 7-class vector by multiplying
each reported probability by retained_mass.

Color boundary: predict_proba(face_rgb) accepts a BGR uint8 face crop (matching the
base-class contract) and converts to RGB before the ONNX model, per Emotion Lab's
RGB model-function rule. The parameter name `face_rgb` is preserved for keyword
compatibility with existing callers; the docstring documents the BGR input boundary.

Temperature profiles (SOURCE-TEMPERATURE policies, not pipeline-equivalence claims):

    opencode-legacy/v1          OpenCode legacy constant 1.3407638433649585.
                                This value occurs in NO committed Emotion Lab source.
                                OpenCode's own source file (this module) and the
                                constant selector are its provenance.
    claude-registry-temperature/v1   frozen models/calibration_v3.json entry
                                    hsemotion_b0_va.temperature (= 1.6427638433649585)
    claude-live-temperature/v1   frozen configs/default.yaml entry
                                    face.expression.temperature (= 1.3407)

The profile selector is keyword-only. When no profile is supplied, the legacy default
temperature (1.3407638433649585) is used, preserving existing callers and behaviour.
Profile identity, source path, selector, raw SHA256 and resolved value are captured
in immutable instance state and emitted in result provenance. Invalid, unknown,
missing, nonfinite, nonpositive or boolean temperature inputs fail clearly with no
fallback. If an expected_source_sha256 is supplied, it must match the resolved
profile's source hash before model loading; mismatch fails closed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from emolab.inference import FaceEmotionAdapter, ModelMetadata, ModelResult

# FER7 order used by the adapter's supported output (7 classes, no contempt)
FER7 = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# HSEmotion native head order (8 classes, includes contempt at index 1)
_HSE8 = [
    "angry",
    "contempt",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
]

# HSE8 index for each FER7 name (by NAME, not position; skips contempt at index 1)
_HSE8_IDX_FOR_FER7 = {
    "angry": 0,
    "disgust": 2,
    "fear": 3,
    "happy": 4,  # HSE8 "happiness"
    "sad": 6,  # HSE8 "sadness"
    "surprise": 7,
    "neutral": 5,
}

# Supported HSE8 indices (all except contempt at index 1), in FER7 name order
_SUPPORTED_HSE8_INDICES = [_HSE8_IDX_FOR_FER7[name] for name in FER7]

# ImageNet normalization (matches HSEmotion training)
_IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Claude Emotion Lab legacy OpenCode temperature constant
_HSE_LEGACY_TEMPERATURE = 1.3407638433649585

# Frozen Claude reference repository paths (read-only, never modified)
_CLAUDE_REPO = Path(
    os.environ.get("EMOLAB_CLAUDE_REFERENCE_REPO", "emotion-lab-claude-reference")
)
_CLAUDE_CALIBRATION_V3 = _CLAUDE_REPO / "models" / "calibration_v3.json"
_CLAUDE_DEFAULT_YAML = _CLAUDE_REPO / "configs" / "default.yaml"

# Adapter source file path (for legacy profile self-identification)
_ADAPTER_SOURCE_PATH = Path(__file__).resolve()

# Temperature profile registry: name -> spec
# These name SOURCE-TEMPERATURE policies, not claims that detector/alignment/crop
# pipelines match across sources.
# Each spec: description, source_path (None for legacy), key_path (list[str]),
# value (float, legacy only), source_sha256 (str or None).
_HSE_PROFILES = {
    "opencode-legacy/v1": {
        "description": "OpenCode legacy temperature constant (default). "
        "Not present in any committed Emotion Lab source.",
        "source_path": None,
        "key_path": ["_HSE_LEGACY_TEMPERATURE"],
        "value": _HSE_LEGACY_TEMPERATURE,
        "source_sha256": None,
    },
    "claude-registry-temperature/v1": {
        "description": "Claude registry calibration_v3.json temperature. "
        "Registry/research path only; never used by the live app for hsemotion_b0_va.",
        "source_path": str(_CLAUDE_CALIBRATION_V3),
        "key_path": ["hsemotion_b0_va", "temperature"],
        "value": None,
        "source_sha256": None,
    },
    "claude-live-temperature/v1": {
        "description": "Claude live runtime default.yaml temperature. "
        "Deployed default in v0.4.1.",
        "source_path": str(_CLAUDE_DEFAULT_YAML),
        "key_path": ["face", "expression", "temperature"],
        "value": None,
        "source_sha256": None,
    },
}


def _file_sha256_bytes(data: bytes) -> str:
    """Return the SHA256 hex digest of file bytes."""
    return hashlib.sha256(data).hexdigest()


def _load_profile_source(source_path: str) -> tuple[dict[str, Any], str]:
    """Read ONE byte snapshot of a profile source file, hash it, and parse it.

    Returns (parsed_data, raw_sha256). The hash and the parsed value are bound
    to the same byte snapshot so a file cannot be hashed at one version and
    parsed at another.

    Raises ValueError if the file is missing, unreadable, or unparseable.
    """
    src = Path(source_path)
    if not src.exists():
        raise ValueError(f"temperature profile source not found: {src}")
    try:
        raw = src.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read temperature source {src}: {exc}") from exc

    raw_sha256 = _file_sha256_bytes(raw)
    text = raw.decode("utf-8")

    try:
        if src.suffix.lower() == ".json":
            data = json.loads(text)
        elif src.suffix.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(text)
        else:
            raise ValueError(f"unsupported temperature source format: {src.suffix}")
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"failed to parse temperature source {src}: {exc}") from exc

    return data, raw_sha256


def _validate_temperature_value(
    value: Any, profile_name: str, source_path: str | None, key_path: list[str] | None
) -> float:
    """Validate and coerce a temperature value.

    Rejects booleans (float(True) == 1.0 would silently corrupt calibration),
    non-numeric values, nonfinite values, and nonpositive values.
    """
    # Reject booleans explicitly: float(True) == 1.0 must not become a temperature
    if isinstance(value, bool):
        raise ValueError(
            f"temperature for profile {profile_name!r} is a boolean "
            f"({value!r}); booleans are not valid temperatures"
        )

    try:
        t = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"temperature value for profile {profile_name!r} at "
            f"{key_path} in {source_path} is not numeric: {value!r}"
        ) from exc

    if not np.isfinite(t) or t <= 0.0:
        raise ValueError(
            f"temperature profile {profile_name!r} resolves to nonfinite or "
            f"nonpositive value {t!r} from {source_path}"
        )

    return t


def _resolve_profile_temperature(profile_name: str) -> dict[str, Any]:
    """Resolve a named temperature profile to its source, value and metadata.

    Returns a dict with keys: profile, source_path, key_path, value, raw_sha256,
    description. Raises ValueError for unknown, missing, nonfinite, nonpositive,
    boolean, or unparseable temperatures. No fallback.

    The source file is hashed and parsed from a single byte snapshot.
    """
    if profile_name not in _HSE_PROFILES:
        raise ValueError(
            f"unknown HSE temperature profile: {profile_name!r}; "
            f"valid profiles: {sorted(_HSE_PROFILES)}"
        )

    spec = _HSE_PROFILES[profile_name]
    source_path = spec["source_path"]
    key_path = spec["key_path"]
    legacy_value = spec.get("value")

    if source_path is None:
        # Legacy OpenCode constant profile: no external source file
        value = _validate_temperature_value(legacy_value, profile_name, None, None)
        return {
            "profile": profile_name,
            "source_path": str(_ADAPTER_SOURCE_PATH),
            "key_path": ["_HSE_LEGACY_TEMPERATURE"],
            "value": value,
            "raw_sha256": _file_sha256_bytes(_ADAPTER_SOURCE_PATH.read_bytes()),
            "description": spec["description"],
        }

    src = Path(source_path)
    data, raw_sha256 = _load_profile_source(source_path)

    # Navigate key path on the parsed data
    node: Any = data
    for key in key_path:
        if not isinstance(node, dict) or key not in node:
            raise ValueError(
                f"temperature key path {key_path} not found in {src} "
                f"for profile {profile_name!r}"
            )
        node = node[key]

    value = _validate_temperature_value(node, profile_name, source_path, key_path)

    return {
        "profile": profile_name,
        "source_path": str(src),
        "key_path": list(key_path),
        "value": value,
        "raw_sha256": raw_sha256,
        "description": spec["description"],
    }


def _logsumexp(x: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


class HSEmotionAdapter(FaceEmotionAdapter):
    """HSEmotion EfficientNet-B0 (oarriaga face_classification, MIT),
    the same frozen checkpoint used by the Claude Code Emotion Lab.

    Preprocessing: BGR 224x224 bilinear -> RGB -> /255 -> ImageNet mean/std.
    Output: 8 expression logits + valence + arousal via ONNX Runtime.
    Expression softmax over first 8 logits only, temperature selected by profile.
    Reports 7 FER7 classes (contempt excluded); probabilities sum to 1 over supported
    classes. Retained mass and contempt mass are disclosed in metadata.provenance.
    """

    name = "claude-hsemotion"
    version = "0.4.1"
    input_shape = (224, 224, 3)
    native_labels = tuple(_HSE8)
    source = "Claude Code Emotion Lab"
    licence = "Apache-2.0 code; AffectNet training data is research-only"
    calibrated = False
    temperature = _HSE_LEGACY_TEMPERATURE

    def __init__(
        self,
        model_path: str | None = None,
        *,
        temperature_profile: str | None = None,
        expected_source_sha256: str | None = None,
    ):
        self.model_path = model_path or self._find_weights()

        # Resolve the temperature profile BEFORE model loading.
        # The expected_source_sha256 fingerprint check applies to ALL branches
        # (including the default legacy branch); mismatch fails closed with
        # no fallback before model loading.
        if temperature_profile is None:
            resolved_profile = _resolve_profile_temperature("opencode-legacy/v1")
            resolved_temperature = _HSE_LEGACY_TEMPERATURE
        else:
            resolved_profile = _resolve_profile_temperature(temperature_profile)
            resolved_temperature = resolved_profile["value"]

        if expected_source_sha256 is not None:
            resolved_hash = resolved_profile.get("raw_sha256")
            if resolved_hash != expected_source_sha256:
                raise ValueError(
                    f"expected source fingerprint {expected_source_sha256} "
                    f"does not match resolved profile source hash "
                    f"{resolved_hash} for profile "
                    f"{temperature_profile or 'opencode-legacy/v1'!r}; "
                    f"failing closed before model loading"
                )

        self._profile = resolved_profile
        # Bind temperature to the resolved profile value at construction.
        # object.__setattr__ bypasses the __setattr__ guard so the first
        # (construction-time) assignment succeeds; later reassignment fails.
        object.__setattr__(self, "temperature", resolved_temperature)

        self._session = None

    def _find_weights(self) -> str:
        """Locate the HSEmotion ONNX weights in the Claude reference repo."""
        onnx_path = _CLAUDE_REPO / "models" / "weights" / "enet_b0_8_va_mtl.onnx"
        if onnx_path.exists():
            return str(onnx_path)
        raise FileNotFoundError("HSEmotion ONNX weights not found in Claude repo")

    @property
    def temperature_profile(self) -> dict[str, Any]:
        """Return a deep, independent, serializable copy of the resolved profile.

        Callers cannot mutate the instance's stored profile through this
        reference, and nested key_path lists are not shared.
        """
        return copy.deepcopy(self._profile)

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent silent reassignment of temperature after construction.

        temperature is bound to the resolved named profile's value at
        construction; changing it after the fact would let inference diverge
        from temperature_profile.value while provenance still reports the old
        named value. Class-level reads (HSEmotionAdapter.temperature) remain
        the numeric legacy default.
        """
        if name == "temperature" and "temperature" in self.__dict__:
            raise AttributeError(
                "temperature is read-only after construction; it is bound to "
                "the named temperature_profile value. Use a different "
                "temperature_profile to select a different source value."
            )
        super().__setattr__(name, value)

    def _load(self):
        if self._session is None:
            import onnxruntime as ort

            self._session = ort.InferenceSession(self.model_path)
        return self._session

    def predict_proba(self, face_rgb: np.ndarray) -> ModelResult:
        session = self._load()
        t0 = time.perf_counter()

        # Preprocess: BGR 224x224 bilinear -> RGB -> /255 -> ImageNet mean/std
        # Parameter name `face_rgb` preserved for keyword compatibility; input is BGR
        # per the base-class contract (Emotion Lab model functions take RGB internally).
        bgr = np.asarray(face_rgb)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.float32) / 255.0
        x = (x - _IMNET_MEAN) / _IMNET_STD
        x = x.transpose(2, 0, 1)[None, :]  # NCHW

        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: x})
        row = np.asarray(outputs[0][0], dtype=np.float64)

        # Split: indices 0-7 are expression logits (HSE8), 8 is valence, 9 is arousal.
        # Expression softmax over the first 8 logits ONLY; valence/arousal never enter
        # expression probabilities.
        expression_logits = row[:8].astype(np.float64)
        valence = float(row[8]) if len(row) > 8 else None
        arousal = float(row[9]) if len(row) > 9 else None

        # Temperature-scaled logits (float64 for stability)
        all_scaled = expression_logits / self.temperature

        # Supported HSE8 indices (all except contempt at index 1)
        supported_scaled = all_scaled[_SUPPORTED_HSE8_INDICES]

        # Log-sum-exp for stable mass computation (avoids 1 - p_contempt cancellation)
        log_sum_all = _logsumexp(all_scaled)
        log_sum_supported = _logsumexp(supported_scaled)
        retained_mass = float(np.exp(log_sum_supported - log_sum_all))
        contempt_mass = 1.0 - retained_mass

        if retained_mass <= 0.0:
            raise ValueError(
                "zero retained mass: contempt dominates all expression logits; "
                "no supported FER7 class has non-zero probability"
            )

        # Native 8-class distribution (float64; contempt may round to ~1.0 but that's
        # only for disclosure, not for conditional computation)
        native8 = np.exp(all_scaled - log_sum_all)

        # Conditional FER7: softmax over supported logits only (stable, matches
        # direct computation from supported logits). Renormalise by retained mass.
        conditional = np.exp(supported_scaled - log_sum_supported)
        # conditional[i] corresponds to FER7[i] because _SUPPORTED_HSE8_INDICES
        # is built in FER7 name order
        probs: dict[str, float] = {}
        for i, fname in enumerate(FER7):
            probs[fname] = float(conditional[i])

        top1_name = max(probs, key=probs.get)

        # Deep copy of profile metadata into result provenance so callers cannot
        # mutate the instance's stored profile through the returned result.
        profile_snapshot = copy.deepcopy(self._profile)

        return ModelResult(
            probabilities=probs,
            top1=top1_name,
            confidence=float(probs[top1_name]),
            uncertainty=float(1.0 - probs[top1_name]),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            metadata=ModelMetadata(
                model_id=self.name,
                version=self.version,
                input_shape=self.input_shape,
                class_mapping={i: l for i, l in enumerate(_HSE8)},
                calibrated=self.calibrated,
                supported_classes=frozenset(FER7),
                provenance={
                    "source": self.source,
                    "licence": self.licence,
                    "weights": self.model_path,
                    "temperature": self.temperature,
                    "temperature_profile": profile_snapshot,
                    "original_classes": list(_HSE8),
                    "retained_mass": retained_mass,
                    "contempt_mass": contempt_mass,
                    "conditional_fer7": True,
                    "valence": valence,
                    "arousal": arousal,
                    "native8_probabilities": {
                        name: float(p) for name, p in zip(_HSE8, native8)
                    },
                },
            ),
        )
