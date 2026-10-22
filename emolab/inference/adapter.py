"""emolab.inference.adapter — face-emotion model adapter contract.

Every facial-expression model (mini-Xception, ViT, HSEmotion, body-language
models, external Claude/Codex models) wraps behind FaceEmotionAdapter and
returns a ModelResult whose probabilities live in a shared class space.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ModelMetadata:
    """Static model description. Never changes between calls."""

    model_id: str
    version: str
    input_shape: tuple[int, int, int]  # (height, width, channels)
    class_mapping: dict[int, str]  # native class index -> class name
    calibrated: bool
    supported_classes: frozenset[str]  # classes the adapter may emit
    provenance: dict[str, str]  # source, licence, weight checksum, etc.


@dataclass(frozen=True)
class ModelResult:
    """One inference result.

    probabilities is keyed by human-readable class names and always sums to 1
    over the adapter's supported classes. confidence is the top-1 probability;
    uncertainty is 1 - confidence.
    """

    probabilities: Mapping[str, float]
    top1: str
    confidence: float
    uncertainty: float
    latency_ms: float
    metadata: ModelMetadata


class FaceEmotionAdapter:
    """Base class for face-emotion model adapters.

    Subclasses must implement predict_proba and declare class attributes:
    name, version, input_shape, native_labels, source, licence, calibrated.
    """

    name: str = "base"
    version: str = "0.0.0"
    input_shape: tuple[int, int, int] = (48, 48, 3)
    native_labels: tuple[str, ...] = ()
    source: str = "unknown"
    licence: str = "unknown"
    calibrated: bool = False

    def predict_proba(self, face_rgb: np.ndarray) -> ModelResult:
        """Return FER-aligned probabilities for a BGR face crop.

        Raises ValueError for malformed input; never silently remaps classes
        outside supported_classes.
        """
        raise NotImplementedError


def _default_metadata(adapter: FaceEmotionAdapter) -> ModelMetadata:
    return ModelMetadata(
        model_id=adapter.name,
        version=adapter.version,
        input_shape=adapter.input_shape,
        class_mapping={i: l for i, l in enumerate(adapter.native_labels)},
        calibrated=adapter.calibrated,
        supported_classes=frozenset(adapter.native_labels),
        provenance={"source": adapter.source, "licence": adapter.licence},
    )


@dataclass(frozen=True)
class InferenceResult:
    """Deprecated alias retained for backwards compatibility."""

    probabilities: Mapping[str, float]
    top1: str
    confidence: float
    uncertainty: float
    latency_ms: float
    metadata: ModelMetadata
