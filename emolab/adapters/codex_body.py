"""Codex body-language adapter skeleton.

Codex's body-language project lives in a separate repository. This adapter
stub raises NotImplementedError until the Codex repo is available locally and
its weights/licence are verified.

See docs/CODEX_BODY.md for the expected repository layout, weight files and
licence requirements.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from emolab.inference import FaceEmotionAdapter, ModelMetadata, ModelResult

FER7 = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class CodexBodyAdapter(FaceEmotionAdapter):
    """Codex body-language model — not integrated in this repo."""

    name = "codex-body"
    version = "0.1"
    input_shape = (224, 224, 3)
    native_labels = tuple(FER7)
    source = "Codex body-language project"
    licence = "unknown — see docs/CODEX_BODY.md"
    calibrated = False

    def predict_proba(self, face_rgb: np.ndarray) -> ModelResult:
        raise NotImplementedError(
            "Codex body-language model lives in the separate Codex repository; "
            "integration pending license/weights review "
            "(see docs/CODEX_BODY.md)"
        )
