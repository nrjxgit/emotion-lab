"""Claude Code Emotion Lab — FER2013-trained CNN adapter skeleton.

Weights live in the separate Claude Code application repository. This adapter
stub exercises the contract against fixture data until the Claude repo is
available locally.

See docs/CLAUDE_EMOTION_LAB.md for the expected repository layout, weight files
and licence.
"""

from __future__ import annotations

from emolab.inference import FaceEmotionAdapter, ModelMetadata, ModelResult

FER7 = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class ClaudeFER2013CNNAdapter(FaceEmotionAdapter):
    """FER2013-trained CNN from the Claude Code Emotion Lab."""

    name = "claude-fer2013-cnn"
    version = "1.0"
    input_shape = (48, 48, 1)
    native_labels = tuple(FER7)
    source = "Claude Code Emotion Lab"
    licence = "unknown — see docs/CLAUDE_EMOTION_LAB.md"
    calibrated = False

    def predict_proba(self, face_rgb: np.ndarray) -> ModelResult:
        raise NotImplementedError(
            "Claude FER2013 CNN lives in the separate Claude Code Emotion Lab; "
            "integration pending license/weights review "
            "(see docs/CLAUDE_EMOTION_LAB.md)"
        )
