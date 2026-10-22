"""emolab.adapters.hsemotion — HSEmotion adapter skeleton.

HSEmotion lives in the separate Claude Code Emotion Lab repository and is not
integrated into this repo. This adapter raises a clear error until the Claude
repo is available locally and its weights/licence are verified.

See docs/CLAUDE_EMOTION_LAB.md for the expected repository layout, weight files
and licence requirements.
"""

from __future__ import annotations

from emolab.inference import FaceEmotionAdapter, ModelMetadata, ModelResult

FER7 = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class HSEmotionAdapter(FaceEmotionAdapter):
    """HSEmotion (Claude Code Emotion Lab) — not integrated in this repo."""

    name = "hsemotion"
    version = "0.1"
    input_shape = (224, 224, 3)
    native_labels = tuple(FER7)
    source = "hsemotion (Claude Code Emotion Lab)"
    licence = "unknown — see docs/CLAUDE_EMOTION_LAB.md"
    calibrated = False

    def predict_proba(self, face_rgb: np.ndarray) -> ModelResult:
        raise NotImplementedError(
            "HSEmotion lives in the separate Claude Code Emotion Lab; "
            "integration pending license/weights review "
            "(see docs/CLAUDE_EMOTION_LAB.md)"
        )
