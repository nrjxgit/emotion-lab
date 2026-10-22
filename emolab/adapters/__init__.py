"""Framework-independent model adapters.

Wraps models from this repo (mini-Xception, ViT) and stubs for external
projects (HSEmotion / Claude Code Emotion Lab / Codex body-language). Adapters
expose the emolab.inference contract; they never train or modify weights.
"""

from .claude_combined import ClaudeCombinedAdapter
from .claude_fer2013_cnn import ClaudeFER2013CNNAdapter
from .claude_hsemotion import HSEmotionAdapter
from .codex_body import CodexBodyAdapter
from .fusion_stub import fused_side_by_side
from .mini_xception import MiniXceptionAdapter
from .vit import ViTAdapter

from emolab.inference import register_adapter

register_adapter("mini-xception", MiniXceptionAdapter)
register_adapter("vit", ViTAdapter)
register_adapter("hsemotion", HSEmotionAdapter)
register_adapter("claude-fer2013-cnn", ClaudeFER2013CNNAdapter)
register_adapter("claude-combined", ClaudeCombinedAdapter)
register_adapter("codex-body", CodexBodyAdapter)

__all__ = [
    "MiniXceptionAdapter",
    "ViTAdapter",
    "HSEmotionAdapter",
    "ClaudeFER2013CNNAdapter",
    "ClaudeCombinedAdapter",
    "CodexBodyAdapter",
    "fused_side_by_side",
]
