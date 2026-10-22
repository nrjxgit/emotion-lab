"""emolab.inference package: model-adapter contract for facial-expression models.

Adapters convert model-native outputs into a shared FER7-ordered probability
vector so models trained on different class sets can be compared fairly.
"""

from .adapter import FaceEmotionAdapter, ModelResult, ModelMetadata
from .registry import register_adapter, get_adapter, list_adapters

__all__ = [
    "FaceEmotionAdapter",
    "ModelResult",
    "ModelMetadata",
    "register_adapter",
    "get_adapter",
    "list_adapters",
]
