"""emolab.inference: model-adapter contract for facial-expression models.

Adapters convert model-native outputs into a shared, class-labelled probability
vector so models trained on different class sets can be compared fairly. Nothing
here is model-specific: adapters bring their own weights, preprocessing and class
mappings; this package only defines the contract they must honour.
"""

from .adapter import FaceEmotionAdapter, ModelMetadata, ModelResult
from .registry import get_adapter, list_adapters, register_adapter

__all__ = [
    "FaceEmotionAdapter",
    "ModelMetadata",
    "ModelResult",
    "register_adapter",
    "get_adapter",
    "list_adapters",
]
