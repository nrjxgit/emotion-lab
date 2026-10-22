"""Adapter registry.

Adapters self-register by name. get_adapter(name) returns a fresh instance;
list_adapters() returns the names in registration order.
"""

from typing import Any

from .adapter import FaceEmotionAdapter

_REGISTRY: dict[str, type[FaceEmotionAdapter]] = {}


def register_adapter(name: str, cls: type[FaceEmotionAdapter]) -> None:
    """Register an adapter class under name.

    Raises ValueError if name is already registered or cls does not subclass
    FaceEmotionAdapter.
    """
    if not (isinstance(cls, type) and issubclass(cls, FaceEmotionAdapter)):
        raise ValueError(f"{cls!r} is not a FaceEmotionAdapter subclass")
    if name in _REGISTRY:
        raise ValueError(f"adapter {name!r} already registered")
    _REGISTRY[name] = cls


def get_adapter(name: str) -> FaceEmotionAdapter:
    """Instantiate and return the registered adapter under name.

    Raises ValueError if no adapter with that name is registered.
    """
    if name not in _REGISTRY:
        raise ValueError(f"unknown adapter {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def list_adapters() -> list[str]:
    """Return registered adapter names in registration order."""
    return list(_REGISTRY)
