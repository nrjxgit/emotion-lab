"""Adapter registry (EL-006 S2): lets a product swap adapters without touching state.py.

Registering a factory grants nothing: ``open_adapter`` always runs the rights
gate, requires the registry key to equal the adapter's declared name, and a new
adapter name must also be on the reviewed allowlist. The gate checks declarations,
not behaviour (see rights.py).
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Callable

from .declarations import CremaDDeclaration, PilotDeclaration, RavdessDeclaration
from .rights import DatasetAdapter, RightsRefusal, authorize
from .synthetic_adapter import SyntheticAdapter
from .synthetic_commercial_adapter import SyntheticCommercialCleanAdapter

_REGISTRY: dict = {
    "synthetic": SyntheticAdapter,
    "synthetic_commercial_clean": SyntheticCommercialCleanAdapter,
    "ravdess": RavdessDeclaration,
    "crema_d": CremaDDeclaration,
    "pilot_consented": PilotDeclaration,
}
_FACTORIES = MappingProxyType(_REGISTRY)  # read-only view; only register_adapter adds entries


def register_adapter(name: str, factory: Callable[[], DatasetAdapter]) -> None:
    """Add a factory. Existing names cannot be replaced."""
    if name in _REGISTRY:
        raise ValueError(f"adapter {name!r} is already registered")
    _REGISTRY[name] = factory


def registered_names() -> tuple:
    return tuple(sorted(_FACTORIES))


def open_adapter(name: str, use: str, run_scope: str, **adapter_kwargs) -> DatasetAdapter:
    """Build the named adapter and return it only if the gate allows ``use`` under ``run_scope``."""
    factory = _FACTORIES.get(name)
    if factory is None:
        raise RightsRefusal("DATASET_NOT_ALLOWLISTED", f"no adapter registered as {name!r}")
    adapter = factory(**adapter_kwargs)
    if getattr(adapter, "name", None) != name:  # the registry key must be the declared dataset name
        raise RightsRefusal("ADAPTER_METADATA_MISMATCH", f"registry key {name!r} does not match the adapter's declared name")
    authorize(adapter, use, run_scope)
    return adapter
