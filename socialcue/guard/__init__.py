"""Dormant SYNTHETIC-only guards; no real launch authority and no import IO."""

from .ledger import ExposureLedger, ExposureOverride, LedgerCheckpoint, require_external_checkpoint
from .session import GuardedSyntheticAdapter, SyntheticPayloadSession
from .close import verify_grant_closed

__all__ = ["ExposureLedger", "ExposureOverride", "LedgerCheckpoint",
           "GuardedSyntheticAdapter", "SyntheticPayloadSession", "verify_grant_closed", "require_external_checkpoint"]
