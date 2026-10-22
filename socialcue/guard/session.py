"""Opaque, revocable, expiring capability checks for manufactured records ONLY.

Public construction issues no authority for real data. This is not a sandbox
against arbitrary Python code; only this synthetic adapter is wired here.
"""
from typing import NamedTuple
import math
import time
from ..data.synthetic_adapter import SyntheticAdapter
from .common import actor_id, actors, stage as valid_stage, refuse

_SEAL = object()
MAX_SESSION_TTL_S = 600.0  # synthetic capability lifetime bound, not an HB-1 threshold


def _finite_time(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        return False


class _Capability:
    __slots__ = ()

    def __init__(self, seal):
        if seal is not _SEAL:
            refuse("CAPABILITY_INVALID", "capabilities are issued by a session")

    def __reduce__(self):
        raise TypeError("payload capabilities cannot be serialized")


class _SyntheticItem(NamedTuple):
    item_id: str
    group: str
    scenario: str
    seed: int

    def records(self, session=None, capability=None, *, stage=None):
        if type(session) is not SyntheticPayloadSession:
            refuse("CAPABILITY_REQUIRED", "a live synthetic session is required")
        return session.decode(self, capability, stage=stage)


class GuardedSyntheticAdapter(SyntheticAdapter):
    """Metadata-only constructor/participants/items; records need a live capability.

    Inherits all four DatasetAdapter implementations, including _participants;
    the existing final rights-gated enumeration methods remain unchanged.
    """

    def _items(self):
        return [_SyntheticItem(i.item_id, i.group, i.scenario, i.seed) for i in super()._items()]


class SyntheticPayloadSession:
    """Self-trusting until supervisor exists; no ledger/authority/protocol admission.

    Synthetic actors/stage are caller declarations. Constructor samples a pure
    monotonic clock, requires expiry within 600 s, performs no application IO.
    Session copy/deepcopy/pickle is blocked. Tokens are single-use and bind an
    immutable metadata snapshot; expiry/revoke is checked on each issue/decode.
    This object cannot authorize real data. A future supervisor must replace
    self-trust with authenticated ledger/stage/protocol/authority admission.
    """

    authority_model = "SELF_TRUSTED_SYNTHETIC_ONLY"

    def __copy__(self):
        raise TypeError("sessions cannot be copied")

    def __deepcopy__(self, memo):
        raise TypeError("sessions cannot be copied")

    def __reduce_ex__(self, protocol):
        raise TypeError("sessions cannot be serialized")

    def __init__(self, actor_ids, stage, expires_at, *, clock=time.monotonic):
        self._actors = actors(actor_ids)
        self._stage = valid_stage(stage)
        if not _finite_time(expires_at):
            refuse("SESSION_INVALID", "finite expiry required")
        now = clock()  # pure monotonic sampling is not application IO
        if (not _finite_time(now)
                or not 0 < expires_at - now <= MAX_SESSION_TTL_S):
            refuse("SESSION_INVALID", "expiry must be within 600 seconds of the monotonic clock")
        self._expiry, self._clock = expires_at, clock
        self._epoch, self._revoked, self._last = 0, False, now
        self._issued = {}

    def _live(self, stage):
        if self._revoked:
            refuse("SESSION_REVOKED", "session closed")
        if stage != self._stage:
            refuse("STAGE_MISMATCH", "capability belongs to another stage")
        now = self._clock()
        if not _finite_time(now):
            refuse("SESSION_INVALID", "invalid clock")
        if self._last is not None and now < self._last:
            self.revoke()
            refuse("SESSION_REVOKED", "clock moved backwards")
        self._last = now
        if now >= self._expiry:
            self.revoke()
            refuse("SESSION_EXPIRED", "session expired")

    def _item(self, item):
        from ..synthetic import SCENARIOS
        if type(item) is not _SyntheticItem or actor_id(item.group) not in self._actors:
            refuse("ACTOR_MISMATCH", "item outside synthetic actor set")
        if (type(item.item_id) is not str or type(item.group) is not str
                or type(item.scenario) is not str or item.scenario not in SCENARIOS or type(item.seed) is not int):
            refuse("CAPABILITY_INVALID", "invalid synthetic metadata")
        return (item.item_id, item.group, item.scenario, item.seed)

    def issue(self, item, *, stage):
        self._live(stage)
        binding = self._item(item)
        cap = _Capability(_SEAL)
        self._issued[cap] = (item, self._epoch, binding)
        return cap

    def decode(self, item, capability, *, stage):
        self._live(stage)
        binding = self._item(item)
        if (type(capability) is not _Capability or capability not in self._issued
                or self._issued[capability][0] is not item
                or self._issued[capability][1] != self._epoch
                or self._issued[capability][2] != binding):
            refuse("CAPABILITY_INVALID", "missing, forged or wrong-object capability")
        del self._issued[capability]  # single-use, even if generation subsequently fails
        from ..synthetic import generate
        result = generate(item.scenario, seed=item.seed)
        self._live(stage)  # do not return work that expired or was revoked during generation
        if self._item(item) != binding:
            refuse("CAPABILITY_INVALID", "item metadata changed during generation")
        return result

    def revoke(self):
        self._revoked = True
        self._epoch += 1
        self._issued.clear()

    close = revoke
