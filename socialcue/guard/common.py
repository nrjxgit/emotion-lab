"""Pure validation for the synthetic guard namespace."""
import re
from ..data.rights import RightsRefusal

STAGES = frozenset({"PILOT", "DEV_VAL", "HELD_OUT"})


def refuse(code, reason):
    raise RightsRefusal(code, reason)


def actor_id(value):
    """Canonical synthetic actor identity, never protocol-dependent or a real actor."""
    if not isinstance(value, str):
        refuse("ACTOR_INVALID", "synthetic actor string required")
    match = re.fullmatch(r"synthetic-actor-([0-9]{1,8})", value, re.I)
    if not match:
        refuse("ACTOR_INVALID", "only synthetic-actor-N identities are admitted")
    return f"synthetic-actor-{int(match[1]):04d}"


def actors(values):
    if isinstance(values, (str, bytes)):
        refuse("ACTOR_INVALID", "actor collection required")
    try:
        result = tuple(sorted({actor_id(a) for a in values}))
    except TypeError:
        refuse("ACTOR_INVALID", "actor collection required")
    if not result:
        refuse("ACTOR_INVALID", "empty actor set")
    return result


def stage(value):
    if not isinstance(value, str) or value not in STAGES:
        refuse("STAGE_INVALID", "PILOT, DEV_VAL or HELD_OUT required")
    return value
