"""Pure snapshot check, NOT process drain, cache revocation or grant rollback."""
from collections.abc import Mapping
from .common import refuse

REAL_ADAPTERS = frozenset({"ravdess", "crema_d", "casme2", "samm", "smic", "pilot_consented"})


def verify_grant_closed(allowlist):
    """Check a complete parsed {name: entry} snapshot; no IO/cache lookup occurs.

    Unknown extra entries also cannot be granted. Missing known real entries
    fail closed. Caller must obtain an authenticated fresh-process snapshot;
    this function does not establish its origin or terminate surviving grants.
    """
    if not isinstance(allowlist, Mapping) or not REAL_ADAPTERS <= allowlist.keys():
        refuse("GRANT_CLOSE_INVALID", "complete parsed allowlist required")
    for name, entry in allowlist.items():
        if not isinstance(entry, Mapping) or entry.get("name") != name:
            refuse("GRANT_CLOSE_INVALID", "malformed adapter entry")
        status = entry.get("status")
        if type(status) is not str or status not in {"granted", "blocked_pending_operator", "not_granted"}:
            refuse("GRANT_CLOSE_INVALID", "unknown grant status")
        if name not in {"synthetic", "synthetic_commercial_clean"} or entry.get("source_class") != "synthetic":
            if status == "granted":
                refuse("REAL_GRANT_STILL_OPEN", "real-data declaration remains granted")
            if status == "blocked_pending_operator":
                if (not isinstance(entry.get("allowed_uses"), (list, tuple)) or entry["allowed_uses"]
                        or not isinstance(entry.get("scopes"), (list, tuple)) or entry["scopes"]):
                    refuse("GRANT_CLOSE_UNRESTRICTED", "blocked entries require empty allowed_uses and scopes")
    return True
