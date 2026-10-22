"""Append-only synthetic exposure reservations, serialized across processes.

No constructor IO. The explicit existing directory is the shared namespace;
protocol hashes NEVER select different ledgers. Persistent lock file is not
deleted: byte locks are released by the OS on owner death. See README.md for
custody, recovery, external checkpoint and OS durability limitations.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import uuid
from .common import actors, stage as valid_stage, refuse

_ZERO = "0" * 64
MAX_RESERVE_ACTORS = 512
MAX_HISTORY_BYTES = 10 * 1024 * 1024


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _unique(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError("duplicate JSON key")
        result[k] = v
    return result


@dataclass(frozen=True)
class LedgerCheckpoint:
    sequence: int
    digest: str


def require_external_checkpoint(checkpoint):
    """Pure presence/shape check, not proof of freshness or external custody."""
    if (type(checkpoint) is not LedgerCheckpoint or type(checkpoint.sequence) is not int
            or checkpoint.sequence < 0 or not _sha(checkpoint.digest)):
        refuse("LEDGER_CHECKPOINT_REQUIRED", "valid external high-water checkpoint required")
    return checkpoint


@dataclass(frozen=True)
class ExposureOverride:
    """Synthetic exception record, not real operator approval or authentication."""
    record_id: str
    actor_ids: tuple
    prior_reservations: tuple
    reason: str


class ExposureLedger:
    def __init__(self, directory, namespace, *, checkpoint=None, lock_timeout=3.0, require_checkpoint=False):
        try:
            value = os.fspath(directory)
            if not isinstance(value, str):
                raise TypeError("string path required")
            self.directory = Path(value)
        except (TypeError, ValueError):
            refuse("LEDGER_NAMESPACE_INVALID", "string or string-valued PathLike directory required")
        if not self.directory.is_absolute() or ".." in self.directory.parts:
            refuse("LEDGER_NAMESPACE_INVALID", "explicit absolute directory required")
        if not isinstance(namespace, str) or not re.fullmatch(r"synthetic:[a-z0-9_-]{1,64}", namespace):
            refuse("LEDGER_NAMESPACE_INVALID", "synthetic cohort namespace required")
        if isinstance(lock_timeout, bool) or not isinstance(lock_timeout, (int, float)) or not math.isfinite(lock_timeout) or lock_timeout <= 0:
            refuse("LEDGER_LOCK_INVALID", "finite positive lock timeout required")
        if checkpoint is not None and (type(checkpoint) is not LedgerCheckpoint or type(checkpoint.sequence) is not int
                                      or checkpoint.sequence < 0 or not _sha(checkpoint.digest)):
            refuse("LEDGER_CHECKPOINT_INVALID", "invalid external checkpoint")
        self.namespace, self.checkpoint, self.lock_timeout = namespace, checkpoint, lock_timeout
        if type(require_checkpoint) is not bool:
            refuse("LEDGER_CHECKPOINT_INVALID", "require_checkpoint must be boolean")
        self.require_checkpoint = require_checkpoint
        self.path = self.directory / "synthetic-exposures.jsonl"
        self.lock_path = self.directory / "synthetic-exposures.lock"

    @contextmanager
    def _locked(self):
        try:
            fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        except OSError as exc:
            refuse("LEDGER_UNAVAILABLE", type(exc).__name__)
        owned = False
        try:
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
                os.fsync(fd)
            if os.name == "nt":
                import msvcrt
                def acquire():
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                def release():
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                def acquire():
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                def release():
                    fcntl.flock(fd, fcntl.LOCK_UN)
            deadline = time.monotonic() + self.lock_timeout
            while True:
                try:
                    acquire()
                    owned = True
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        refuse("LEDGER_LOCK_BUSY", "exclusive ledger lock unavailable")
                    time.sleep(.01)
            yield
        finally:
            try:
                if owned:
                    release()
            finally:
                os.close(fd)

    def _override(self, override, overlap, prior_ids, used_ids):
        if not overlap:
            if override is not None:
                refuse("OVERRIDE_INVALID", "override supplied without an intersection")
            return None
        if type(override) is not ExposureOverride:
            refuse("EXPOSURE_ALREADY_CONSUMED", "actor exposure intersects prior reservations")
        if (type(override.actor_ids) is not tuple or type(override.prior_reservations) is not tuple
                or not isinstance(override.record_id, str) or not re.fullmatch(r"SYNTHETIC-override-[a-z0-9-]{1,64}", override.record_id)
                or override.record_id in used_ids or tuple(override.actor_ids) != tuple(sorted(overlap))
                or tuple(override.prior_reservations) != tuple(sorted(prior_ids))
                or not isinstance(override.reason, str) or not 1 <= len(override.reason.strip()) <= 300):
            refuse("OVERRIDE_INVALID", "exact actors, exposures and unused override id required")
        return {"record_id": override.record_id, "actor_ids": list(override.actor_ids),
                "prior_reservations": list(override.prior_reservations), "reason": override.reason}

    def _history(self):
        if self.require_checkpoint:
            require_external_checkpoint(self.checkpoint)
        try:
            data = self.path.read_bytes()
        except OSError as exc:
            refuse("LEDGER_UNAVAILABLE", type(exc).__name__)
        if not data or not data.endswith(b"\n") or len(data) > MAX_HISTORY_BYTES:
            refuse("LEDGER_HISTORY_INVALID", "empty, partial or oversized history")
        try:
            lines = data.split(b"\n")[:-1]
            records = [json.loads(line.decode("ascii"), object_pairs_hook=_unique) for line in lines]
            for line, row in zip(lines, records):
                if line != json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii"):
                    refuse("LEDGER_HISTORY_INVALID", "noncanonical record bytes")
        except (ValueError, UnicodeError):
            refuse("LEDGER_HISTORY_INVALID", "unparseable history")
        prev, exposed, ids, used_overrides = _ZERO, {}, set(), set()
        for seq, row in enumerate(records):
            if not isinstance(row, dict):
                refuse("LEDGER_HISTORY_INVALID", "record object required")
            body = {k: v for k, v in row.items() if k != "digest"}
            common = {"kind", "namespace", "seq", "prev", "synthetic_only", "digest"}
            keys = common if seq == 0 else common | {"protocol_sha256", "stage", "actors", "reservation_id", "override"}
            if (set(row) != keys or type(row.get("seq")) is not int or row["seq"] != seq
                    or row["prev"] != prev or row["namespace"] != self.namespace
                    or row["synthetic_only"] is not True or row["digest"] != _digest(body)
                    or row["kind"] != ("genesis" if seq == 0 else "reservation")):
                refuse("LEDGER_HISTORY_INVALID", "chain, schema or namespace mismatch")
            if seq:
                actor_set = actors(row["actors"])
                if list(actor_set) != row["actors"] or not _sha(row["protocol_sha256"]):
                    refuse("LEDGER_HISTORY_INVALID", "invalid canonical reservation")
                valid_stage(row["stage"])
                rid = row["reservation_id"]
                if not isinstance(rid, str) or not re.fullmatch(r"[a-f0-9]{32}", rid) or rid in ids:
                    refuse("LEDGER_HISTORY_INVALID", "invalid reservation id")
                overlap = set(actor_set) & exposed.keys()
                prior_ids = {rid for a in overlap for rid in exposed[a]}
                raw = row["override"]
                override = None
                if raw is not None:
                    if (not isinstance(raw, dict) or set(raw) != {"record_id", "actor_ids", "prior_reservations", "reason"}
                            or not isinstance(raw["actor_ids"], list) or not isinstance(raw["prior_reservations"], list)):
                        refuse("LEDGER_HISTORY_INVALID", "invalid override record")
                    override = ExposureOverride(raw["record_id"], tuple(raw["actor_ids"]), tuple(raw["prior_reservations"]), raw["reason"])
                self._override(override, overlap, prior_ids, used_overrides)
                if override:
                    used_overrides.add(override.record_id)
                ids.add(rid)
                for a in actor_set:
                    exposed.setdefault(a, set()).add(rid)
            prev = row["digest"]
        if self.checkpoint is not None:
            cp = self.checkpoint
            if cp.sequence >= len(records) or records[cp.sequence]["digest"] != cp.digest:
                refuse("LEDGER_ROLLBACK", "external high-water checkpoint missing or changed")
        return records, exposed, used_overrides

    def _append(self, body, *, genesis=False):
        row = {**body, "digest": _digest(body)}
        encoded = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
        flags = os.O_WRONLY | getattr(os, "O_BINARY", 0) | (os.O_CREAT | os.O_EXCL if genesis else os.O_APPEND)
        try:
            if not genesis and self.path.stat().st_size + len(encoded) > MAX_HISTORY_BYTES:
                refuse("LEDGER_CAPACITY", "append exceeds history bound; no bytes written")
            fd = os.open(self.path, flags, 0o600)
            try:
                remaining = memoryview(encoded)
                while remaining:
                    n = os.write(fd, remaining)
                    if n <= 0:
                        raise OSError("short ledger write")
                    remaining = remaining[n:]
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError as exc:
            refuse("LEDGER_COMMIT_FAILED", type(exc).__name__)
        records, _, _ = self._history()
        if records[-1] != row:
            refuse("LEDGER_COMMIT_FAILED", "durable readback mismatch")
        return row

    def initialize(self, *, explicit_genesis=False):
        """Exclusive genesis only. Existing/partial history is NEVER overwritten."""
        if explicit_genesis is not True:
            refuse("LEDGER_GENESIS_REQUIRED", "separate explicit synthetic genesis required")
        if self.require_checkpoint:
            require_external_checkpoint(self.checkpoint)
        with self._locked():
            row = self._append({"kind": "genesis", "namespace": self.namespace, "seq": 0,
                                "prev": _ZERO, "synthetic_only": True}, genesis=True)
            return LedgerCheckpoint(row["seq"], row["digest"])

    def history(self):
        with self._locked():
            records, _, _ = self._history()
            return tuple(records)

    def reserve(self, actor_ids, *, stage, protocol_sha256, override=None):
        """All stages conservatively consume actors forever, even failed/crashed runs.

        Overrides must exactly identify the intersection and ALL prior exposures;
        no protocol/role change, close or decode failure restores unused status.
        """
        if isinstance(actor_ids, (str, bytes)):
            refuse("ACTOR_INVALID", "actor collection required")
        bounded = []
        try:
            for value in actor_ids:
                if len(bounded) >= MAX_RESERVE_ACTORS:
                    refuse("ACTOR_LIMIT", "at most 512 actor inputs per reservation")
                bounded.append(value)
        except TypeError:
            refuse("ACTOR_INVALID", "actor collection required")
        requested, stage = actors(bounded), valid_stage(stage)
        if not _sha(protocol_sha256):
            refuse("PROTOCOL_INVALID", "full lowercase SHA256 required")
        with self._locked():
            records, exposed, used = self._history()
            overlap = set(requested) & exposed.keys()
            prior = {rid for a in overlap for rid in exposed[a]}
            override_record = self._override(override, overlap, prior, used)
            row = self._append({"kind": "reservation", "namespace": self.namespace, "seq": len(records),
                                "prev": records[-1]["digest"], "synthetic_only": True,
                                "stage": stage, "protocol_sha256": protocol_sha256,
                                "actors": list(requested), "reservation_id": uuid.uuid4().hex,
                                "override": override_record})
            return row
