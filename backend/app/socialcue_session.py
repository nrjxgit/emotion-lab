"""Per-session, in-memory state for the social-cue research prototype (EL-006 S4).

Used only when /api/infer is called with ``socialcue=1``. A client-chosen
ephemeral session id (header ``X-Socialcue-Session``) keys a bounded ring of
FrameRecord and optional 33-landmark pose frames built from the existing per-frame outputs. Nothing is written to
disk, no image or frame is kept, and idle sessions expire after 60 s (lazily; see the limits below).

Without a valid session id nothing is stored and the block reports
insufficient evidence (reason ``no_session_id`` / ``invalid_session_id``).
Any failure in the cue path yields insufficient=true, reason ``internal_error``;
it never raises into the analyse route.

Scope and limits (single-user local prototype, NOT a shared service):
- Session ids are bearer secrets. Whoever knows an id can post frames into that
  session, read its state through /api/infer and clear it with the unauthenticated
  DELETE route. Clients must generate a high-entropy random id (for example
  crypto.randomUUID(), 122 bits); ids shorter than 22 characters are rejected.
- The 8-session cap is global (least recently used is evicted). There are no
  per-client or per-origin caps, so any caller can evict other sessions. CORS in
  main.py is open, so this is acceptable only on a single-user local machine.
- Expiry is LAZY with purge-on-access: a session idle for more than the TTL is
  logically gone and is physically dropped the next time the store is touched
  (add, clear, frame_count, len or purge). There is no background timer, so idle
  data is not wiped from process memory until then; nothing is ever written to disk.
- The repo root is appended to sys.path on first import of this module so that
  ``socialcue`` resolves; a backend-only copy without it returns ``internal_error``.
"""

import re
import math
import sys
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:  # socialcue/ lives at the repo root; append so nothing is shadowed
    sys.path.append(_ROOT)

HEADER = "X-Socialcue-Session"
TTL_SECONDS = 60.0
MAX_SESSIONS = 8
MAX_FRAMES = 64
MAX_GENERATIONS = 128  # bounded admission/tombstone map; evicted tokens fail closed
MIN_ID_LENGTH = 22  # >= 128 bits of base64url or 122-bit UUID text; ids are bearer secrets
SESSION_ID_RE = re.compile(r"[A-Za-z0-9_-]{22,64}")

BANNER = (
    "Social cues (research prototype): signs consistent with observable behaviour, "
    "not a reading of feelings. Hypothesis heuristic (HB-1); not validated. "
    "Single consenting person only; only the last seconds are held in memory."
)
HEURISTIC = "HB-1 hypothesis, unvalidated"  # mirrors socialcue.state.HEURISTIC (asserted in tests)

_CLASS_TO_FER7 = {
    "happiness": "happy",
    "sadness": "sad",
    "anger": "angry",
    "surprise": "surprise",
    "fear": "fear",
    "disgust": "disgust",
    "neutral": "neutral",
}


def insufficient_block(*reasons, usable_frames=0, fps=0.0, window_seconds=3.0):
    """Shape-compatible block that carries no state, used for refusals and errors."""
    return {
        "states": [],
        "insufficient": {"flag": True, "reasons": list(reasons)},
        "window": {"seconds": window_seconds, "usable_frames": usable_frames, "fps": fps},
        "heuristic": HEURISTIC,
        "proxy_eval_ref": None,
        "banner": BANNER,
    }


def _pace_frames(records, frame_type, usable, origin):
    """One representative per fixed-origin 125 ms bin: multi-face evidence, else a usable record, else any record.

    Multi-face ties keep the latest instant, so the rolling left edge cannot
    discard a later same-bin multi-face tick prematurely. Other ties keep the
    earliest record. The fixed origin is shared with HB-1 coverage; a usable
    record still beats an unusable tick. One representative cannot preserve every
    full-stream feature or onset, and multi-face replacement can shorten dwell.
    """
    if not records or not all(isinstance(r, frame_type) and isinstance(r.t, (int, float)) and math.isfinite(r.t) for r in records):
        return records
    bins = {}
    def rank(r):
        return 2 if isinstance(r.face_count, (int, float)) and r.face_count > 1 else 1 if usable(r) else 0
    for r in records:
        key = math.floor((r.t - origin) * 8.0 + 1e-9)
        prior = bins.get(key)
        if (prior is None or rank(r) > rank(prior)
                or (rank(r) == rank(prior) and (r.t > prior.t if rank(r) == 2 else r.t < prior.t))):
            bins[key] = r
    return sorted(bins.values(), key=lambda r: r.t)


class SessionInvalidated(RuntimeError):
    pass


class NonFiniteClock(ValueError):
    """The store clock (or a record's instant) was not a finite real number; the call changed nothing."""


def _finite_instant(x) -> bool:
    try:
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    except (TypeError, ValueError, OverflowError):
        return False


class SocialCueSessions:
    """Bounded in-memory store: TTL per session, cap on sessions and on frames per session."""

    def __init__(self, clock=time.monotonic, ttl=TTL_SECONDS, max_sessions=MAX_SESSIONS, max_frames=MAX_FRAMES):
        self.clock = clock
        self.ttl = ttl
        self.max_sessions = max_sessions
        self.max_frames = max_frames
        self._sessions = OrderedDict()  # id -> {"last": float, "frames": deque}; oldest first
        self._lock = threading.Lock()
        self._generations = OrderedDict()
        self._last_deletes = {}  # same bounded key set as _generations
        self._next_generation = 0
        self._evicted_generation = 0  # watermark: forgotten tokens fail closed

    def _now(self):
        """Validate each clock read before any purge, branch or retained mutation."""
        now = self.clock()
        if not _finite_instant(now):
            raise NonFiniteClock("non-finite clock value")
        return now

    def _purge(self, now):
        for sid in [s for s, v in self._sessions.items() if now - v["last"] > self.ttl]:
            del self._sessions[sid]

    def purge(self):
        """Drop every expired session now (also done on every access)."""
        with self._lock:
            self._purge(self._now())

    def _advance(self, session_id, *, deleted=False):
        self._next_generation += 1
        self._generations.pop(session_id, None)
        self._generations[session_id] = self._next_generation
        self._last_deletes[session_id] = self._next_generation if deleted else self._last_deletes.get(session_id, 0)
        while len(self._generations) > MAX_GENERATIONS:
            evicted_id, evicted = self._generations.popitem(last=False)
            self._last_deletes.pop(evicted_id, None)
            self._evicted_generation = max(self._evicted_generation, evicted)
        return self._next_generation

    def capture_generation(self, session_id):
        """Read-only pre-upload token; no registration, LRU refresh or retained payload."""
        if not isinstance(session_id, str) or not SESSION_ID_RE.fullmatch(session_id):
            return None
        with self._lock:
            return self._next_generation, self._generations.get(session_id)

    def admit(self, session_id, *, captured=None):
        """Register only after a CPU slot; validate the read-only pre-upload token."""
        if not isinstance(session_id, str) or not SESSION_ID_RE.fullmatch(session_id):
            return None
        with self._lock:
            generation = self._generations.get(session_id)
            if captured is not None:
                captured_at, old_generation = captured
                # A different id's DELETE does not invalidate this upload. If this
                # id changed, or bounded eviction forgot history since capture,
                # refuse rather than risk resurrecting a deleted session.
                if ((old_generation is None and (
                        self._last_deletes.get(session_id, 0) > captured_at
                        or self._evicted_generation > captured_at))
                        or (old_generation is not None and generation != old_generation)):
                    raise SessionInvalidated("session changed during upload")
            if generation is None:
                generation = self._advance(session_id)
            else:
                self._generations.move_to_end(session_id)
            return generation

    def generation_current(self, session_id, generation):
        with self._lock:
            return generation is not None and self._generations.get(session_id) == generation

    def _entry(self, session_id, now):
        if not _finite_instant(now):  # checked before anything is popped, created or evicted
            raise NonFiniteClock("non-finite clock value")
        entry = self._sessions.pop(session_id, None)
        if entry is None:
            while len(self._sessions) >= self.max_sessions:
                self._sessions.popitem(last=False)
            entry = {"frames": deque(maxlen=self.max_frames), "poses": deque(maxlen=self.max_frames),
                     "movement": deque(maxlen=30), "movement_identity": None, "bin_origin": None}
        entry["last"] = now
        self._sessions[session_id] = entry
        return entry

    def movement(self, session_id, generation, points, identity=None):
        """Legacy numeric movement shares session TTL/LRU/DELETE, never anonymous history."""
        now = self._now()  # also validate anonymous and empty-points calls
        if generation is None:
            return [points] if points else []
        from socialcue.body import BODY_CONFIG
        cfg = BODY_CONFIG["scale"]
        with self._lock:
            self._purge(now)
            if self._generations.get(session_id) != generation:
                raise SessionInvalidated("session generation changed")
            if not points:
                entry = self._sessions.get(session_id)
                if entry is not None:
                    entry["movement"].clear()
                    entry["poses"].clear()
                    entry["movement_identity"] = None
                return []
            entry = self._entry(session_id, now)
            previous = entry["movement_identity"]
            if identity is not None and previous is not None:
                size, scale, x, y = identity
                old_size, old_scale, old_x, old_y = previous
                if (size != old_size or not cfg["disc_s_ratio_lo"] <= scale / old_scale <= cfg["disc_s_ratio_hi"]
                        or math.hypot(x - old_x, y - old_y) / old_scale > cfg["disc_o_jump_sw"]):
                    entry["movement"].clear()
                    entry["poses"].clear()
            entry["movement_identity"] = identity
            entry["movement"].append(points)
            return list(entry["movement"])

    def add(self, session_id, make_record, make_pose=None, *, generation=None):
        """Build bounded internal snapshots OUTSIDE the mutation lock, then commit atomically.

        A builder may inspect the store without deadlock. No user callback is accepted
        by HTTP. Optional generation is required by the HTTP path; direct internal
        tests may omit it. Failure leaves retained records unchanged.
        """
        from socialcue.state import FrameRecord, PacedRecords, _usable
        now = self._now()
        if not _finite_instant(now):
            raise NonFiniteClock("non-finite clock value")
        record = make_record(now)
        if isinstance(record, FrameRecord) and not _finite_instant(record.t):
            raise NonFiniteClock("non-finite record instant")
        pose_record = make_pose(now, record) if make_pose is not None else None
        with self._lock:
            if generation is not None and self._generations.get(session_id) != generation:
                raise SessionInvalidated("session generation changed")
            committed_at = self._now()
            if not _finite_instant(committed_at):
                raise NonFiniteClock("non-finite clock value")
            self._purge(committed_at)
            entry = self._entry(session_id, committed_at)
            frames = entry["frames"]
            if entry["bin_origin"] is None and isinstance(record, FrameRecord):
                entry["bin_origin"] = record.t
            paced = _pace_frames([*frames, record], FrameRecord, _usable, entry["bin_origin"])
            frames.clear()
            # Limit both occupied-bin count and elapsed history; low-rate gaps
            # cannot retain an arbitrarily old presence record.
            if paced and isinstance(paced[-1], FrameRecord):
                newest = max(record.t, paced[-1].t)
                paced = [r for r in paced if newest - self.max_frames / 8.0 < r.t <= newest]
            frames.extend(paced)
            if make_pose is not None:
                poses = entry["poses"]
                if (not pose_record["lm"] or record.face_count != 1
                        or (poses and poses[-1]["size"] != pose_record["size"])):
                    poses.clear()
                if pose_record["lm"] and record.face_count == 1:
                    poses.append(pose_record)
                snapshot = PacedRecords(entry["frames"], entry["bin_origin"])
                snapshot.current = record
                return snapshot, list(poses)
            snapshot = PacedRecords(entry["frames"], entry["bin_origin"])
            snapshot.current = record
            return snapshot

    def clear(self, session_id):
        with self._lock:
            self._purge(self._now())
            if not isinstance(session_id, str) or not SESSION_ID_RE.fullmatch(session_id):
                return False
            self._advance(session_id, deleted=True)  # also invalidate pending uploads when no entry exists
            return self._sessions.pop(session_id, None) is not None

    def clear_all(self):
        with self._lock:
            self._sessions.clear()
            self._generations.clear()
            self._last_deletes.clear()
            self._next_generation += 1
            self._evicted_generation = self._next_generation

    def __len__(self):
        with self._lock:
            self._purge(self._now())
            return len(self._sessions)

    def frame_count(self, session_id):
        with self._lock:
            self._purge(self._now())
            e = self._sessions.get(session_id)
            return len(e["frames"]) if e else 0


STORE = SocialCueSessions()


def _fer7(probs):
    p = {k7: float((probs or {}).get(c, 0.0) or 0.0) for c, k7 in _CLASS_TO_FER7.items()}
    total = sum(p.values())
    return {k: v / total for k, v in p.items()} if total > 0 else None


def _make_record_fn(faces, face, face_out, fq):
    """Build FrameRecord from the existing per-frame outputs; only the largest face is ever used."""
    from socialcue.state import FrameRecord

    def make(now):
        if face is None or not face_out or not face_out.get("face_found"):
            return FrameRecord(now, None, 0.0, 0.0, 0.0, None, None, 0)
        expr = face_out["expression"]
        lm = face.get("landmarks5")
        landmarks = None
        if lm:
            landmarks = tuple((float(lm[k][0]), float(lm[k][1])) for k in ("eye_R", "eye_L", "nose", "mouth_R", "mouth_L"))
        return FrameRecord(
            now,
            _fer7(expr.get("probs")),
            float(fq),
            float(expr.get("arousal") or 0.0),
            float(expr.get("valence") or 0.0),
            (float(face["x"]), float(face["y"]), float(face["w"]), float(face["h"])),
            landmarks,
            len(faces),
        )

    return make



def _pose_record_fn(pose, width, height):
    """Snapshot only a bounded numeric 33x3 payload; never retain the image/model dict."""
    from body.pose_payload import validate_pose
    payload = validate_pose(pose)
    landmarks = payload.keypoints
    reason = payload.reason

    def make(now, record):
        frame = {"t": now, "lm": landmarks, "size": (width, height),
                 "face_count": record.face_count, "face_box": record.box}
        if reason is not None:
            frame["reason"] = reason
        return frame
    return make


def _with_body(result, poses, current, consent):
    from socialcue import body

    if not consent:
        extracted = {"status": "none", "reason": "consent_off", "cues": []}
    elif not current["lm"]:
        extracted = {"status": "unavailable", "reason": current["reason"], "cues": []}
    else:
        extracted = body.extract_body_cues(
            [p["t"] for p in poses], [p["lm"] for p in poses], *current["size"],
            consent=True, face_count=current["face_count"], face_box=current["face_box"])
    attached = body.attach_body(result, extracted)
    if extracted.get("status") != "ok":
        # Unavailable/refused evidence must not add body_support or round HB-1 scores.
        return dict(result, body=attached["body"])
    return attached


def socialcue_block(session_id, faces, face, face_out, fq, store=None, *, pose=None, width=None, height=None, generation=None):
    """Return the additive ``socialcue`` block. Never raises."""
    try:
        from socialcue.state import analyze

        store = STORE if store is None else store
        make = _make_record_fn(faces, face, face_out, fq)
        make_pose = _pose_record_fn(pose, width, height) if pose is not None else None
        poses, current = [], None
        if session_id is None or not SESSION_ID_RE.fullmatch(str(session_id)):
            reason = "no_session_id" if session_id is None else "invalid_session_id"
            result = analyze([make(store._now())])  # one-shot, nothing is stored
            result["insufficient"] = {"flag": True, "reasons": result["insufficient"]["reasons"] + [reason]}
            result["states"] = []
        else:
            if make_pose is None:
                result = analyze(store.add(session_id, make, generation=generation))
            else:
                frames, poses = store.add(session_id, make, make_pose, generation=generation)
                result = analyze(frames)
                # pose metadata comes from the record this request just added, not the (possibly older) retained bin member
                added = frames.current if frames.current is not None else frames[-1]
                current = make_pose(added.t, added)
        if make_pose is not None:
            valid_session = session_id is not None and SESSION_ID_RE.fullmatch(str(session_id)) is not None
            try:
                result = _with_body(result, poses, current, valid_session)
            except Exception:
                # Body processing is optional: preserve the already computed HB-1 result.
                result = dict(result, body={"status": "unavailable", "reason": "internal_error", "cues": []})
        result["banner"] = BANNER
        return result
    except SessionInvalidated:
        return insufficient_block("session_invalidated")
    except Exception:
        return insufficient_block("internal_error")
