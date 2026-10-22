"""Window-level observable-behaviour annotation records (EL-006 S15; pilot protocol section 4).

A record says what a camera shows or what the interaction log says in one fixed 3 s window; it never labels
feelings. The validator is a closed schema: unknown keys are rejected, there are NO free-text fields, tokens are
opaque random codes (``ANN-`` / ``PT-`` plus hex), and keys that look like names, faces, contact details or notes
are rejected by name. Pure stdlib; no IO.

Block starts are NOT forced onto a 3 s grid (documentation only, no enforcement): the protocol expects non-overlapping
3 s blocks, normally starting at multiples of 3 s from the capture start, but a block may start at any non-negative
time and ``window_scoring.align`` copes with offsets. Only overlap and duplication are rejected.

Synthetic participants use the reserved code form ``PT-5eed`` + 4 hex digits; real pilot codes must never use it.

Opaque tokens only prove the *shape* of an identifier; they cannot prove that a code was not derived from a person's
name. The operator assigns random codes and keeps the key (if any) outside this repository.
"""
from __future__ import annotations

import math
import re

SCHEMA_VERSION = 1
WINDOW_SECONDS = 3.0
LABELS = ("look_away", "pause", "asks_help", "fidget")  # observable behaviour; no_cue is derived
FLAGS = ("unscorable", "unsure_rate", "confound_flag")
PRESENT, ABSENT = "present", "absent"
ANNOTATOR_RE = re.compile(r"ANN-[0-9a-f]{6,16}")
PARTICIPANT_RE = re.compile(r"PT-[0-9a-f]{6,16}")
SYNTHETIC_PARTICIPANT_RE = re.compile(r"PT-5eed[0-9a-f]{4}")  # reserved for manufactured participants
ALLOWED_KEYS = frozenset({"schema_version", "annotator", "participant", "window_start", "window_end", "labels", *FLAGS})
PII_KEYS = frozenset({"name", "first_name", "last_name", "full_name", "email", "phone", "address", "dob", "birth", "age", "face",
                      "image", "photo", "video", "frame", "landmarks", "embedding", "ip", "device", "location", "gender", "ethnicity",
                      "note", "notes", "comment", "comments", "text", "free_text", "description", "feeling", "emotion", "mood",
                      "nervous", "confused", "intent", "personality"})


class AnnotationError(ValueError):
    """Rejected record. ``codes`` lists every reason (stable strings)."""

    def __init__(self, codes):
        super().__init__("; ".join(codes))
        self.codes = list(codes)


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_annotation(rec) -> dict:
    """Return a normalised copy of ``rec`` or raise AnnotationError listing all problems."""
    errs = []
    if not isinstance(rec, dict):
        raise AnnotationError(["not_a_dict"])
    keys = set(rec)
    for k in sorted(keys - ALLOWED_KEYS, key=str):
        lowered = str(k).lower()
        if lowered in PII_KEYS or any(p in lowered for p in ("name", "mail", "phone", "face", "photo", "note", "comment", "text")):
            errs.append(f"pii_or_free_text_field:{k}")
        else:
            errs.append(f"unknown_key:{k}")
    for k in sorted(ALLOWED_KEYS - keys):
        errs.append(f"missing_key:{k}")
    if errs:
        raise AnnotationError(errs)
    if type(rec["schema_version"]) is not int or rec["schema_version"] != SCHEMA_VERSION:
        errs.append("bad_schema_version")
    if not (isinstance(rec["annotator"], str) and ANNOTATOR_RE.fullmatch(rec["annotator"])):
        errs.append("bad_annotator_token")
    if not (isinstance(rec["participant"], str) and PARTICIPANT_RE.fullmatch(rec["participant"])):
        errs.append("bad_participant_token")
    s, e = rec["window_start"], rec["window_end"]
    if not (_num(s) and _num(e)):
        errs.append("bad_window_times")
    elif s < 0 or abs((e - s) - WINDOW_SECONDS) > 1e-6:
        errs.append("window_must_be_3s_and_nonnegative")
    labels = rec["labels"]
    if not isinstance(labels, dict) or set(labels) != set(LABELS):
        errs.append("labels_must_be_exactly_" + "_".join(LABELS))
    elif any(labels[l] not in (PRESENT, ABSENT) or not isinstance(labels[l], str) for l in LABELS):
        errs.append("label_values_must_be_present_or_absent")
    for f in FLAGS:
        if not isinstance(rec[f], bool):
            errs.append(f"flag_must_be_bool:{f}")
    if errs:
        raise AnnotationError(errs)
    return {"schema_version": SCHEMA_VERSION, "annotator": rec["annotator"], "participant": rec["participant"],
            "window_start": float(s), "window_end": float(e), "labels": dict(labels), **{f: rec[f] for f in FLAGS}}


def is_synthetic_participant(token) -> bool:
    return isinstance(token, str) and SYNTHETIC_PARTICIPANT_RE.fullmatch(token) is not None


def derived_no_cue(rec: dict) -> bool:
    """no_cue is derived: scorable and every behaviour label absent."""
    return (not rec["unscorable"]) and all(rec["labels"][l] == ABSENT for l in LABELS)


def validate_batch(records) -> list:
    """Validate every record, then reject duplicates and overlapping windows per (annotator, participant)."""
    if isinstance(records, (str, bytes, dict)):
        raise AnnotationError(["not_a_list"])
    try:
        records = list(records)
    except TypeError:
        raise AnnotationError(["not_a_list"]) from None
    out = [validate_annotation(r) for r in records]
    by_key = {}
    for r in out:
        by_key.setdefault((r["annotator"], r["participant"]), []).append(r)
    errs = []
    for key, rs in by_key.items():
        rs.sort(key=lambda r: r["window_start"])
        for a, b in zip(rs, rs[1:]):
            if b["window_start"] < a["window_end"] - 1e-9:
                errs.append(f"overlapping_or_duplicate_windows:{key[0]}/{key[1]}@{b['window_start']}")
    if errs:
        raise AnnotationError(errs)
    return out
