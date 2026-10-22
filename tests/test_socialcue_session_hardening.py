"""Session-store and default-path hardening from the independent S4 review (EL-006 S24b). Fakes only."""
import pathlib
import sys
import threading

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "app"))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import main  # noqa: E402
import socialcue_session as scs  # noqa: E402
from socialcue.state import FrameRecord  # noqa: E402

SID = "session-hardening-0001-aaaa"
BLOCK_KEYS = {"states", "insufficient", "window", "heuristic", "proxy_eval_ref", "banner"}


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def rec(now):
    return FrameRecord(now, None, 0.0, 0.0, 0.0, None, None, 0)


def boom(now):
    raise RuntimeError("record builder failed")


# --- G2: a failing add keeps the session --------------------------------------------------

def test_failing_add_keeps_the_existing_session():
    clock = Clock()
    store = scs.SocialCueSessions(clock=clock)
    store.add(SID, rec)
    clock.t += .125  # distinct retained bin; this test concerns failed-add atomicity
    store.add(SID, rec)
    with pytest.raises(RuntimeError):
        store.add(SID, boom)
    assert store.frame_count(SID) == 2
    clock.t += .125
    store.add(SID, rec)
    assert store.frame_count(SID) == 3


def test_failing_add_of_a_new_session_evicts_nobody_and_stores_nothing():
    store = scs.SocialCueSessions(clock=Clock())
    ids = [f"session-{i:04d}-0123456789ab" for i in range(scs.MAX_SESSIONS)]
    for sid in ids:
        store.add(sid, rec)
    with pytest.raises(RuntimeError):
        store.add("session-newcomer-0123456789", boom)
    assert len(store) == scs.MAX_SESSIONS and all(store.frame_count(s) == 1 for s in ids)
    assert store.frame_count("session-newcomer-0123456789") == 0


def test_a_failing_record_through_the_api_path_keeps_the_window(monkeypatch):
    store = scs.SocialCueSessions(clock=Clock())
    monkeypatch.setattr(scs, "STORE", store)
    store.add(SID, rec)
    good = scs._make_record_fn
    monkeypatch.setattr(scs, "_make_record_fn", lambda *a, **k: boom)
    blk = scs.socialcue_block(SID, [], None, None, 0.0)
    assert blk["insufficient"]["reasons"] == ["internal_error"]
    monkeypatch.setattr(scs, "_make_record_fn", good)
    assert store.frame_count(SID) == 1


# --- G3: lazy expiry with purge on every access ----------------------------------------------

def test_frame_count_purges_expired_sessions_without_len():
    clock = Clock()
    store = scs.SocialCueSessions(clock=clock)
    store.add(SID, rec)
    clock.t += scs.TTL_SECONDS + 1
    assert store.frame_count(SID) == 0          # no len() call needed first
    assert SID not in store._sessions            # and it is physically dropped


def test_clear_and_purge_drop_expired_sessions():
    clock = Clock()
    store = scs.SocialCueSessions(clock=clock)
    store.add(SID, rec)
    store.add("session-other-0001-abcdefgh", rec)
    clock.t += scs.TTL_SECONDS + 1
    assert store.clear(SID) is False             # already expired: nothing to clear, but everything purged
    assert store._sessions == {} or list(store._sessions) == []
    store.add(SID, rec)
    clock.t += 59
    store.purge()
    assert store.frame_count(SID) == 1
    clock.t += 2
    store.purge()
    assert len(store._sessions) == 0


def test_ttl_is_measured_from_last_use_not_creation():
    clock = Clock()
    store = scs.SocialCueSessions(clock=clock)
    for _ in range(5):
        store.add(SID, rec)
        clock.t += 30
    assert len(store) == 1  # touches prevent TTL expiry even beyond creation + TTL
    assert store.frame_count(SID) == 1  # only the latest 8-second history remains


def test_expiry_is_documented_as_lazy_not_a_physical_wipe():
    doc = " ".join(scs.__doc__.lower().split())
    assert "lazy" in doc and "purge-on-access" in doc and "no background timer" in doc
    assert "wiped" in doc and "idle sessions expire after 60 s (lazily" in doc


# --- G1/G6/DELETE: limits are documented --------------------------------------------------------

def test_security_limits_are_documented():
    doc = " ".join(scs.__doc__.lower().split())
    for phrase in ("bearer secrets", "single-user local prototype", "global", "no per-client", "unauthenticated delete",
                   "crypto.randomuuid", "sys.path"):
        assert phrase in doc, phrase
    assert "unauthenticated" in main.socialcue_session_clear.__doc__.lower()


def test_minimum_session_id_length_is_22():
    assert scs.MIN_ID_LENGTH == 22
    assert scs.SESSION_ID_RE.fullmatch("a" * 22) and scs.SESSION_ID_RE.fullmatch("a" * 64)
    assert not scs.SESSION_ID_RE.fullmatch("a" * 21) and not scs.SESSION_ID_RE.fullmatch("a" * 65)
    assert not scs.SESSION_ID_RE.fullmatch("a" * 21 + "\n") and not scs.SESSION_ID_RE.fullmatch("a" * 21 + "٣")


# --- concurrency -------------------------------------------------------------------------------------

def test_concurrent_adds_keep_the_caps_and_never_raise():
    store = scs.SocialCueSessions()
    errors = []

    def work(k):
        try:
            for i in range(400):
                store.add(f"session-thread{k % 12:02d}-0123456789", lambda now: rec(now))
                store.frame_count(f"session-thread{(k + i) % 12:02d}-0123456789")
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(k,)) for k in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
    assert len(store) <= scs.MAX_SESSIONS
    assert all(store.frame_count(s) <= scs.MAX_FRAMES for s in list(store._sessions))


# --- G4/G5: default path and fallback ---------------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "detect_faces", lambda img: [])
    monkeypatch.setattr(main, "estimate_pose", lambda img: {"visible": False})
    monkeypatch.setattr(main, "_track", [])
    monkeypatch.setattr(scs, "STORE", scs.SocialCueSessions(clock=Clock()))
    return TestClient(main.app)


def _post(client, query=""):
    ok, buf = cv2.imencode(".jpg", np.full((240, 320, 3), 128, dtype=np.uint8))
    return client.post("/api/infer" + query, files={"image": ("f.jpg", buf.tobytes(), "image/jpeg")},
                       headers={scs.HEADER: SID})


@pytest.mark.parametrize("query", ["?socialcue=abc", "?socialcue=", "?socialcue=0", "?socialcue=2", "?socialcue=1.0",
                                   "?socialcue=true", "?socialcue=01", "?socialcue=-1", "?socialcue=%00", "?socialcue=1e0",
                                   "?socialcue=a&socialcue=b"])
def test_malformed_or_non_one_flag_behaves_exactly_like_the_default(client, query):
    default = _post(client)
    r = _post(client, query)
    assert default.status_code == 200 and r.status_code == 200, r.text
    assert r.json() == default.json()
    assert "socialcue" not in r.json()
    assert len(scs.STORE) == 0


def test_only_the_exact_value_one_enables_the_block(client):
    r = _post(client, "?socialcue=1")
    assert r.status_code == 200 and set(r.json()["socialcue"]) == BLOCK_KEYS | {"body"}
    assert len(scs.STORE) == 1


def test_openapi_declares_the_flag_as_a_plain_string_not_an_int(client):
    params = client.get("/openapi.json").json()["paths"]["/api/infer"]["post"]["parameters"]
    flag = next(p for p in params if p["name"] == "socialcue")
    assert flag["schema"]["type"] == "string" and flag["schema"]["default"] == "0"


def test_fallback_banner_is_the_shared_banner(client, monkeypatch):
    monkeypatch.setattr(scs, "socialcue_block", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    blk = _post(client, "?socialcue=1").json()["socialcue"]
    assert set(blk) == BLOCK_KEYS and blk["banner"] == scs.BANNER and blk["insufficient"]["reasons"] == ["internal_error"]


def test_last_resort_block_when_even_the_session_module_fails(client, monkeypatch):
    def fail(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(scs, "socialcue_block", fail)
    monkeypatch.setattr(scs, "insufficient_block", fail)
    r = _post(client, "?socialcue=1")
    blk = r.json()["socialcue"]
    assert r.status_code == 200 and set(blk) == BLOCK_KEYS
    assert blk["insufficient"] == {"flag": True, "reasons": ["internal_error"]}
    assert blk["banner"] == scs.BANNER  # the shared banner, imported in a nested try (review R2)


def test_last_resort_block_keeps_a_hedged_banner_if_even_the_banner_import_fails(client, monkeypatch):
    def fail(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(scs, "socialcue_block", fail)
    monkeypatch.setattr(scs, "insufficient_block", fail)
    monkeypatch.delattr(scs, "BANNER")
    blk = _post(client, "?socialcue=1").json()["socialcue"]
    assert set(blk) == BLOCK_KEYS and "not a reading of feelings" in blk["banner"] and "HB-1" in blk["banner"]


# --- S58: a non-finite clock never drops retained history (review S57 N4) ----------------------------------------------------------

class _SeqClock:
    def __init__(self, *values):
        self.values, self.i = list(values), 0

    def __call__(self):
        v = self.values[min(self.i, len(self.values) - 1)]
        self.i += 1
        return v


def _history(store, n=5):
    for k in range(n):
        store.add(SID, lambda now, k=k: FrameRecord(100.0 + 0.125 * k, None, 0.0, 0.0, 0.0, None, None, 0))
    return list(store._sessions[SID]["frames"])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), None, "1", True, [1.0]])
def test_s58_non_finite_clock_refuses_the_add_and_keeps_history(bad):
    clock = Clock()
    store = scs.SocialCueSessions(clock=clock)
    before = _history(store)
    snapshot = (list(store._sessions[SID]["frames"]), store._sessions[SID]["last"], store._sessions[SID]["bin_origin"], dict(store._generations))
    clock.t = bad
    with pytest.raises(scs.NonFiniteClock):
        store.add(SID, rec)
    clock.t = 100.9
    assert (list(store._sessions[SID]["frames"]), store._sessions[SID]["last"], store._sessions[SID]["bin_origin"], dict(store._generations)) == snapshot
    assert len(before) == 5 and store.frame_count(SID) == 5  # still there, still counted


def test_s58_clock_that_turns_non_finite_between_build_and_commit_keeps_history():
    clock = Clock()
    store = scs.SocialCueSessions(clock=clock)
    _history(store)
    frames, last = list(store._sessions[SID]["frames"]), store._sessions[SID]["last"]
    store.clock = _SeqClock(100.9, float("nan"))  # fine when the record is built, NaN at commit
    with pytest.raises(scs.NonFiniteClock):
        store.add(SID, rec)
    store.clock = Clock()
    assert list(store._sessions[SID]["frames"]) == frames and store._sessions[SID]["last"] == last


def test_s58_non_finite_clock_does_not_create_or_evict_sessions():
    store = scs.SocialCueSessions(clock=lambda: float("nan"), max_sessions=2)
    other = ["session-%04d-0123456789ab" % i for i in range(2)]
    real = scs.SocialCueSessions(max_sessions=2)
    for sid in other:
        real.add(sid, rec)
    real.clock = lambda: float("nan")
    with pytest.raises(scs.NonFiniteClock):
        real.add("session-new0-0123456789ab", rec)
    assert sorted(real._sessions) == sorted(other) and "session-new0-0123456789ab" not in real._sessions
    with pytest.raises(scs.NonFiniteClock):
        store.add(SID, rec)
    assert len(store._sessions) == 0


def test_s58_record_with_a_non_finite_instant_is_refused_before_any_mutation():
    store = scs.SocialCueSessions(clock=Clock())
    frames = _history(store)
    for t in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(scs.NonFiniteClock):
            store.add(SID, lambda now, t=t: FrameRecord(t, None, 0.0, 0.0, 0.0, None, None, 0))
    assert list(store._sessions[SID]["frames"]) == frames


def test_s58_movement_path_also_refuses_a_non_finite_clock_before_mutation():
    clock = Clock()
    store = scs.SocialCueSessions(clock=clock)
    gen = store.admit(SID)
    pts = [(0.1 * i, 0.2) for i in range(7)]
    store.movement(SID, gen, pts, (640, 160.0, 320.0, 200.0))
    before = list(store._sessions[SID]["movement"])
    clock.t = float("nan")
    with pytest.raises(scs.NonFiniteClock):
        store.movement(SID, gen, pts, (640, 160.0, 320.0, 200.0))
    assert list(store._sessions[SID]["movement"]) == before


def test_s58_block_builder_reports_internal_error_and_keeps_history_on_a_bad_clock():
    clock = Clock()
    store = scs.SocialCueSessions(clock=clock)
    _history(store)
    frames = list(store._sessions[SID]["frames"])
    clock.t = float("nan")
    face = {"x": 10, "y": 10, "w": 100, "h": 100, "confidence": 0.9,
            "landmarks5": {n: (20.0 + i, 20.0) for i, n in enumerate(("eye_R", "eye_L", "nose", "mouth_R", "mouth_L"))}}
    out = scs.socialcue_block(SID, [face], face, {"face_found": True, "expression": {"probs": {"neutral": 1.0}}}, 0.9, store=store)
    assert out["insufficient"]["reasons"] == ["internal_error"] and out["states"] == []
    assert list(store._sessions[SID]["frames"]) == frames


# --- S60 (review S59 finding 5): `current` comes from the record just added -----------------------------------------------------------

def test_s60_f5_current_pose_metadata_comes_from_the_request_just_added(monkeypatch):
    from dataclasses import replace
    clock = Clock()
    store = scs.SocialCueSessions(clock=clock)
    calls = []
    monkeypatch.setattr(scs, "_with_body", lambda result, poses, current, valid: calls.append(current) or result)
    clock.t = 100.00
    first = FrameRecord(100.00, {"neutral": 1.0}, 0.9, 0.1, 0.0, (0.0, 0.0, 50.0, 50.0),
                        ((10.0, 20.0), (40.0, 20.0), (25.0, 30.0), (15.0, 40.0), (35.0, 40.0)), 1)
    second = replace(first, t=100.10, box=(300.0, 0.0, 50.0, 50.0))
    face = lambda rec_: {"x": int(rec_.box[0]), "y": 0, "w": 50, "h": 50, "confidence": 0.9, "landmarks5": {
        n: p for n, p in zip(("eye_R", "eye_L", "nose", "mouth_R", "mouth_L"), rec_.landmarks)}}
    pose = [[0.5, 0.5, 0.9]] * 33
    for rec_, now in ((first, 100.00), (second, 100.10)):
        clock.t = now
        out = scs.socialcue_block("session-s60-0001-abcdefgh", [face(rec_)], face(rec_), {"face_found": True, "expression": {"probs": {"neutral": 1.0}}}, 0.9,
                                  store=store, pose=[[0.5, 0.5, 0.9, 0.9]] * 33, width=640, height=480)
    assert len(store._sessions["session-s60-0001-abcdefgh"]["frames"]) == 1  # same 125 ms bin: the older record is the retained one
    assert len(calls) == 2 and calls[1]["face_box"] == (300.0, 0.0, 50.0, 50.0), calls[1]["face_box"]  # before: the first request's box
    assert calls[1]["t"] == pytest.approx(100.10)


def test_s60_f5_add_exposes_the_record_it_just_added():
    clock = Clock()
    store = scs.SocialCueSessions(clock=clock)
    a = FrameRecord(100.00, None, 0.0, 0.0, 0.0, None, None, 0)
    b = FrameRecord(100.05, None, 0.0, 0.0, 0.0, None, None, 0)
    store.add(SID, lambda now: a)
    out = store.add(SID, lambda now: b)
    assert out.current is b and len(out) == 1 and out[0] is a


# S65 / S64: same-endpoint gates and all clock-reading paths.
def test_s65_latest_same_bin_multiface_survives_the_rolling_left_edge():
    from dataclasses import replace
    from socialcue import synthetic as Y, state as S
    rows = Y.generate("nervous-like", fps=13, duration=3.2)
    rows = [r for r in rows if r.t <= 3.076923 and r.t != 3.0]
    rows += [replace(rows[0], t=.05, face_count=2), replace(rows[0], t=.10, face_count=2)]
    rows.sort(key=lambda r: r.t)
    store = scs.SocialCueSessions(clock=lambda: 0.)
    for row in rows:
        retained = store.add(SID, lambda now, row=row: row)
    endpoint = retained[-1].t
    assert endpoint == 3.076923 and endpoint - 3 > .05 and endpoint - 3 < .10
    full = S.analyze(rows, evaluation_time=endpoint)
    ring = S.analyze(retained, evaluation_time=endpoint)
    assert "multiple_faces" in full["insufficient"]["reasons"]
    assert "multiple_faces" in ring["insufficient"]["reasons"] and not ring["states"]
    assert len(retained) <= 64
    same_bin = [r for r in retained if 0 <= r.t < .125]
    assert len(same_bin) == 1 and same_bin[0].t == .10


_S65_BAD_CLOCKS = [float("inf"), -float("inf"), float("nan"), None, True, False, "not-a-clock"]


@pytest.mark.parametrize("bad", _S65_BAD_CLOCKS, ids=["inf", "negative-inf", "nan", "none", "true", "false", "text"])
@pytest.mark.parametrize("mode", ["points", "empty", "anonymous"])
def test_s65_movement_clock_is_validated_before_purge_or_any_branch(bad, mode):
    import copy
    clock = [100.]
    store = scs.SocialCueSessions(clock=lambda: clock[0])
    generation = store.admit(SID)
    store.movement(SID, generation, [(1, 2)])
    # Includes a second session: +inf must not purge even unrelated entries.
    store.movement("session-other-0123456789", store.admit("session-other-0123456789"), [(3, 4)])
    before = copy.deepcopy((store._sessions, store._generations, store._last_deletes))
    clock[0] = bad
    points = [] if mode == "empty" else [(5, 6)]
    with pytest.raises(scs.NonFiniteClock):
        store.movement(SID, None if mode == "anonymous" else generation, points)
    assert (store._sessions, store._generations, store._last_deletes) == before


@pytest.mark.parametrize("bad", _S65_BAD_CLOCKS, ids=["inf", "negative-inf", "nan", "none", "true", "false", "text"])
@pytest.mark.parametrize("operation", ["purge", "clear", "len", "frame_count"])
def test_s65_other_clock_readers_refuse_before_mutation(bad, operation):
    import copy
    clock = [100.]
    store = scs.SocialCueSessions(clock=lambda: clock[0])
    store.add(SID, rec)
    before = copy.deepcopy((store._sessions, store._generations, store._last_deletes))
    clock[0] = bad
    with pytest.raises(scs.NonFiniteClock):
        {"purge": store.purge, "clear": lambda: store.clear(SID), "len": lambda: len(store),
         "frame_count": lambda: store.frame_count(SID)}[operation]()
    assert (store._sessions, store._generations, store._last_deletes) == before
