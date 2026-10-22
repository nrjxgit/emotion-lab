"""Additive socialcue block behind ?socialcue=1 with per-session in-memory state (EL-006 S4).

In-process TestClient only; the detector, predictor and pose are fakes driven by
synthetic socialcue streams. No camera, media, weights, server or port is used.
"""
import builtins
import pathlib
import sys

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
from socialcue import state as hb1  # noqa: E402
from socialcue import synthetic as syn  # noqa: E402

SID_A = "session-aaaa-0001-abcdefgh"
SID_B = "session-bbbb-0002-abcdefgh"
HDR = scs.HEADER
BLOCK_KEYS = {"states", "insufficient", "window", "heuristic", "proxy_eval_ref", "banner"}
FER7_TO_CLASS = {"happy": "happiness", "sad": "sadness", "angry": "anger", "surprise": "surprise",
                 "fear": "fear", "disgust": "disgust", "neutral": "neutral"}


def _jpeg():
    ok, buf = cv2.imencode(".jpg", np.full((480, 640, 3), 128, dtype=np.uint8))
    assert ok
    return buf.tobytes()


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Feed:
    """Drives the fake detector/predictor: each post consumes the next synthetic record."""

    source = "fake"

    def __init__(self, records, extra_faces=0):
        self.records = list(records)
        self.i = -1
        self.extra_faces = extra_faces

    def advance(self):
        self.i += 1
        return self.records[self.i]

    def detect(self, img):
        r = self.records[self.i]
        if r.box is None:
            return []
        x, y, w, h = r.box
        names = ("eye_R", "eye_L", "nose", "mouth_R", "mouth_L")
        face = {"x": int(x), "y": int(y), "w": int(w), "h": int(h), "confidence": r.quality,
                "landmarks5": {n: p for n, p in zip(names, r.landmarks)}}
        faces = [face]
        for k in range(self.extra_faces):
            faces.append({"x": 10, "y": 10, "w": 40 + k, "h": 40 + k, "confidence": 0.9,
                          "landmarks5": {n: (20.0 + i, 20.0) for i, n in enumerate(names)}})
        return faces

    def predict(self, crop, aus, calibrated=False):
        r = self.records[self.i]
        probs = {FER7_TO_CLASS[k]: v for k, v in r.expression.items()}
        probs["unknown"] = 0.0
        return {"probs": probs, "valence": 0.0, "arousal": r.arousal, "expressiveness": 0.0,
                "confidence": max(probs.values()), "uncertainty": 1 - max(probs.values()),
                "source": "fake", "profile": "corrected", "calibrated": False}


@pytest.fixture
def env(monkeypatch):
    clock = FakeClock()
    store = scs.SocialCueSessions(clock=clock)
    monkeypatch.setattr(scs, "STORE", store)
    monkeypatch.setattr(main, "estimate_pose", lambda img: {"visible": False})
    monkeypatch.setattr(main, "_track", [])
    client = TestClient(main.app)
    jpg = _jpeg()

    class E:
        pass

    e = E()
    e.clock, e.store, e.client = clock, store, client

    def attach(feed):
        monkeypatch.setattr(main, "detect_faces", feed.detect)
        monkeypatch.setattr(main, "get_predictor", lambda profile: feed)
        e.feed = feed

    def post(sid=SID_A, flag=1, step=0.125, query=""):
        if e.feed.i + 1 < len(e.feed.records):
            e.feed.advance()
        clock.t += step
        headers = {HDR: sid} if sid is not None else {}
        q = f"?socialcue={flag}{query}" if flag is not None else ""
        return client.post("/api/infer" + q, files={"image": ("f.jpg", jpg, "image/jpeg")}, headers=headers)

    e.attach, e.post = attach, post
    return e


def _shape(o):
    if isinstance(o, dict):
        return {k: _shape(v) for k, v in o.items()}
    if isinstance(o, list):
        return [type(x).__name__ for x in o[:1]]
    return type(o).__name__


def _run(env, scenario, n, sid=SID_A, **kw):
    env.attach(Feed(syn.generate(scenario, **kw)))
    last = None
    for _ in range(n):
        last = env.post(sid)
        assert last.status_code == 200, last.text
    return last.json()["socialcue"]


# --- contract ----------------------------------------------------------------

def test_default_response_has_no_socialcue_key_and_the_same_shape(env):
    env.attach(Feed(syn.generate("calm")))
    default = env.post(flag=None).json()
    explicit_off = env.post(flag=0).json()
    on = env.post(flag=1).json()
    assert "socialcue" not in default and "socialcue" not in explicit_off
    assert set(default) == {"selected_model", "face", "body", "fusion", "ethics_notice"}
    assert set(default["face"]) == {"face_found", "box", "aus", "expression", "observable_analysis",
                                    "geometry", "quality", "insufficient_evidence", "abstained", "abstain_threshold"}
    on_without = {k: v for k, v in on.items() if k != "socialcue"}
    assert _shape(on_without) == _shape(default)
    assert set(on) - set(default) == {"socialcue"}
    assert env.store.frame_count(SID_A) == 1  # only the flagged call stored a frame


def test_default_calls_never_touch_the_session_store(env):
    env.attach(Feed(syn.generate("calm")))
    for _ in range(5):
        assert env.post(flag=None).status_code == 200
    assert len(env.store) == 0


def test_block_has_exact_keys_and_the_banner(env):
    blk = _run(env, "calm", 1)
    assert set(blk) == BLOCK_KEYS | {"body"}
    assert set(blk["insufficient"]) == {"flag", "reasons"} and set(blk["window"]) == {"seconds", "usable_frames", "fps"}
    assert blk["banner"] == scs.BANNER and "not a reading of feelings" in blk["banner"]
    assert blk["heuristic"] == hb1.HEURISTIC == scs.HEURISTIC
    assert blk["proxy_eval_ref"] is None
    assert blk["insufficient"]["flag"] is True and "window_filling" in blk["insufficient"]["reasons"]


def test_flag_values_other_than_one_do_not_enable(env):
    env.attach(Feed(syn.generate("calm")))
    assert "socialcue" not in env.post(flag=2).json()


# --- accumulation -----------------------------------------------------------

def test_state_accumulates_and_appears_only_with_enough_coverage(env):
    env.attach(Feed(syn.generate("nervous-like", duration=8.0)))
    seen = []
    for i in range(64):
        blk = env.post().json()["socialcue"]
        seen.append((i, blk))
    early = [b for i, b in seen if i < 10]
    assert all(b["states"] == [] and b["insufficient"]["flag"] for b in early)
    final = seen[-1][1]
    assert not final["insufficient"]["flag"]
    assert [s["name"] for s in final["states"]] == ["nervous-like"]
    assert 0 < final["states"][0]["confidence"] <= 0.8
    first_state = next(i for i, b in seen if b["states"])
    assert first_state >= 15  # needs >= 2 s of window and >= 0.6 coverage
    assert env.store.frame_count(SID_A) == 64


def test_matches_the_pure_module_on_the_same_frames(env):
    recs = syn.generate("unsure-like")
    env.attach(Feed(recs))
    for _ in range(len(recs)):
        blk = env.post().json()["socialcue"]
    expected = hb1.analyze([hb1.FrameRecord(1000.0 + 0.125 * (i + 1), r.expression, r.quality, r.arousal, 0.0, r.box,
                                            r.landmarks, 1) for i, r in enumerate(recs)])
    assert [s["name"] for s in blk["states"]] == [s["name"] for s in expected["states"]] == ["unsure-like"]


def test_face_dropouts_leave_the_block_insufficient(env):
    env.attach(Feed(syn.generate("dropout", duration=8.0)))
    for _ in range(64):
        blk = env.post().json()["socialcue"]
    assert blk["insufficient"]["flag"] and "low_coverage" in blk["insufficient"]["reasons"] and blk["states"] == []


def test_timestamps_come_from_the_server_clock(env):
    env.attach(Feed(syn.generate("calm")))
    for _ in range(3):
        env.post()
    frames = list(env.store._sessions[SID_A]["frames"])
    assert [round(f.t, 3) for f in frames] == [1000.125, 1000.25, 1000.375]


# --- TTL, caps, isolation ----------------------------------------------------

def test_ttl_expiry(env):
    env.attach(Feed(syn.generate("calm")))
    env.post()
    assert env.store.frame_count(SID_A) == 1
    env.clock.t += 59.0
    assert len(env.store) == 1
    env.clock.t += 2.0  # idle 61 s > 60 s
    assert len(env.store) == 0 and env.store.frame_count(SID_A) == 0
    blk = env.post().json()["socialcue"]  # a fresh session restarts the window
    assert env.store.frame_count(SID_A) == 1 and "window_filling" in blk["insufficient"]["reasons"]


def test_frame_cap_and_oldest_eviction(env):
    env.attach(Feed(syn.generate("calm", duration=10.0)))
    for _ in range(80):
        env.post()
    frames = list(env.store._sessions[SID_A]["frames"])
    assert len(frames) == scs.MAX_FRAMES == 64
    assert frames[0].t == pytest.approx(1000.0 + 0.125 * 17)  # the oldest 16 were dropped


def test_session_cap_evicts_the_least_recently_used(env):
    env.attach(Feed(syn.generate("calm", duration=10.0)))
    ids = [f"session-{i:04d}-xxxxxxxxxxxxxx" for i in range(scs.MAX_SESSIONS)]
    for sid in ids:
        env.post(sid)
    assert len(env.store) == 8
    env.post(ids[0])  # touch the oldest so ids[1] becomes least recently used
    env.post("session-new0-xxxxxxxxxxxxxx")
    assert len(env.store) == 8
    assert env.store.frame_count(ids[1]) == 0 and env.store.frame_count(ids[0]) == 2


def test_two_sessions_are_isolated(env):
    nerv = Feed(syn.generate("nervous-like", duration=8.0))
    calm = Feed(syn.generate("calm", duration=8.0))
    out = {}
    for _ in range(64):
        for sid, feed in ((SID_A, nerv), (SID_B, calm)):
            env.attach(feed)
            out[sid] = env.post(sid, step=0.0625).json()["socialcue"]  # two posts per 0.125 s tick
    assert [s["name"] for s in out[SID_A]["states"]] == ["nervous-like"]
    assert out[SID_B]["states"] == [] and not out[SID_B]["insufficient"]["flag"]
    assert env.store.frame_count(SID_A) == env.store.frame_count(SID_B) == 64


def test_delete_clears_one_session(env):
    env.attach(Feed(syn.generate("calm")))
    env.post(SID_A)
    env.post(SID_B)
    r = env.client.delete("/api/socialcue/session", headers={HDR: SID_A})
    assert r.status_code == 200 and r.json() == {"ok": True, "cleared": True}
    assert env.store.frame_count(SID_A) == 0 and env.store.frame_count(SID_B) == 1
    assert env.client.delete("/api/socialcue/session", headers={HDR: SID_A}).json()["cleared"] is False
    assert env.client.delete("/api/socialcue/session").json()["cleared"] is False


# --- refusals ------------------------------------------------------------------

def test_multiple_faces_never_produce_states(env):
    env.attach(Feed(syn.generate("nervous-like", duration=8.0), extra_faces=1))
    for _ in range(64):
        blk = env.post().json()["socialcue"]
    assert blk["states"] == [] and blk["insufficient"]["flag"]
    assert "multiple_faces" in blk["insufficient"]["reasons"]
    # only the largest face was used for the stored record
    frames = list(env.store._sessions[SID_A]["frames"])
    assert all(f.face_count == 2 and f.box[2] == 100.0 for f in frames)


def test_no_session_id_is_stateless_and_insufficient(env):
    env.attach(Feed(syn.generate("nervous-like", duration=8.0)))
    for _ in range(40):
        blk = env.post(None).json()["socialcue"]
    assert blk["states"] == [] and blk["insufficient"]["flag"]
    assert "no_session_id" in blk["insufficient"]["reasons"]
    assert len(env.store) == 0


@pytest.mark.parametrize("bad", ["short", "has space in it", "x" * 65, "semi;colon-12345", "a" * 21, "session-aaaa-0001"])
def test_invalid_session_id_is_stateless_and_insufficient(env, bad):
    env.attach(Feed(syn.generate("calm")))
    blk = env.post(bad).json()["socialcue"]
    assert "invalid_session_id" in blk["insufficient"]["reasons"] and blk["states"] == []
    assert len(env.store) == 0


# --- failure containment ------------------------------------------------------

def test_exception_in_the_cue_path_returns_200_with_internal_error(env, monkeypatch):
    env.attach(Feed(syn.generate("calm")))

    def boom(*a, **k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(hb1, "analyze", boom)
    r = env.post()
    assert r.status_code == 200
    j = r.json()
    blk = j["socialcue"]
    assert set(blk) == BLOCK_KEYS and blk["states"] == []
    assert blk["insufficient"] == {"flag": True, "reasons": ["internal_error"]}
    assert set(j) - {"socialcue"} == {"selected_model", "face", "body", "fusion", "ethics_notice"}


def test_failure_building_the_record_is_contained(env, monkeypatch):
    env.attach(Feed(syn.generate("calm")))
    monkeypatch.setattr(scs, "_make_record_fn", lambda *a, **k: (_ for _ in ()).throw(ValueError("x")))
    r = env.post()
    assert r.status_code == 200 and r.json()["socialcue"]["insufficient"]["reasons"] == ["internal_error"]


def test_failure_of_the_block_builder_itself_cannot_produce_a_500(env, monkeypatch):
    env.attach(Feed(syn.generate("calm")))
    monkeypatch.setattr(scs, "socialcue_block", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("y")))
    r = env.post()
    assert r.status_code == 200
    blk = r.json()["socialcue"]
    assert set(blk) == BLOCK_KEYS and blk["insufficient"]["reasons"] == ["internal_error"] and blk["states"] == []
    assert blk["banner"] == scs.BANNER  # the shared banner, not a shorter private copy


def test_fallback_literals_match_the_pure_module():
    assert scs.HEURISTIC == hb1.HEURISTIC
    assert scs.insufficient_block("internal_error")["window"]["seconds"] == hb1.HB1_CONFIG["window"]["seconds"]


# --- no persistence ---------------------------------------------------------------

def test_no_files_are_written(env, monkeypatch, tmp_path):
    writes = []
    real_open = builtins.open

    def spy(file, mode="r", *a, **k):
        if any(c in str(mode) for c in "wax+"):
            writes.append((str(file), mode))
        return real_open(file, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", spy)
    env.attach(Feed(syn.generate("nervous-like", duration=8.0)))
    before = sorted(p.name for p in tmp_path.iterdir())
    for _ in range(64):
        assert env.post().status_code == 200
    env.client.delete("/api/socialcue/session", headers={HDR: SID_A})
    assert writes == [] and sorted(p.name for p in tmp_path.iterdir()) == before


def test_session_module_has_no_io_or_logging_code():
    src = (ROOT / "backend" / "app" / "socialcue_session.py").read_text(encoding="utf-8")
    for needle in ("open(", ".write(", "logging", "print(", "imwrite", "tempfile", "pickle", "json.dump", "sqlite"):
        assert needle not in src, needle
    assert "os.environ" not in src
