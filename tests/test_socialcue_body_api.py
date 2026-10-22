"""S31: manufactured landmarks through in-process TestClient; no pose model or camera."""
import builtins
import io
import os
import copy

import pytest
from fastapi.testclient import TestClient

from test_socialcue_api import Feed, FakeClock, _jpeg, SID_A, SID_B
import main
import socialcue_session as scs
from socialcue import body as B, state as hb1, synthetic as syn, synthetic_skeleton as sk


@pytest.fixture
def env(monkeypatch):
    clock = FakeClock()
    store = scs.SocialCueSessions(clock=clock)
    monkeypatch.setattr(scs, "STORE", store)
    monkeypatch.setattr(main, "_track", [])
    seq = sk.build("calm")
    feed = Feed(syn.generate("nervous-like", fps=10, duration=8))
    original_detect = feed.detect

    def detect(image):
        faces = original_detect(image)
        for face in faces[:1]:  # Align the primary face; keep bystanders inside the image.
            face["y"] -= 100
            face["landmarks5"] = {name: (x, y - 100) for name, (x, y) in face["landmarks5"].items()}
        return faces

    monkeypatch.setattr(main, "detect_faces", detect)
    monkeypatch.setattr(main, "get_predictor", lambda profile: feed)
    monkeypatch.setattr(main, "yunet_ok", lambda: False)
    client = TestClient(main.app)
    jpg = _jpeg()

    class Env:
        model = True
        def post(self, sid=SID_A, flag="1", query="", advance=True):
            if advance:
                feed.advance()
            clock.t = 1000.0 + feed.i / 10
            headers = {scs.HEADER: sid} if sid is not None else {}
            url = "/api/infer" + ("?socialcue=" + flag + query if flag is not None else "")
            response = client.post(url, files={"image": ("synthetic.jpg", self.jpg, "image/jpeg")}, headers=headers)
            assert response.status_code == 200, response.text
            return response
        def run(self, name, n=30):
            self.seq = sk.build(name, n=n)
            for _ in range(n):
                response = self.post()
            return response.json()["socialcue"]

    e = Env()
    e.seq, e.store, e.clock, e.feed = seq, store, clock, feed
    e.jpg, e.client = jpg, client

    def pose(image):
        if not e.model:
            return {"visible": False, "keypoints": [], "quality": 0.0, "estimator_available": False}
        return {"visible": True, "keypoints": e.seq.lm[min(feed.i, len(e.seq.lm) - 1)], "quality": 0.9,
                "timestamp": -999999}  # deliberately untrusted estimator metadata
    monkeypatch.setattr(main, "estimate_pose", pose)
    return e


def pure_result(env):
    entry = env.store._sessions[SID_A]
    # Preserve the same retention/coverage origin as the request under test.
    result = hb1.analyze(hb1.PacedRecords(entry["frames"], entry["bin_origin"]))
    result["banner"] = scs.BANNER
    return result


@pytest.mark.parametrize("name", ["calm", "fidget", "self_touch", "occluded", "swap"])
def test_manufactured_landmarks_reach_body_extractor_and_caps(env, name):
    result = env.run(name)
    block = result["body"]
    if name == "swap":
        assert block["status"] == "none" and block["reason"] == "window_filling"
        assert len(env.store._sessions[SID_A]["poses"]) == 15  # identity reset at manufactured swap
    else:
        assert block["status"] == "ok", block
        assert block["bystander_checked"] is True
        assert block["window"]["frames"] == 30
        assert all(0 <= cue["confidence"] <= 0.6 for cue in block["cues"])
        present = {cue["cue"] for cue in block["cues"] if cue["present"]}
        if name == "fidget": assert "fidget" in present
        if name == "self_touch": assert "self_touch" in present
        if name == "occluded":
            assert any(cue["status"] == "none" for cue in block["cues"])
    baseline = pure_result(env)
    assert {s["name"] for s in result["states"]} == {s["name"] for s in baseline["states"]}
    for state, original in zip(result["states"], baseline["states"]):
        assert 0 <= state["confidence"] <= 0.8
        if block["status"] == "ok":
            delta = state["body_support"]["delta"]
            assert -0.05 <= delta <= 0.10
            assert state["confidence"] == round(min(0.8, max(0, original["confidence"] + delta)), 4)
    if block["status"] != "ok":
        assert {k: v for k, v in result.items() if k != "body"} == baseline


def test_no_pose_reports_unavailable_and_hb1_is_exactly_unchanged(env):
    env.model = False
    result = env.run("calm")
    assert result["body"]["status"] == "unavailable"
    assert result["body"]["reason"] == "no_pose_model"
    assert {k: v for k, v in result.items() if k != "body"} == pure_result(env)
    assert list(env.store._sessions[SID_A]["poses"]) == []


def test_default_and_off_response_bytes_unchanged_and_no_socialcue_frames(env, monkeypatch):
    default = env.post(flag=None)
    monkeypatch.setattr(main, "_track", [])
    explicit_off = env.post(flag="0", advance=False)
    assert default.content == explicit_off.content
    assert "socialcue" not in default.json()
    assert env.store.frame_count(SID_A) == 0
    assert list(env.store._sessions[SID_A]["poses"]) == []  # valid header retains only legacy movement
    def forbidden(*args, **kwargs):
        raise AssertionError("default route must not enter Stage 2")
    monkeypatch.setattr(B, "extract_body_cues", forbidden)
    monkeypatch.setattr(B, "attach_body", forbidden)
    monkeypatch.setattr(main, "_track", [])
    assert env.post(flag=None, advance=False).content == default.content


@pytest.mark.parametrize("sid", [None, "invalid"])
def test_no_valid_consent_session_never_extracts_or_stores_pose(env, monkeypatch, sid):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not extract without session consent")
    monkeypatch.setattr(B, "extract_body_cues", forbidden)
    result = env.post(sid=sid).json()["socialcue"]
    assert result["body"]["reason"] == "consent_off"
    assert result["states"] == [] and len(env.store) == 0


def test_multi_face_refusal_resets_body_history(env):
    env.run("fidget")
    env.feed.extra_faces = 1
    result = env.post().json()["socialcue"]
    assert result["states"] == [] and "multiple_faces" in result["insufficient"]["reasons"]
    assert result["body"]["reason"] == "multiple_people"
    assert len(env.store._sessions[SID_A]["poses"]) == 0
    env.feed.extra_faces = 0
    # No window can silently reuse the pre-bystander landmarks.
    result = env.post().json()["socialcue"]
    assert result["body"]["status"] != "ok"


def test_pose_ring_server_time_numeric_snapshot_caps_ttl_and_isolation(env):
    env.seq = sk.build("calm", n=70)
    for _ in range(70): env.post(query="&timestamp=999999999")
    poses = list(env.store._sessions[SID_A]["poses"])
    assert len(poses) == scs.MAX_FRAMES == 64
    assert poses[-1]["t"] == env.clock.t == 1006.9
    assert poses[0]["t"] == 1000.6
    assert set(poses[-1]) == {"t", "lm", "size", "face_count", "face_box"}
    assert isinstance(poses[-1]["lm"], tuple) and len(poses[-1]["lm"]) == 33
    snapshot = copy.deepcopy(poses[-1]["lm"])
    env.seq.lm[-1][0][0] = 99
    assert poses[-1]["lm"] == snapshot
    env.post(sid=SID_B, advance=False)
    assert len(env.store._sessions[SID_B]["poses"]) == 1
    env.clock.t += scs.TTL_SECONDS + 1
    env.store.purge()
    assert len(env.store) == 0


def test_lost_pose_clears_history_and_does_not_reuse_stale_evidence(env):
    env.run("fidget")
    env.model = False
    result = env.post().json()["socialcue"]
    assert result["body"]["reason"] == "no_pose_model"
    assert len(env.store._sessions[SID_A]["poses"]) == 0
    env.model = True
    assert env.post().json()["socialcue"]["body"]["reason"] == "window_filling"


def test_body_cannot_rescue_insufficient_hb1_or_create_a_state(env):
    result = env.post().json()["socialcue"]
    assert result["states"] == [] and result["insufficient"]["flag"]
    # Calm expression window with rich body movement still creates no state.
    env.feed.records = syn.generate("calm", fps=10, duration=8)
    result = env.run("self_touch")
    assert result["body"]["status"] == "ok" and result["states"] == []


def test_unchecked_face_box_keeps_existing_body_cue_cap():
    seq = sk.build("self_touch")
    extracted = B.extract_body_cues(seq.t, seq.lm, seq.W, seq.H, consent=True, face_count=1)
    evidence = B.body_evidence(extracted)
    assert evidence and all(not e["bystander_checked"] and e["confidence"] <= B.UNCHECKED_CUE_CAP for e in evidence)


def test_requests_write_no_files_and_keep_no_image(env, monkeypatch):
    opened = []
    def guard(original):
        def checked(file, mode="r", *args, **kwargs):
            if any(c in str(mode) for c in "wax+"):
                opened.append(str(file))
                raise AssertionError("request attempted a file write")
            return original(file, mode, *args, **kwargs)
        return checked
    monkeypatch.setattr(builtins, "open", guard(builtins.open))
    monkeypatch.setattr(io, "open", guard(io.open))
    original_os_open = os.open
    def checked_os_open(file, flags, *args, **kwargs):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            opened.append(str(file))
            raise AssertionError("request attempted os.open write")
        return original_os_open(file, flags, *args, **kwargs)
    monkeypatch.setattr(os, "open", checked_os_open)
    result = env.run("fidget")
    assert result["body"]["status"] == "ok" and opened == []
    assert all(set(p) == {"t", "lm", "size", "face_count", "face_box"} for p in env.store._sessions[SID_A]["poses"])


@pytest.mark.parametrize("case", ["invisible", "empty", "missing", "low_quality", "threshold_quality"])
def test_s33_working_estimator_without_visible_body_does_not_append(env, monkeypatch, case):
    env.run("fidget")
    env.post(sid=SID_B, advance=False)
    other = copy.deepcopy(list(env.store._sessions[SID_B]["poses"]))
    payload = {"visible": True, "keypoints": env.seq.lm[0], "quality": 0.9, "estimator_available": True}
    if case == "invisible": payload["visible"] = False
    if case == "empty": payload["keypoints"] = []
    if case == "missing": payload.pop("keypoints")
    if case == "low_quality": payload["quality"] = 0.2
    if case == "threshold_quality": payload["quality"] = 0.3
    # Missing/empty payload with visible True would enter the legacy motion path;
    # that existing path is bypassed here to exercise the session boundary directly.
    r = env.feed.records[env.feed.i]
    face = env.feed.detect(None)[0]
    result = scs.socialcue_block(SID_A, [face], face,
                                {"face_found": True, "expression": env.feed.predict(None, None)}, r.quality,
                                pose=payload, width=640, height=480)
    assert result["body"]["reason"] == "pose_not_visible"
    assert list(env.store._sessions[SID_A]["poses"]) == []
    assert list(env.store._sessions[SID_B]["poses"]) == other
    assert {k: v for k, v in result.items() if k != "body"} == pure_result(env)


@pytest.mark.parametrize("case", ["nan", "short", "none", "point_none", "huge_int", "bool", "quality_nan"])
def test_s33_malformed_pose_payload_through_session_preserves_hb1(env, case):
    env.run("fidget")
    payload = {"visible": True, "quality": 0.9, "keypoints": copy.deepcopy(env.seq.lm[0]), "estimator_available": True}
    if case == "nan": payload["keypoints"][0][0] = float("nan")
    if case == "short": payload["keypoints"].pop()
    if case == "none": payload["keypoints"] = None
    if case == "point_none": payload["keypoints"][0] = None
    if case == "huge_int": payload["keypoints"][0][0] = 10**400
    if case == "bool": payload["keypoints"][0][0] = True
    if case == "quality_nan": payload["quality"] = float("nan")
    r = env.feed.records[env.feed.i]
    face = env.feed.detect(None)[0]
    result = scs.socialcue_block(SID_A, [face], face,
                                {"face_found": True, "expression": env.feed.predict(None, None)}, r.quality,
                                pose=payload, width=640, height=480)
    assert result["body"]["reason"] == "invalid_pose_payload"
    assert list(env.store._sessions[SID_A]["poses"]) == []
    assert {k: v for k, v in result.items() if k != "body"} == pure_result(env)


def test_s33_size_change_resets_pose_history(env):
    import cv2
    import numpy as np
    env.run("fidget")
    ok, jpg = cv2.imencode(".jpg", np.full((600, 800, 3), 128, dtype=np.uint8))
    assert ok
    env.jpg = jpg.tobytes()
    result = env.post().json()["socialcue"]
    poses = list(env.store._sessions[SID_A]["poses"])
    assert len(poses) == 1 and poses[0]["size"] == (800, 600)
    assert result["body"]["reason"] == "window_filling"


def test_s33_delete_clears_pose_and_face_rings(env):
    env.run("fidget")
    assert env.store._sessions[SID_A]["poses"]
    response = env.client.delete("/api/socialcue/session", headers={scs.HEADER: SID_A})
    assert response.status_code == 200 and response.json()["cleared"] is True
    assert SID_A not in env.store._sessions and env.store.frame_count(SID_A) == 0
    result = env.post().json()["socialcue"]
    assert len(env.store._sessions[SID_A]["poses"]) == 1
    assert result["body"]["reason"] == "window_filling"


def test_s33_lru_eviction_drops_whole_pose_ring(env):
    env.post()
    for i in range(scs.MAX_SESSIONS - 1):
        env.post(sid=f"synthetic-session-{i:02d}-abcdefgh")
    # Touch original to make a different ring the oldest.
    env.post(sid=SID_A)
    env.post(sid="synthetic-session-new-abcdefgh")
    assert SID_A in env.store._sessions
    assert "synthetic-session-00-abcdefgh" not in env.store._sessions
    assert len(env.store) == scs.MAX_SESSIONS
    assert all(entry["poses"] for entry in env.store._sessions.values())


@pytest.mark.parametrize("operation", ["extract_body_cues", "attach_body"])
def test_s33_body_exception_is_http_200_with_intact_hb1(env, monkeypatch, operation):
    env.run("fidget")
    def boom(*args, **kwargs): raise RuntimeError("manufactured body failure")
    monkeypatch.setattr(B, operation, boom)
    result = env.post().json()["socialcue"]
    assert result["body"]["reason"] == "internal_error"
    assert {k: v for k, v in result.items() if k != "body"} == pure_result(env)


def test_s33_estimator_timestamp_is_never_read(env, monkeypatch):
    class TimestampTrap(dict):
        def get(self, key, *args):
            if key in ("timestamp", "t"): raise AssertionError("untrusted timestamp read")
            return super().get(key, *args)
        def __getitem__(self, key):
            if key in ("timestamp", "t"): raise AssertionError("untrusted timestamp read")
            return super().__getitem__(key)
    monkeypatch.setattr(main, "estimate_pose", lambda image: TimestampTrap(
        visible=True, quality=0.9, keypoints=env.seq.lm[0], estimator_available=True))
    result = env.run("calm")
    assert result["body"]["status"] == "ok"
    assert env.store._sessions[SID_A]["poses"][-1]["t"] == env.clock.t


def test_s33_estimator_explicitly_distinguishes_missing_model_from_no_detection(monkeypatch):
    import sys
    import numpy as np
    from types import SimpleNamespace
    from body import pose as pose_module
    original = builtins.__import__
    def no_model(name, *args, **kwargs):
        if name == "mediapipe": raise ModuleNotFoundError("manufactured absent model")
        return original(name, *args, **kwargs)
    with monkeypatch.context() as scoped:
        scoped.setattr(builtins, "__import__", no_model)
        absent = pose_module.estimate_pose(np.zeros((8, 8, 3), dtype=np.uint8))
    assert absent["estimator_available"] is False
    fake = SimpleNamespace(process=lambda image: SimpleNamespace(pose_landmarks=None))
    monkeypatch.setitem(sys.modules, "mediapipe", SimpleNamespace())
    monkeypatch.setattr(pose_module, "_pose", fake, raising=False)
    empty = pose_module.estimate_pose(np.zeros((8, 8, 3), dtype=np.uint8))
    assert empty["estimator_available"] is True and empty["visible"] is False


# S35: complete-route regressions; manufactured data only.
def test_s35_movement_is_session_keyed_absence_delete_and_ttl(env):
    env.run("fidget", n=6)
    first_b = env.post(sid=SID_B).json()
    assert first_b["body"]["visible"] is False
    env.model = False
    env.post(sid=SID_A)
    env.model = True
    assert env.post(sid=SID_A).json()["body"]["visible"] is False
    for _ in range(5): env.post()
    env.client.delete("/api/socialcue/session", headers={scs.HEADER: SID_A})
    assert env.post().json()["body"]["visible"] is False
    for _ in range(5): env.post()
    env.clock.t += scs.TTL_SECONDS + 1
    env.store.purge()
    assert env.post().json()["body"]["visible"] is False


def test_s35_anonymous_unflagged_requests_keep_no_movement_history(env):
    for _ in range(8):
        result = env.post(sid=None, flag=None).json()
        assert result["body"]["visible"] is False
    assert main._track == [] and len(env.store) == 0


def test_s35_single_caller_default_matches_pre_fix_legacy_sequence(env):
    import hashlib
    import json
    oracle, responses = [], []
    for _ in range(8):
        response = env.post(flag=None).json()
        oracle.append([(p[0], p[1]) for p in env.seq.lm[env.feed.i]])
        expected_body = main.predict_body_emotion(main.movement_features(oracle if len(oracle) >= 5 else []))
        expr = response["face"]["expression"]
        expected_fusion = main.fuse(
            {k: expr[k] for k in ("probs", "confidence", "uncertainty")},
            {k: expected_body[k] for k in ("probs", "confidence", "uncertainty")},
            {"face_q": response["face"]["quality"], "body_q": 0.9})
        assert response["body"] == expected_body and response["fusion"] == expected_fusion
        assert set(response) == {"selected_model", "face", "body", "fusion", "ethics_notice"}
        responses.append(response)
    print("S35_SINGLE_CALLER_DEFAULT_SHA256=" + hashlib.sha256(json.dumps(responses, sort_keys=True).encode()).hexdigest())


@pytest.mark.parametrize("payload", ["32", "100000", "nan", "short_point", "quality_string"])
def test_s35_bad_estimator_rejected_before_legacy_consumer_and_mutation(env, monkeypatch, payload):
    env.run("fidget", n=4)
    points = copy.deepcopy(env.seq.lm[0])
    quality = 0.9
    if payload == "32": points = points[:32]
    if payload == "100000": points = [(0.5, 0.5, 0.9)] * 100000
    if payload == "nan": points[0][0] = float("nan")
    if payload == "short_point": points[0] = (0.5,)
    if payload == "quality_string": quality = "not-a-number"
    monkeypatch.setattr(main, "estimate_pose", lambda image: {"visible": True, "quality": quality, "keypoints": points})
    response = env.post().json()
    assert response["socialcue"]["body"]["reason"] == "invalid_pose_payload"
    assert response["body"]["visible"] is False
    assert main._track == []
    assert not env.store._sessions[SID_A].get("movement")


def test_s35_upload_read_is_limit_plus_one(env):
    import asyncio
    from fastapi import HTTPException
    sizes = []
    class Upload:
        async def read(self, size=-1):
            sizes.append(size)
            return b"x" * (5 * 1024 * 1024 + 1)
    async def check():
        with pytest.raises(HTTPException) as exc:
            await main.infer(image=Upload(), socialcue="0", x_socialcue_session=None)
        assert exc.value.status_code == 413
    asyncio.run(check())
    assert sizes == [5 * 1024 * 1024 + 1]


def test_s35_pixel_header_rejected_before_decode(env, monkeypatch):
    import struct
    import zlib
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 4096, 4096, 8, 2, 0, 0, 0))
            + chunk(b"tEXt", b"manufactured\0" + b"x" * 120)
            + chunk(b"IDAT", zlib.compress(b"")) + chunk(b"IEND", b""))
    def forbidden(*args, **kwargs): raise AssertionError("oversized dimensions reached allocating decoder")
    monkeypatch.setattr(main.cv2, "imdecode", forbidden)
    response = env.client.post("/api/infer", files={"image": ("synthetic.png", data, "image/png")})
    assert response.status_code == 413
    assert "pixel" in response.json()["detail"]


def test_s35_delete_during_paused_upload_cannot_resurrect(env):
    import asyncio
    from fastapi import HTTPException
    env.post()
    async def check():
        started, resume = asyncio.Event(), asyncio.Event()
        class Upload:
            async def read(self, size=-1):
                started.set()
                await resume.wait()
                return env.jpg[:size] if size >= 0 else env.jpg
        pending = asyncio.create_task(main.infer(image=Upload(), socialcue="1", x_socialcue_session=SID_A))
        await started.wait()
        assert main.socialcue_session_clear(SID_A)["cleared"]
        resume.set()
        try:
            await pending
        except HTTPException as exc:
            assert exc.status_code == 409
        else:
            pytest.fail("pre-DELETE upload was not refused")
        assert env.store.frame_count(SID_A) == 0
    asyncio.run(check())


def test_s35_upload_admission_is_bounded_without_waiting(env):
    import asyncio
    from fastapi import HTTPException
    async def check():
        started, resume = asyncio.Event(), asyncio.Event()
        count = 0
        limit = getattr(main, "MAX_UPLOAD_REQUESTS", 4)
        class Upload:
            async def read(self, size=-1):
                nonlocal count
                count += 1
                if count == limit: started.set()
                await resume.wait()
                return env.jpg[:size] if size >= 0 else env.jpg
        class MustNotRead:
            async def read(self, size=-1): raise AssertionError("over-cap request reached upload read")
        # No face is needed for this admission/await-only test.
        env.feed.advance()
        tasks = [asyncio.create_task(main.infer(image=Upload(), socialcue="0", x_socialcue_session=None)) for _ in range(limit)]
        await started.wait()
        try:
            with pytest.raises(HTTPException) as exc:
                await main.infer(image=MustNotRead(), socialcue="0", x_socialcue_session=None)
            assert exc.value.status_code == 503
        finally:
            resume.set()
            await asyncio.gather(*tasks)
    asyncio.run(check())


def test_s35_body_exception_remains_additive_on_current_code(env, monkeypatch):
    env.run("fidget")
    monkeypatch.setattr(B, "extract_body_cues", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("body failed")))
    result = env.post().json()["socialcue"]
    assert result["body"]["reason"] == "internal_error"
    assert {k: v for k, v in result.items() if k != "body"} == pure_result(env)


def test_s35_record_builder_may_inspect_store_without_deadlock():
    import threading
    store = scs.SocialCueSessions()
    completed = threading.Event()
    def work():
        store.add(SID_A, lambda now: (now, len(store)))
        completed.set()
    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    assert completed.wait(0.3), "record builder deadlocked under mutation lock"
    thread.join(1)


def test_s35_generation_tombstones_are_bounded_and_eviction_refuses_old_admission():
    store = scs.SocialCueSessions()
    admit = getattr(store, "admit", None)
    assert callable(admit)
    old = admit(SID_A)
    for i in range(300):
        sid = f"synthetic-tombstone-{i:03d}-abcdefgh"
        store.clear(sid)
    assert len(store._generations) <= scs.MAX_GENERATIONS
    assert not store.generation_current(SID_A, old)


def test_s35_body_module_wiring_documentation_is_current():
    assert "it is not wired into the backend" not in B.__doc__.lower()
    assert "stage 2 remains unwired" not in B.__doc__.lower()


@pytest.mark.parametrize("counts", [[0]*8, [2]*8, [0,1,2,0,1,2,0,1]], ids=["no-face", "two-face", "intermittent-face"])
def test_s39_admitted_default_matches_36df602_regardless_of_faces(env, monkeypatch, counts):
    import hashlib
    import json
    import types, sys, subprocess, asyncio
    try:
        source = subprocess.check_output(["git", "show", "36df602:backend/app/main.py"], text=True, encoding="utf-8")
    except (subprocess.CalledProcessError, OSError):
        pytest.skip("S39 git oracle unavailable: requires a git checkout with 36df602 history (not an archive)")
    legacy = types.ModuleType("s39_legacy_reference")
    legacy.__file__ = main.__file__
    monkeypatch.setitem(sys.modules, legacy.__name__, legacy)
    exec(compile(source, "36df602/backend/app/main.py", "exec"), legacy.__dict__)
    legacy.get_predictor = main.get_predictor
    legacy.estimate_pose = main.estimate_pose
    detect = main.detect_faces
    oracle, outputs = [], []
    for count in counts:
        env.feed.extra_faces = max(0, count - 1)
        monkeypatch.setattr(main, "detect_faces", detect if count else lambda image: [])
        result = env.post(flag=None).json()
        legacy.detect_faces = main.detect_faces
        class Upload:
            async def read(self, size=-1): return env.jpg
        old = asyncio.run(legacy.infer(image=Upload(), socialcue="0", x_socialcue_session=SID_A))
        assert result == json.loads(json.dumps(old, allow_nan=False)), "full default response differs from 36df602"
        oracle.append([(x, y) for x, y, _ in env.seq.lm[env.feed.i]])
        expected = main.predict_body_emotion(main.movement_features(oracle if len(oracle) >= 5 else []))
        assert result["body"] == expected
        face = result["face"]
        expected_fusion = main.fuse(
            {k: face["expression"][k] for k in ("probs", "confidence", "uncertainty")} if count else None,
            {k: expected[k] for k in ("probs", "confidence", "uncertainty")},
            {"face_q": face["quality"], "body_q": 0.9})
        assert result["fusion"] == expected_fusion
        assert "socialcue" not in result
        outputs.append(result)
    print("S39_DEFAULT_ORACLE_" + str(counts) + "=" + hashlib.sha256(json.dumps(outputs, sort_keys=True).encode()).hexdigest())


@pytest.mark.parametrize("kind,size", [("JPEG",(2592,1944)), ("JPEG",(4000,3000)), ("JPEG",(4032,3024)), ("MPO",(320,240))], ids=["5MP", "12MP", "12.2MP", "MPO"])
def test_s39_ordinary_photo_header_and_route_are_accepted(env, monkeypatch, kind, size):
    from PIL import Image
    buffer = io.BytesIO()
    image = Image.new("RGB", size, (128,128,128))
    if kind == "MPO":
        image.save(buffer, format="MPO", save_all=True, append_images=[Image.new("RGB", size, (64,64,64))])
        with Image.open(io.BytesIO(buffer.getvalue())) as header: assert header.format == "MPO"
    else:
        image.save(buffer, format="JPEG")
    env.jpg = buffer.getvalue()
    assert len(env.jpg) < 5*1024*1024
    env.model = False
    monkeypatch.setattr(main, "detect_faces", lambda image: [])
    assert env.post(flag=None).status_code == 200


@pytest.mark.parametrize("sid", [SID_A, "synthetic-rejected-new-abcdefgh"])
def test_s39_503_does_not_create_or_refresh_generation(env, sid):
    import asyncio
    from fastapi import HTTPException
    env.store.admit(SID_A); env.store.admit(SID_B)
    before = list(env.store._generations.items()), env.store._next_generation
    held = []
    for _ in range(main.MAX_INFER_REQUESTS):
        assert main._INFER_SLOTS.acquire(False)
        held.append(True)
    class Upload:
        async def read(self, size=-1): return env.jpg
    async def check():
        with pytest.raises(HTTPException) as error:
            await main.infer(image=Upload(), socialcue="0", x_socialcue_session=sid)
        assert error.value.status_code == 503
    try: asyncio.run(check())
    finally:
        for _ in held: main._INFER_SLOTS.release()
    assert (list(env.store._generations.items()), env.store._next_generation) == before


def test_s39_paused_upload_holds_no_inference_slot_or_admission(env):
    import asyncio
    env.feed.advance()
    async def check():
        started, resume = asyncio.Event(), asyncio.Event()
        class Upload:
            async def read(self, size=-1):
                started.set(); await resume.wait(); return env.jpg
        pending = asyncio.create_task(main.infer(image=Upload(), socialcue="0", x_socialcue_session=SID_A))
        await started.wait()
        acquired = []
        try:
            for _ in range(main.MAX_INFER_REQUESTS):
                acquired.append(main._INFER_SLOTS.acquire(False))
            assert all(acquired)
            assert list(env.store._generations.items()) == []
        finally:
            for success in acquired:
                if success: main._INFER_SLOTS.release()
            resume.set(); await pending
    asyncio.run(check())


def test_s39_cooperative_upload_timeout_does_not_admit_and_releases_capacity(env, monkeypatch):
    import asyncio
    from fastapi import HTTPException
    monkeypatch.setattr(main, "UPLOAD_TIMEOUT_S", 0.01)
    class Upload:
        async def read(self, size=-1): await asyncio.Event().wait()
    async def check():
        with pytest.raises(HTTPException) as error:
            await main.infer(image=Upload(), socialcue="0", x_socialcue_session=SID_A)
        assert error.value.status_code == 408
    asyncio.run(check())
    assert not env.store._generations
    acquired = []
    try:
        for _ in range(main.MAX_UPLOAD_REQUESTS): acquired.append(main._UPLOAD_SLOTS.acquire(False))
        assert all(acquired)
    finally:
        for success in acquired:
            if success: main._UPLOAD_SLOTS.release()


@pytest.mark.parametrize("array", [False, True])
def test_s39_bounded_numpy_estimator_payload_is_normalized(array):
    import numpy as np
    from body.pose_payload import validate_pose
    points = np.asarray(sk.build("calm").lm[0], dtype=np.float32)
    payload = validate_pose({"visible": True, "quality": np.float32(0.9),
                             "keypoints": points if array else [tuple(row) for row in points]})
    assert payload.visible and len(payload.keypoints) == 33
    assert all(type(v) is float for point in payload.keypoints for v in point)


def test_s39_movement_import_is_outside_mutation_lock(monkeypatch):
    original = builtins.__import__
    store = scs.SocialCueSessions()
    token = store.admit(SID_A)
    store.movement(SID_A, token, ((0.0,0.0),)*33, ((640,480),160.0,320.0,200.0))
    def inspect_import(name, *args, **kwargs):
        if name == "socialcue.body": assert not store._lock.locked(), "import under mutation lock"
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", inspect_import)
    store.movement(SID_A, token, ((0.0,0.0),)*33, ((640,480),160.0,320.0,200.0))


@pytest.mark.parametrize("deleted", [SID_A, SID_B], ids=["same-session", "unrelated-session"])
def test_s43_delete_during_new_session_upload(env, deleted):
    import asyncio
    from fastapi import HTTPException
    async def check():
        started, resume = asyncio.Event(), asyncio.Event()
        class Upload:
            async def read(self, size=-1):
                started.set()
                await resume.wait()
                return env.jpg[:size] if size >= 0 else env.jpg
        pending = asyncio.create_task(main.infer(image=Upload(), socialcue="1", x_socialcue_session=SID_A))
        await started.wait()
        main.socialcue_session_clear(deleted)
        resume.set()
        if deleted == SID_A:
            with pytest.raises(HTTPException) as exc:
                await pending
            assert exc.value.status_code == 409
            assert env.store.frame_count(SID_A) == 0
        else:
            result = await pending
            assert "socialcue" in result
            assert env.store.frame_count(SID_A) == 1
    asyncio.run(check())


def test_s43_evicted_new_session_delete_token_fails_closed():
    store = scs.SocialCueSessions()
    token = store.capture_generation(SID_A)
    store.clear(SID_A)
    for i in range(scs.MAX_GENERATIONS + 1):
        store.clear(f"synthetic-other-{i:04d}-abcdefgh")
    assert len(store._generations) <= scs.MAX_GENERATIONS
    assert len(store._last_deletes) <= scs.MAX_GENERATIONS
    with pytest.raises(scs.SessionInvalidated):
        store.admit(SID_A, captured=token)


def test_s43_clear_all_invalidates_unregistered_upload():
    store = scs.SocialCueSessions()
    token = store.capture_generation(SID_A)
    store.clear_all()
    with pytest.raises(scs.SessionInvalidated):
        store.admit(SID_A, captured=token)


def test_s43_standalone_ephemeral_session_header():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "frontend-standalone/index.html").read_text(encoding="utf-8")
    assert "const sessionId=crypto.randomUUID()" in source
    assert "'X-Socialcue-Session':sessionId" in source
    assert source.count("crypto.randomUUID()") == 1
    assert "sessionId.length<22" in source
    for storage in ("localStorage", "sessionStorage", "indexedDB", "document.cookie", "caches."):
        assert storage not in source


@pytest.mark.parametrize("kind", ["PNG", "JPEG"])
def test_s45_metadata_count_rejected_before_pillow(kind, monkeypatch):
    import struct, zlib
    import upload_policy
    from PIL import Image
    from fastapi import HTTPException
    if kind == "PNG":
        def chunk(t, data):
            return struct.pack(">I", len(data)) + t + data + struct.pack(">I", zlib.crc32(t + data) & 0xffffffff)
        buf = io.BytesIO(); Image.new("RGB", (8, 8)).save(buf, format="PNG")
        original = buf.getvalue()
        data = original[:33] + chunk(b"abCD", b"") * 300 + original[33:]
    else:
        data = b"\xff\xd8" + b"\xff\xe1\x00\x02" * 300 + _jpeg()[2:]
    def forbidden(*a, **k): raise AssertionError("unbounded metadata reached Pillow")
    monkeypatch.setattr(Image, "open", forbidden)
    with pytest.raises(HTTPException) as error:
        upload_policy.check_image_header(data)
    assert error.value.status_code == 422
    assert "metadata" in error.value.detail


def test_s45_handbuilt_pose_payload_cannot_retain_oversize(env, monkeypatch):
    from body.pose_payload import PosePayload, validate_pose
    payload = PosePayload(True, ((0.0, 0.0, 1.0),) * 100000, 0.9, None)
    assert not validate_pose(payload).visible
    monkeypatch.setattr(main, "estimate_pose", lambda image: payload)
    result = env.post().json()
    assert result["socialcue"]["body"]["reason"] == "invalid_pose_payload"
    assert not env.store._sessions[SID_A]["poses"]


@pytest.mark.parametrize("kind", ["unsupported", "undecodable", "oversize", "profile"])
def test_s45_rejected_image_does_not_admit(env, monkeypatch, kind):
    import struct, zlib
    from fastapi import HTTPException
    data, profile = env.jpg, "corrected"
    expected = 422
    if kind == "unsupported": data = b"GIF89a" + b"x" * 200
    elif kind == "undecodable":
        monkeypatch.setattr(main.cv2, "imdecode", lambda *a: None)
    elif kind == "oversize":
        buf = io.BytesIO()
        from PIL import Image
        Image.new("RGB", (8, 8)).save(buf, format="PNG")
        raw = bytearray(buf.getvalue()); raw[16:24] = struct.pack(">II", 20000, 20000)
        raw[29:33] = struct.pack(">I", zlib.crc32(raw[12:29]) & 0xffffffff)
        data = bytes(raw) + b"x" * 100; expected = 413
    else:
        profile = "zzz"
        def reject(profile): raise HTTPException(status_code=422, detail="unknown profile")
        monkeypatch.setattr(main, "get_predictor", reject)
    before = list(env.store._generations.items()), env.store._next_generation
    response = env.client.post("/api/infer?profile=" + profile,
        files={"image": ("synthetic.jpg", data, "image/jpeg")}, headers={scs.HEADER: SID_A})
    assert response.status_code == expected
    assert (list(env.store._generations.items()), env.store._next_generation) == before


def test_s45_concurrent_new_session_uploads_share_generation(env):
    import asyncio
    async def check():
        ready, resume = asyncio.Event(), asyncio.Event()
        started = 0
        class Upload:
            async def read(self, size=-1):
                nonlocal started
                started += 1
                if started == 2: ready.set()
                await resume.wait()
                return env.jpg
        tasks = [asyncio.create_task(main.infer(image=Upload(), socialcue="1", x_socialcue_session=SID_A)) for _ in range(2)]
        await ready.wait(); resume.set()
        results = await asyncio.gather(*tasks)
        assert all("socialcue" in result for result in results)
        assert env.store.frame_count(SID_A) == 1  # both succeeded; same instant/bin retained once
        assert len(env.store._generations) == 1
    asyncio.run(check())


def test_s45_delete_between_first_admissions_still_rejects():
    store = scs.SocialCueSessions()
    token = store.capture_generation(SID_A)
    store.admit(SID_A, captured=token)
    store.clear(SID_A)
    with pytest.raises(scs.SessionInvalidated):
        store.admit(SID_A, captured=token)


@pytest.mark.parametrize("size", [(2592, 1944), (4000, 3000)], ids=["5MP", "12MP"])
def test_s45_ordinary_png_route_still_accepted(env, monkeypatch, size):
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", size, (128, 128, 128)).save(buffer, format="PNG")
    env.jpg = buffer.getvalue(); env.model = False
    monkeypatch.setattr(main, "detect_faces", lambda image: [])
    assert env.post(flag=None).status_code == 200


@pytest.mark.parametrize("kind", ["PNG", "JPEG", "JPEG-fill"])
def test_s45_metadata_byte_budget_before_pillow(kind, monkeypatch):
    import struct, zlib, upload_policy
    from PIL import Image
    from fastapi import HTTPException
    if kind == "PNG":
        payload = b"x" * (256 * 1024)
        chunk = struct.pack(">I", len(payload)) + b"abCD" + payload + struct.pack(">I", zlib.crc32(b"abCD" + payload) & 0xffffffff)
        buffer = io.BytesIO(); Image.new("RGB", (8, 8)).save(buffer, format="PNG")
        raw = buffer.getvalue(); data = raw[:33] + chunk + raw[33:]
    elif kind == "JPEG":
        data = b"\xff\xd8" + (b"\xff\xe1\xff\xff" + b"x" * 65533) * 5 + _jpeg()[2:]
    else:
        data = b"\xff\xd8" + b"\xff" * (256 * 1024) + _jpeg()[2:]
    def forbidden(*a, **k): raise AssertionError("excess metadata reached Pillow")
    monkeypatch.setattr(Image, "open", forbidden)
    with pytest.raises(HTTPException) as error:
        upload_policy.check_image_header(data)
    assert error.value.status_code == 422


@pytest.mark.parametrize("marker", [b"\xff\xfe", b"\xff\xe1"], ids=["COM", "APP1"])
def test_s47_early_eoi_refused_without_walking_later_markers(marker):
    import time
    from fastapi import HTTPException
    import upload_policy
    # Manufactured SOI/EOI, 1.2 MB empty metadata, then a real synthetic header.
    data = b"\xff\xd8\xff\xd9" + (marker + b"\x00\x02") * 300000 + _jpeg()[2:]
    started = time.perf_counter()
    with pytest.raises(HTTPException) as error:
        upload_policy.check_image_header(data)
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert error.value.status_code == 422
    assert elapsed_ms < 100, f"early EOI refusal took {elapsed_ms:.3f} ms"
    print(f"S47_EARLY_EOI_{marker.hex()}_MS={elapsed_ms:.3f}")


def test_s47_progressive_12mp_jpeg_remains_accepted(env, monkeypatch):
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", (4000, 3000), (128, 128, 128)).save(buffer, format="JPEG", progressive=True)
    env.jpg = buffer.getvalue(); env.model = False
    monkeypatch.setattr(main, "detect_faces", lambda image: [])
    assert env.post(flag=None).status_code == 200
