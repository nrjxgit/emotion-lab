from fastapi.testclient import TestClient
import numpy as np, cv2, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "app"))


def _jpeg(img):
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_model_info_lists_real_models():
    from main import app

    c = TestClient(app)
    j = c.get("/api/model-info").json()
    assert j["expression"]["loaded"] is True
    assert j["expression"]["source"] == "trained-minixception-fer2013"
    assert j["face_detector"]["name"] == "YuNet-2023mar"
    assert j["body"]["type"] == "motion-only"


def test_gray_no_face_found():
    from main import app

    c = TestClient(app)
    gray = np.full((240, 320, 3), 128, dtype=np.uint8)
    r = c.post("/api/infer", files={"image": ("g.jpg", _jpeg(gray), "image/jpeg")})
    assert r.status_code == 200
    j = r.json()
    assert j["face"]["face_found"] is False
    assert j["face"]["expression"]["source"] == "trained-minixception-fer2013"
    assert j["fusion"]["probs"]["unknown"] > 0.4


def test_corrupt_rejected():
    from main import app

    c = TestClient(app)
    r = c.post(
        "/api/infer", files={"image": ("x.jpg", b"not-an-image" * 10, "image/jpeg")}
    )
    assert r.status_code in (400, 422)


def test_webcam_error_typed():
    from webcam import open_capture, WebcamError
    import pytest

    with pytest.raises(WebcamError):
        open_capture(99)


def test_body_labeled_motion_only():
    sys.path.insert(0, str(ROOT / "backend" / "app" / "body"))
    from movement import movement_features, predict_body_emotion

    r = predict_body_emotion(movement_features([]))
    assert r["type"] == "motion-only"


def test_session_flow_consent_gated():
    from main import app

    c = TestClient(app)
    assert c.post("/api/session/start", json={"consent": False}).status_code == 422
    assert (
        c.post(
            "/api/session/start", json={"consent": True, "participant": "t"}
        ).status_code
        == 200
    )
    r = c.post(
        "/api/session/record",
        json={
            "participant": "t",
            "ai_label": "happiness",
            "self_report": "happiness",
        },
    )
    assert r.status_code == 200
    j = c.get("/api/session/report", params={"participant": "t"}).json()
    assert j["agreement_rate"] == 1.0 and j["n"] == 1


def test_calibrate_requires_consent(monkeypatch, tmp_path):
    from main import app

    monkeypatch.setenv("PERSONAL_BASELINE_PATH", str(tmp_path / "b.json"))
    c = TestClient(app)
    assert c.post("/api/calibrate", json={"aus": []}).status_code == 422
    r = c.post("/api/calibrate", json={"consent": True, "aus": []})
    assert r.status_code == 200 and r.json()["ok"] is True


def test_calibrate_writes_only_to_configured_store(monkeypatch, tmp_path):
    import main

    store = tmp_path / "baseline.json"
    monkeypatch.setenv("PERSONAL_BASELINE_PATH", str(store))
    c = TestClient(main.app)
    c.post(
        "/api/calibrate",
        json={"consent": True, "aus": [{"code": "AU12", "intensity": 0.5}]},
    )
    assert store.exists()


def test_session_delete_removes_participant():
    from main import app

    c = TestClient(app)
    c.post("/api/session/start", json={"consent": True, "participant": "delme"})
    assert c.delete("/api/session", params={"participant": "delme"}).status_code == 200
    assert (
        c.get("/api/session/report", params={"participant": "delme"}).status_code == 422
    )


def test_oversized_upload_rejected():
    from main import app

    c = TestClient(app)
    big = b"\xff" * (6 * 1024 * 1024)
    r = c.post("/api/infer", files={"image": ("big.jpg", big, "image/jpeg")})
    assert r.status_code == 413
