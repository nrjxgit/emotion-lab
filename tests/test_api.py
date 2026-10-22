from fastapi.testclient import TestClient
import numpy as np, cv2, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "app"))


def _gray_jpeg():
    img = np.full((240, 320, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_out_of_frame_box_never_500(monkeypatch):
    import main

    monkeypatch.setattr(
        main,
        "detect_faces",
        lambda img: [{"x": -9999, "y": -9999, "w": 10, "h": 10, "confidence": 0.9}],
    )
    r = TestClient(main.app).post(
        "/api/infer", files={"image": ("g.jpg", _gray_jpeg(), "image/jpeg")}
    )
    assert r.status_code == 200
    assert r.json()["face"]["face_found"] is False


def test_health():
    from main import app

    c = TestClient(app)
    r = c.get("/api/health")
    assert r.status_code == 200 and "status" in r.json()


def test_infer_gray():
    from main import app

    c = TestClient(app)
    r = c.post("/api/infer", files={"image": ("gray.jpg", _gray_jpeg(), "image/jpeg")})
    assert r.status_code == 200
    j = r.json()
    assert abs(sum(j["fusion"]["probs"].values()) - 1.0) < 1e-4
    assert "uncertainty" in j["fusion"] and "ethics_notice" in j
