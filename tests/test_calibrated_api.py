import sys, pathlib
import numpy as np
import cv2
from fastapi.testclient import TestClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "app"))


def _face_jpeg():
    return cv2.imencode(".jpg", np.full((240, 320, 3), 200, np.uint8))[1].tobytes()


def _box(img):
    return [
        {
            "x": 100,
            "y": 60,
            "w": 120,
            "h": 120,
            "confidence": 0.9,
            "landmarks5": {
                "eye_R": (150, 100),
                "eye_L": (190, 100),
                "nose": (170, 120),
                "mouth_R": (155, 150),
                "mouth_L": (185, 150),
            },
        }
    ]


def _post(client, qs=""):
    return client.post(
        "/api/infer" + qs, files={"image": ("f.jpg", _face_jpeg(), "image/jpeg")}
    )


def test_calibrated_off_by_default(monkeypatch):
    import main

    monkeypatch.setattr(main, "detect_faces", _box)
    j = (
        TestClient(main.app)
        .post("/api/infer", files={"image": ("f.jpg", _face_jpeg(), "image/jpeg")})
        .json()
    )
    assert j["face"]["expression"].get("calibrated", False) is False
    assert "probs_raw" not in j["face"]["expression"]


def test_calibrated_requires_validated_combo(monkeypatch):
    import main

    monkeypatch.setattr(main, "detect_faces", _box)
    c = TestClient(main.app)
    bad = c.post(
        "/api/infer?calibrated=true&profile=legacy",
        files={"image": ("f.jpg", _face_jpeg(), "image/jpeg")},
    )
    assert bad.status_code == 422
    ok = _post(c, "?calibrated=true").json()
    ex = ok["face"]["expression"]
    assert ex["calibrated"] is True and ex["temperature"] == 1.045
    assert abs(sum(ex["probs"].values()) - 1.0) < 1e-6
    assert abs(sum(ex["probs_raw"].values()) - 1.0) < 1e-6


def test_abstain_threshold_allowlist_and_flag(monkeypatch):
    import main

    monkeypatch.setattr(main, "detect_faces", _box)
    c = TestClient(main.app)
    assert _post(c, "?abstain=0.63").status_code == 422
    j = _post(c, "?abstain=0.95").json()
    assert j["face"]["abstained"] in (True, False)
    assert j["face"]["abstain_threshold"] == 0.95
    if j["face"]["abstained"]:
        assert (
            "below-abstention-threshold"
            in j["face"]["insufficient_evidence"]["reasons"]
        )
