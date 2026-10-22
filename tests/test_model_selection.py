import pathlib
import sys
import cv2
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend/app"))


def test_model_listing_does_not_replace_default():
    from main import app

    data = TestClient(app).get("/api/models").json()
    assert data["default"] == "mini-xception"
    assert data["models"]["vit"]["experimental"] is True
    assert data["models"]["hsemotion"]["available"] is False


def test_unknown_requested_model_cannot_silently_fall_back():
    from main import app

    _, data = cv2.imencode(".jpg", np.zeros((240, 320, 3), np.uint8))
    r = TestClient(app).post(
        "/api/infer?model=not-a-model",
        files={"image": ("f.jpg", data.tobytes(), "image/jpeg")},
    )
    assert r.status_code == 422


def test_explicit_default_is_equivalent_to_omitting_selection():
    from main import app

    client = TestClient(app)
    _, data = cv2.imencode(".jpg", np.zeros((240, 320, 3), np.uint8))
    payload = {"image": ("f.jpg", data.tobytes(), "image/jpeg")}
    a = client.post("/api/infer", files=payload).json()
    b = client.post("/api/infer?model=mini-xception", files=payload).json()
    assert a["face"]["expression"] == b["face"]["expression"]
    assert b["selected_model"] == "mini-xception"
