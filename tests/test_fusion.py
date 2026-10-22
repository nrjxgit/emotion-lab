import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "app" / "fusion"))
sys.path.insert(0, str(ROOT / "backend" / "app" / "calibration"))


def test_fusion_modes():
    from fuse import fuse

    face = {
        "probs": {"happiness": 0.8, "neutral": 0.1, "unknown": 0.1},
        "confidence": 0.8,
        "uncertainty": 0.2,
    }
    body = {
        "probs": {"happiness": 0.2, "neutral": 0.7, "unknown": 0.1},
        "confidence": 0.5,
        "uncertainty": 0.5,
    }
    r_both = fuse(face, body, {"face_q": 0.9, "body_q": 0.9})
    r_face = fuse(face, None, {"face_q": 0.9, "body_q": 0.0})
    r_none = fuse(None, None, {"face_q": 0.0, "body_q": 0.0})
    assert abs(sum(r_both["probs"].values()) - 1.0) < 1e-5
    assert r_face["mode"] == "face-only"
    assert r_none["probs"]["unknown"] > 0.5
    # correlated agreement must not inflate to >0.95 when quality low
    r_low = fuse(face, face, {"face_q": 0.2, "body_q": 0.2})
    assert r_low["confidence"] < 0.95
