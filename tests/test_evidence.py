import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "app"))


def test_no_face_is_insufficient():
    from fusion.evidence import evidence_status

    r = evidence_status(face_found=False, quality=0.0, top_prob=0.0)
    assert "no-face-detected" in r["reasons"] and r["insufficient"] is True


def test_poor_quality_and_uncertainty_flagged():
    from fusion.evidence import evidence_status

    r = evidence_status(face_found=True, quality=0.2, top_prob=0.9)
    assert "poor-detection-quality" in r["reasons"]
    r = evidence_status(face_found=True, quality=0.9, top_prob=0.3)
    assert "high-uncertainty" in r["reasons"]


def test_clean_reading_has_no_reasons():
    from fusion.evidence import evidence_status

    r = evidence_status(face_found=True, quality=0.9, top_prob=0.8)
    assert r["reasons"] == [] and r["insufficient"] is False
