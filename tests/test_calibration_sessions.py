import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "app" / "calibration"))


def test_no_consent_refused():
    from sessions import CalibrationSession
    import pytest

    s = CalibrationSession()
    with pytest.raises(PermissionError):
        s.start(consent=False)


def test_record_and_agreement():
    from sessions import CalibrationSession

    s = CalibrationSession()
    s.start(consent=True, participant="anon-1")
    s.record(
        ai_label="happiness",
        self_report="happiness",
        probs={"happiness": 0.8, "neutral": 0.2},
    )
    s.record(
        ai_label="neutral",
        self_report="sadness",
        probs={"neutral": 0.5, "sadness": 0.4},
    )
    r = s.report()
    assert r["n"] == 2 and r["agreement_rate"] == 0.5
    assert "per_expression" in r and r["consent"] is True
