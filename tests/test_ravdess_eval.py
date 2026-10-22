import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))


def test_filename_decode():
    from eval_ravdess import decode_ravdess, RAVDESS_MAP

    assert decode_ravdess("03-01-05-01-02-01-12.mp4") == ("angry", 12)
    assert decode_ravdess("03-01-07-01-02-01-12.mp4") == ("disgust", 12)
    assert decode_ravdess("03-01-03-01-01-01-01.mp4") == ("happy", 1)
    assert set(RAVDESS_MAP) == {
        "neutral",
        "calm",
        "happy",
        "sad",
        "angry",
        "fearful",
        "disgust",
        "surprised",
    }


def test_label_map_with_calm_flag():
    from eval_ravdess import to_fer7

    assert to_fer7("calm", include_calm=True) == "neutral"
    assert to_fer7("calm", include_calm=False) is None
    assert to_fer7("fearful", include_calm=True) == "fear"


def test_abstention_accounting():
    from eval_ravdess import summarize

    rows = [{"y": 0, "pred": 0, "face": True}, {"y": 1, "pred": None, "face": False}]
    s = summarize(rows, n_classes=2)
    assert s["n"] == 2 and s["abstentions"] == 1
    assert s["metrics"]["dataset"].startswith("ravdess")
