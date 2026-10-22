import sys, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "app" / "face"))
sys.path.insert(0, str(ROOT / "training"))


def test_register_adapter_roundtrip():
    from adapters import FaceEmotionAdapter, get_adapter, register_adapter, FER7

    class Dummy(FaceEmotionAdapter):
        name = "dummy-test"
        native_labels = list(FER7)
        source = "test-double"

        def predict_proba(self, face_bgr):
            return np.full(7, 1 / 7)

    register_adapter("dummy-test", Dummy)
    assert isinstance(get_adapter("dummy-test"), Dummy)


def test_evaluate_adapter_reports_timing_and_abstention():
    from evaluate_adapter import evaluate_adapter
    from adapters import FaceEmotionAdapter, FER7

    class Dummy(FaceEmotionAdapter):
        name = "dummy-eval"
        native_labels = list(FER7)
        source = "test-double"

        def predict_proba(self, face_bgr):
            p = np.zeros(7)
            p[3] = 1.0
            return p

    X = np.zeros((4, 48, 48), np.uint8)
    r = evaluate_adapter(Dummy(), X, np.array([3, 3, 0, 3]))
    assert r["n"] == 4 and r["accuracy"] == 0.75
    assert r["ms_per_image"] >= 0 and "abstention" in r and "ece" in r


def test_dup_flagging_by_pixel_hash():
    from analyze_overlap import flag_test_rows_in_train

    assert flag_test_rows_in_train(["aa", "bb"], {"bb", "cc"}).tolist() == [False, True]
