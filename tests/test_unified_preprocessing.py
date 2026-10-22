import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "eval_data" / "fer2013"
NEED = ROOT / "eval_results" / "v0.5" / "crops" / "test_detections.json"

requires_data = pytest.mark.skipif(
    not (DATA.exists() and NEED.exists()),
    reason="needs local FER images + committed study manifest (gitignored data)",
)


@requires_data
def test_serving_tensor_matches_study_tensor():
    sys.path.insert(0, str(ROOT / "backend" / "app"))
    sys.path.insert(0, str(ROOT / "training"))
    import cv2
    from face.preprocessing import serving_crop_tensor
    from crop_science import preprocess_crop

    det = json.loads(NEED.read_text())
    ids = det["ids"]
    boxes = det["boxes"]
    checked = 0
    for i, (rid, box) in enumerate(zip(ids, boxes)):
        if box is None or checked >= 5:
            continue
        img = cv2.imread(str(DATA / rid), cv2.IMREAD_GRAYSCALE)
        big = cv2.resize(img, (192, 192), interpolation=cv2.INTER_LINEAR)
        a = serving_crop_tensor(big, box)
        b = preprocess_crop(big, box, "margin25", source=img)
        assert a.shape == (64, 64, 1) and b.shape == (64, 64, 1)
        assert np.array_equal(a, b), f"serving/study tensor mismatch on {rid}"
        checked += 1
    assert checked == 5


@requires_data
def test_serving_recomputes_committed_margin25_proba():
    sys.path.insert(0, str(ROOT / "backend" / "app"))
    from face.preprocessing import serving_crop_tensor
    from face.fer_model import load_fer, predict_fer

    ref = np.load(ROOT / "eval_results" / "v0.5" / "crops" / "test_margin25.npz")
    det = json.loads(NEED.read_text())
    import cv2

    w = load_fer()
    n = 0
    for rid, box, y, p_ref in zip(
        det["ids"][:60], det["boxes"][:60], ref["y"][:60], ref["proba"][:60]
    ):
        if box is None:
            continue
        img = cv2.imread(str(DATA / rid), cv2.IMREAD_GRAYSCALE)
        big = cv2.resize(img, (192, 192), interpolation=cv2.INTER_LINEAR)
        p = predict_fer(
            serving_crop_tensor(cv2.cvtColor(big, cv2.COLOR_GRAY2BGR), box), w
        )
        assert int(np.argmax(p)) == int(np.argmax(p_ref))
        n += 1
    assert n > 30
