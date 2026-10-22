import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "eval_data" / "fer2013"


def test_loader_maps_labels(tmp_path):
    import numpy as np, cv2, sys

    sys.path.insert(0, str(ROOT / "training"))
    from fer_images import CLASSES7, load_split

    d = tmp_path / "test" / "happy"
    d.mkdir(parents=True)
    cv2.imwrite(str(d / "a.jpg"), np.full((48, 48), 200, np.uint8))
    X, y, files = load_split(str(tmp_path / "test"))
    assert X.shape == (1, 48, 48) and list(y) == [CLASSES7.index("happy")]
    assert files[0].endswith("a.jpg")


def test_splits_disjoint_by_filename():
    import sys

    sys.path.insert(0, str(ROOT / "training"))
    from fer_images import assert_disjoint

    assert_disjoint(["a.jpg", "b.jpg"], ["c.jpg"])
    import pytest

    with pytest.raises(ValueError):
        assert_disjoint(["a.jpg"], ["a.jpg"])
