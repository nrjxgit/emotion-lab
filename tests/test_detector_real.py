import sys, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "app" / "face"))


def test_blank_yields_no_faces():
    from detector import detect_faces

    blank = np.full((240, 320, 3), 128, dtype=np.uint8)
    assert detect_faces(blank) == []


def test_never_synthesizes_boxes():
    from detector import detect_faces
    import inspect

    src = inspect.getsource(detect_faces)
    assert "w // 2" not in src and "center box" not in src.lower()


def test_corrupt_input_raises():
    from detector import detect_faces
    import pytest

    with pytest.raises((ValueError, TypeError, AttributeError)):
        detect_faces(None)
