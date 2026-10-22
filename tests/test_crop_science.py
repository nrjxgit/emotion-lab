import numpy as np
import pytest


def test_crop_clamps_original_edges_without_shifting_box():
    from training.crop_science import crop_image

    im = np.arange(100, dtype=np.uint8).reshape(10, 10)
    b = dict(x=-3, y=-2, w=8, h=7)
    assert np.array_equal(crop_image(im, b), im[:5, :5])
    with pytest.raises(ValueError):
        crop_image(im, dict(x=11, y=2, w=3, h=3))


def test_margin_and_square_geometry():
    from training.crop_science import crop_image

    im = np.ones((100, 100), np.uint8)
    box = dict(x=30, y=40, w=20, h=10)
    assert crop_image(im, box, margin=0.1).shape == (12, 24)
    assert crop_image(im, box, square=True).shape == (20, 20)


def test_alignment_levels_eyes_without_requiring_generated_landmarks():
    from training.crop_science import align_eyes

    im = np.ones((100, 100), np.uint8)
    box = dict(
        x=20, y=20, w=60, h=60, landmarks5={"eye_L": [35, 40], "eye_R": [65, 60]}
    )
    aligned, matrix = align_eyes(im, box)
    eyes = np.array([[35, 40, 1], [65, 60, 1]]) @ matrix.T
    assert aligned.shape == im.shape
    assert abs(eyes[0, 1] - eyes[1, 1]) < 1e-6


def test_profiles_keep_legacy_and_upstream_ranges_distinct():
    from training.crop_science import preprocess_crop

    black = np.zeros((48, 48), np.uint8)
    box = dict(x=0, y=0, w=48, h=48)
    assert preprocess_crop(black, box, "tight_legacy").shape == (64, 64, 1)
    assert np.all(preprocess_crop(black, box, "tight_legacy") == 0)
    assert np.all(preprocess_crop(black, box, "tight_linear") == -1)


def test_detection_misses_are_false_negatives_not_silently_dropped():
    from training.crop_science import metrics

    p = np.zeros((3, 7))
    p[0, 0] = p[1, 1] = p[2, 2] = 1
    m = metrics([0, 1, 2], p, [True, True, False])
    assert m["conditional"]["accuracy"] == 1
    assert m["end_to_end"]["accuracy"] == pytest.approx(2 / 3)
    assert m["end_to_end"]["recall"][2] == 0
    assert m["end_to_end"]["confusion"][2][7] == 1


def test_selection_cannot_consume_test_scores():
    from training.crop_science import select_validation_winner

    scores = {
        "a": {"conditional": {"macro_f1": 0.5}},
        "b": {"conditional": {"macro_f1": 0.6}},
    }
    assert select_validation_winner(scores, "validation") == "b"
    with pytest.raises(ValueError, match="validation"):
        select_validation_winner(scores, "test")


def test_ece_missing_or_all_abstained_is_not_fabricated_zero():
    from training.crop_science import metrics

    m = metrics([0, 1], np.zeros((2, 7)), [False, False])
    assert m["conditional"]["ece"] is None
    assert m["end_to_end"]["accuracy"] == 0
