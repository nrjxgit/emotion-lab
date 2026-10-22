import numpy as np
import pytest


def test_intended_label_and_actor_do_not_confuse_sentence_or_intensity():
    from training.cremad_science import parse_filename

    assert parse_filename("1002_IEO_DIS_HI.flv") == ("1002", 1)
    with pytest.raises(ValueError):
        parse_filename("1002_IEO_SUR_HI.flv")


def test_visual_vote_ties_are_explicit_and_not_arbitrarily_resolved():
    from training.cremad_science import parse_face_vote

    assert parse_face_vote("D") == 1
    assert parse_face_vote("A:D") is None
    assert parse_face_vote("") is None


def test_surprise_prediction_is_a_mistake_not_removed_from_denominator():
    from training.cremad_science import score_six

    p = np.zeros((2, 7))
    p[0, 5] = 1
    p[1, 1] = 1
    s = score_six([0, 1], p, [True, True])
    assert s["accuracy"] == 0.5
    assert s["surprise_predictions"] == 1
    assert s["confusion"][0][5] == 1
    assert len(s["recall"]) == 6


def test_official_summary_csv_uses_capital_FileName():
    from training.cremad_science import read_summary

    text = '"","FileName","FaceVote"\n"1","1001_DFA_DIS_XX","A:D"\n'
    rows = read_summary(text)
    assert rows["1001_DFA_DIS_XX"]["FaceVote"] == "A:D"
