import sys, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))


def test_paired_agreement_counts():
    from compare_clips import paired_stats

    r = paired_stats([0, 1, 1, 0], [0, 0, 1, 1], [0, 1, 0, 1])
    assert r["n"] == 4 and r["agreement"] == 0.5
    assert r["both_correct"] == 1 and r["both_wrong"] == 1


def test_actor_split_disjoint_and_labelled():
    from compare_clips import actor_split

    actors = ["a1", "a2", "a3", "a4"]
    cal, tst = actor_split(actors, n_cal=2, seed=505)
    assert set(cal).isdisjoint(set(tst)) and len(cal) == 2 and len(tst) == 2
    cal2, _ = actor_split(actors, n_cal=2, seed=505)
    assert cal == cal2


def test_clip_manifest_unique_and_eval_only():
    from compare_clips import audit_clips

    clips = [
        {"file": "1001_DFA_ANG_XX.flv", "actor": "1001"},
        {"file": "1001_DFA_ANG_XX.flv", "actor": "1001"},
    ]
    r = audit_clips(clips, train_files=set(), selected_actors={"1001"})
    assert r["duplicates"] == ["1001_DFA_ANG_XX.flv"] and r["n_unique"] == 1
