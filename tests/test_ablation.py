import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))


def test_ablation_reports_delta():
    from ablation import ablate

    rows = [
        {"face": [0.7, 0.3], "body": [0.4, 0.6], "y": 0},
        {"face": [0.2, 0.8], "body": [0.4, 0.6], "y": 1},
        {"face": [0.6, 0.4], "body": [0.9, 0.1], "y": 0},
    ]
    r = ablate(rows, n_classes=2)
    assert "face_only_acc" in r and "fusion_acc" in r and "delta" in r
    assert r["n"] == 3 and r["dataset_note"] == "labeled rows required"
