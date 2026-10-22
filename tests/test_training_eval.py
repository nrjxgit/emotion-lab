import sys, pathlib, os, importlib.util

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_mlp = _load("training_mlp", str(ROOT / "training" / "train_mlp.py"))
_eval = _load("training_evaluate", str(ROOT / "training" / "evaluate.py"))
_csv = _load("training_fer_csv", str(ROOT / "training" / "fer_csv.py"))


def test_mlp_synthetic_smoke(tmp_path):
    out = _mlp.run_synthetic(str(tmp_path / "mlp.joblib"), n=60)
    assert os.path.exists(str(tmp_path / "mlp.joblib"))
    assert 0.0 <= out["train_acc"] <= 1.0
    assert out["synthetic"] is True  # never claimed as emotion training


def test_evaluate_metrics():
    r = _eval.evaluate([0, 1, 1, 0, 2, 2], [0, 1, 0, 0, 2, 1], n_classes=3)
    assert 0.0 <= r["macroF1"] <= 1.0
    assert r["confusion"].shape == (3, 3)
    assert 0.0 <= r["ece"] <= 1.0
    assert "per_class" in r


def test_fer_csv_missing_gives_typed_error():
    import pytest

    with pytest.raises(FileNotFoundError):
        _csv.load_fer_csv("nonexistent-fer2013.csv")
