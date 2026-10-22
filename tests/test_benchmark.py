import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))


def test_benchmark_labels_and_shapes():
    from benchmark_fer import benchmark

    r = benchmark(
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        proba=[[0.7, 0.3], [0.4, 0.6], [0.5, 0.5], [0.8, 0.2]],
        dataset="toy",
        n_classes=2,
    )
    assert r["dataset"] == "toy" and r["n"] == 4
    assert 0.0 <= r["accuracy"] <= 1.0 and 0.0 <= r["macroF1"] <= 1.0
    assert "per_class_recall" in r and "confusion" in r and "ece" in r


def test_benchmark_needs_labels():
    from benchmark_fer import benchmark
    import pytest

    with pytest.raises(ValueError):
        benchmark([], [], dataset="empty")
