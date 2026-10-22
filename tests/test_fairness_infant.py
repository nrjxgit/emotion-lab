import sys, pathlib, importlib.util

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_fair = _load(
    "fairness_evaluate", str(ROOT / "backend" / "app" / "fairness" / "evaluate.py")
)
_infant = _load(
    "infant_pipeline", str(ROOT / "backend" / "app" / "infant" / "pipeline.py")
)


def test_stratified_report():
    r = _fair.stratified_report([0, 1, 0, 1], [0, 1, 1, 0], ["a", "a", "b", "b"])
    assert "macroF1" in r and 0 <= r["macroF1"] <= 1


def test_infant_experimental_flag():
    r = _infant.infant_predict({}, [])
    assert r.get("experimental") is True
