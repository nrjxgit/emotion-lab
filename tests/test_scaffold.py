import pathlib
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_config_loads():
    import sys

    sys.path.insert(0, str(ROOT / "training"))
    from dataset_inventory import load_config

    cfg = load_config(str(ROOT / "configs" / "default.yaml"))
    assert "model" in cfg and "fusion" in cfg


def test_catalog_has_entries():
    import sys

    sys.path.insert(0, str(ROOT / "training"))
    from dataset_inventory import DATASET_CATALOG

    assert len(DATASET_CATALOG) >= 10
    for d in DATASET_CATALOG:
        for k in ("name", "license", "access", "labels", "notes"):
            assert k in d, f"missing {k} in {d.get('name')}"
