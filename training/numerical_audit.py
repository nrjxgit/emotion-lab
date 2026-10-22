"""Independent numerical audit of the numpy mini-Xception (no TF needed)."""

import json
import os
import sys

import h5py
import numpy as np

BN_EPS = 0.001
HDF5 = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "fer_mini_xception_h5.hdf5")
)
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "backend", "app", "face")
    ),
)


def _expected_shapes():
    return [
        ("conv2d_1", (62, 62, 8)),
        ("conv2d_2", (60, 60, 8)),
        ("add_1", (30, 30, 16)),
        ("add_2", (15, 15, 32)),
        ("add_3", (8, 8, 64)),
        ("add_4", (4, 4, 128)),
        ("conv2d_7", (4, 4, 7)),
        ("predictions", (7,)),
    ]


def audit_shapes() -> dict:
    import minixception_numpy as M
    from fer_model import forward, load_fer, preprocess

    w = load_fer(HDF5)
    x = preprocess(np.full((80, 80, 3), 200, dtype=np.uint8))

    def bn(arr, name):
        p = w[name]
        return M.batchnorm(
            arr, p["gamma"], p["beta"], p["moving_mean"], p["moving_variance"]
        )

    got = {}
    c1 = M.conv2d(x, w["conv2d_1"]["kernel"], padding="valid")
    got["conv2d_1"] = c1.shape
    c2 = M.conv2d(
        M.relu(bn(c1, "batch_normalization_1")),
        w["conv2d_2"]["kernel"],
        padding="valid",
    )
    got["conv2d_2"] = c2.shape
    got["predictions"] = forward(x, w).shape
    got["add_1"] = M.maxpool_same_s2(np.zeros((60, 60, 16), np.float32)).shape
    got["add_2"] = M.maxpool_same_s2(np.zeros((30, 30, 32), np.float32)).shape
    got["add_3"] = M.maxpool_same_s2(np.zeros((15, 15, 64), np.float32)).shape
    got["add_4"] = M.maxpool_same_s2(np.zeros((8, 8, 128), np.float32)).shape
    got["conv2d_7"] = M.conv2d(
        np.zeros((4, 4, 128), np.float32),
        w["conv2d_7"]["kernel"],
        padding="same",
        bias=w["conv2d_7"]["bias"],
    ).shape

    mismatches = []
    checked = 0
    for name, exp in _expected_shapes():
        if tuple(got[name]) != exp:
            mismatches.append((name, tuple(got[name]), exp))
        checked += 1

    f = h5py.File(HDF5, "r")
    cfg = json.loads(f.attrs["model_config"])
    eps = {
        layer["name"]: layer["config"].get("epsilon")
        for layer in cfg["config"]["layers"]
        if layer["class_name"] == "BatchNormalization"
    }
    f.close()
    bad = {k: v for k, v in eps.items() if abs(v - BN_EPS) > 1e-9}
    if bad:
        mismatches.append(("bn_eps", bad, BN_EPS))
    return {"mismatches": mismatches, "layers_checked": checked, "bn_layers": len(eps)}


def audit_determinism() -> dict:
    from fer_model import load_fer, preprocess, predict_fer

    w = load_fer(HDF5)
    rng = np.random.default_rng(1)
    img = (rng.random((90, 90, 3)) * 255).astype(np.uint8)
    x = preprocess(img)
    p1 = predict_fer(x, w)
    p2 = predict_fer(x, w)
    return {"max_abs_diff": float(np.max(np.abs(p1 - p2))), "prob_sum": float(p1.sum())}
