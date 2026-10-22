"""Genuine FER2013 mini-Xception inference from oarriaga hdf5 weights (no TF needed).

Architecture + hyperparams read from the hdf5 model_config; class order is the
FER2013/oarriaga order: angry, disgust, fear(sacred->fear), happy, sad, surprise, neutral.
Legacy default preserved: gray face, resize 64x64, /255. IMPORTANT: the
upstream oarriaga v2 pipeline additionally maps this to [-1,1]. See the
explicit research profiles in training/crop_science.py and v0.5 report.
Forward-pass parity on equal tensors does not establish preprocessing parity.
"""

import os
import h5py
import numpy as np
import cv2

try:
    from .minixception_numpy import (
        conv2d,
        conv1x1_s2,
        sepconv_same,
        batchnorm,
        relu,
        maxpool_same_s2,
        global_avg_pool,
        softmax,
    )
except ImportError:
    from minixception_numpy import (
        conv2d,
        conv1x1_s2,
        sepconv_same,
        batchnorm,
        relu,
        maxpool_same_s2,
        global_avg_pool,
        softmax,
    )

FER7 = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
HDF5 = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "models",
        "fer_mini_xception_h5.hdf5",
    )
)


def weights_available() -> bool:
    return os.path.exists(HDF5) and os.path.getsize(HDF5) > 100000


def load_fer(path: str = HDF5) -> dict:
    f = h5py.File(path, "r")
    mw = f["model_weights"]
    out = {}
    for layer in mw.keys():
        g = mw[layer]
        inner_names = list(g.keys())
        if not inner_names:
            continue
        inner = g[inner_names[0]]
        if not hasattr(inner, "keys"):
            continue
        params = {}
        for ds in inner.keys():
            o = inner[ds]
            if hasattr(o, "shape"):
                params[ds.split(":")[0]] = np.asarray(o, dtype=np.float32)
        out[layer] = params
    f.close()
    return out


def preprocess(
    face_bgr: np.ndarray, v2: bool = False, interpolation=None
) -> np.ndarray:
    if interpolation is None:
        interpolation = cv2.INTER_AREA
    gray = (
        cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if face_bgr.ndim == 3 else face_bgr
    )
    small = cv2.resize(gray, (64, 64), interpolation=interpolation)
    x = small.astype(np.float32) / 255.0
    if v2:  # upstream oarriaga v2 pipeline; op order matches study (*2-1, not (x-.5)*2)
        x = x * 2.0 - 1.0
    return x[:, :, None]


def _bn(w, name):
    p = w[name]
    return p["gamma"], p["beta"], p["moving_mean"], p["moving_variance"]


def forward(x: np.ndarray, w: dict) -> np.ndarray:
    x = relu(
        batchnorm(
            conv2d(x, w["conv2d_1"]["kernel"], padding="valid"),
            *_bn(w, "batch_normalization_1"),
        )
    )
    a = relu(
        batchnorm(
            conv2d(x, w["conv2d_2"]["kernel"], padding="valid"),
            *_bn(w, "batch_normalization_2"),
        )
    )

    def block(inp, s1, b1, s2, b2, shortcut, bs):
        m = sepconv_same(inp, w[s1]["depthwise_kernel"], w[s1]["pointwise_kernel"])
        m = relu(batchnorm(m, *_bn(w, b1)))
        m = sepconv_same(m, w[s2]["depthwise_kernel"], w[s2]["pointwise_kernel"])
        m = batchnorm(m, *_bn(w, b2))
        s = batchnorm(conv1x1_s2(inp, w[shortcut]["kernel"]), *_bn(w, bs))
        return maxpool_same_s2(m) + s

    # NOTE: maxpool branch pools m, shortcut branch is strided conv; both halve.
    # block() above pools only m; shortcut conv1x1_s2 already halves. Fix: pool m then add.
    x = block(
        a,
        "separable_conv2d_1",
        "batch_normalization_4",
        "separable_conv2d_2",
        "batch_normalization_5",
        "conv2d_3",
        "batch_normalization_3",
    )
    x = block(
        x,
        "separable_conv2d_3",
        "batch_normalization_7",
        "separable_conv2d_4",
        "batch_normalization_8",
        "conv2d_4",
        "batch_normalization_6",
    )
    x = block(
        x,
        "separable_conv2d_5",
        "batch_normalization_10",
        "separable_conv2d_6",
        "batch_normalization_11",
        "conv2d_5",
        "batch_normalization_9",
    )
    x = block(
        x,
        "separable_conv2d_7",
        "batch_normalization_13",
        "separable_conv2d_8",
        "batch_normalization_14",
        "conv2d_6",
        "batch_normalization_12",
    )
    logits = global_avg_pool(
        conv2d(x, w["conv2d_7"]["kernel"], padding="same", bias=w["conv2d_7"]["bias"])
    )
    return softmax(logits)


def predict_fer(x: np.ndarray, w: dict) -> np.ndarray:
    return forward(x, w)
