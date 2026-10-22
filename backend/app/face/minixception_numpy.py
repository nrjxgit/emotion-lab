"""Numpy inference ops for mini-Xception (TF SAME/VALID semantics, channels-last)."""

import numpy as np


def _pad(x, pt, pb, pl, pr, value=0.0):
    if pt == pb == pl == pr == 0:
        return x
    return np.pad(
        x, ((pt, pb), (pl, pr), (0, 0)), mode="constant", constant_values=value
    )


def _im2col(x, kh, kw, sy, sx, pt, pb, pl, pr, pad_value=0.0):
    x = _pad(x, pt, pb, pl, pr, pad_value)
    H, W, C = x.shape
    Ho, Wo = (H - kh) // sy + 1, (W - kw) // sx + 1
    s0, s1, s2 = x.strides
    cols = np.lib.stride_tricks.as_strided(
        x, shape=(Ho, Wo, kh, kw, C), strides=(s0 * sy, s1 * sx, s0, s1, s2)
    )
    return cols.reshape(Ho * Wo, kh * kw * C), Ho, Wo


def conv2d(x, K, stride=1, padding="valid", bias=None):
    kh, kw, Cin, Cout = K.shape
    assert x.shape[2] == Cin
    if padding == "valid":
        pt = pb = pl = pr = 0
    else:  # same stride 1, odd k
        pt = pb = pl = pr = kh // 2
    cols, Ho, Wo = _im2col(x, kh, kw, stride, stride, pt, pb, pl, pr)
    out = cols @ K.reshape(-1, Cout)
    if bias is not None:
        out = out + bias
    return out.reshape(Ho, Wo, Cout)


def conv1x1_s2(x, K):
    return np.tensordot(x[::2, ::2, :], K[0, 0, :, :], axes=([2], [0]))


def sepconv_same(x, Dw, Pw):
    kh, kw, Cin, _m = Dw.shape
    assert _m == 1 and x.shape[2] == Cin
    xp = _pad(x, kh // 2, kh // 2, kw // 2, kw // 2)
    H, W, _ = x.shape
    s0, s1, s2 = xp.strides
    win = np.lib.stride_tricks.as_strided(
        xp, shape=(H, W, kh, kw, Cin), strides=(s0, s1, s0, s1, s2)
    )
    dw = np.einsum("hwijc,ijc->hwc", win, Dw[:, :, :, 0])
    return np.tensordot(dw, Pw[0, 0, :, :], axes=([2], [0]))


def batchnorm(x, gamma, beta, mean, var, eps=0.001):
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def relu(x):
    return np.maximum(x, 0)


def maxpool_same_s2(x):
    H, W, C = x.shape
    Ho = (H + 1) // 2
    pad_total_h = max((Ho - 1) * 2 + 3 - H, 0)
    pad_total_w = (
        max((Ho - 1) * 2 + 3 - W, 0)
        if H == W
        else max((((W + 1) // 2 - 1) * 2 + 3 - W), 0)
    )
    pt, pb = pad_total_h // 2, pad_total_h - pad_total_h // 2
    pl, pr = pad_total_w // 2, pad_total_w - pad_total_w // 2
    cols, Ho2, Wo2 = _im2col(
        x, 3, 3, 2, 2, pt, pb, pl, pr, pad_value=-np.inf
    )  # TF pads pooled regions with -inf
    cols = cols.reshape(Ho2 * Wo2, 9, C)
    return cols.max(axis=1).reshape(Ho2, Wo2, C)


def global_avg_pool(x):
    return x.mean(axis=(0, 1))


def softmax(v):
    v = np.asarray(v, float)
    v = v - v.max()
    e = np.exp(v)
    return (e / e.sum()).astype(np.float64)
