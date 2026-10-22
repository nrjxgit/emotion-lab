"""Landmarks from YuNet 5 points (real) with geometric brow/chin completion."""

import numpy as np


def get_landmarks(img: np.ndarray, box: dict) -> dict:
    W, H = img.shape[1], img.shape[0]
    pts = {}
    if "landmarks5" in box:
        L = box["landmarks5"]
        ex_r = np.array(L["eye_R"]) / [W, H]
        ex_l = np.array(L["eye_L"]) / [W, H]
        nose = np.array(L["nose"]) / [W, H]
        mo_r = np.array(L["mouth_R"]) / [W, H]
        mo_l = np.array(L["mouth_L"]) / [W, H]
        eye_vec = ex_l - ex_r
        brow_off = np.array([0.0, -0.045])
        pts = {
            "brow_inner_L": tuple(ex_l * 0.6 + ex_r * 0.4 + brow_off),
            "brow_inner_R": tuple(ex_r * 0.6 + ex_l * 0.4 + brow_off),
            "brow_outer_L": tuple(ex_l + np.array([-0.03, -0.01])),
            "brow_outer_R": tuple(ex_r + np.array([0.03, -0.01])),
            "eye_top_L": tuple(ex_l + [0, -0.008]),
            "eye_bot_L": tuple(ex_l + [0, 0.008]),
            "eye_top_R": tuple(ex_r + [0, -0.008]),
            "eye_bot_R": tuple(ex_r + [0, 0.008]),
            "nose_tip": tuple(nose),
            "nose_L": tuple(nose + [-0.028, 0]),
            "nose_R": tuple(nose + [0.028, 0]),
            "mouth_L": tuple(mo_l),
            "mouth_R": tuple(mo_r),
            "mouth_top": tuple((mo_l + mo_r) / 2 + [0, -0.012]),
            "mouth_bot": tuple((mo_l + mo_r) / 2 + [0, 0.012]),
            "chin": (
                float((mo_l[0] + mo_r[0]) / 2),
                float(min(0.98, (mo_l[1] + mo_r[1]) / 2 + 0.16)),
            ),
            "cheek_L": (float(box["x"] / W + 0.02), float(nose[1])),
            "cheek_R": (float((box["x"] + box["w"]) / W - 0.02), float(nose[1])),
        }
        pts["_source"] = "yunet-5pt"
        return pts
    # MediaPipe path (optional dep)
    try:
        import mediapipe as mp

        raise RuntimeError("mediapipe path not implemented without box mesh crop")
    except Exception:
        raise RuntimeError("no real landmarks available for this face box")
