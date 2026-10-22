"""Face-only vs face+body ablation on labeled rows.

answers: does adding body change top-1 accuracy? Runs on any labeled rows;
with no rows it raises instead of inventing scores.
"""

import numpy as np


def ablate(rows: list, n_classes: int = 7, w_face: float = 0.7) -> dict:
    if not rows:
        raise ValueError("no labeled rows: ablation needs face/body probs + labels")
    f_ok = b_ok = 0
    for r in rows:
        f = np.asarray(r["face"], float)
        b = np.asarray(r["body"], float)
        if f.argmax() == r["y"]:
            f_ok += 1
        if (w_face * f + (1 - w_face) * b).argmax() == r["y"]:
            b_ok += 1
    n = len(rows)
    return {
        "n": n,
        "face_only_acc": f_ok / n,
        "fusion_acc": b_ok / n,
        "delta": (b_ok - f_ok) / n,
        "w_face": w_face,
        "dataset_note": "labeled rows required",
        "caution": "delta on unlabeled/demo data is not evidence of improvement",
    }
