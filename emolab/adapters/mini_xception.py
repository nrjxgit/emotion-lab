"""mini-Xception adapter wrapping the frozen FER2013 weights in this repo.

Weights: models/fer_mini_xception_h5.hdf5 (oarriaga face_classification, MIT),
the same frozen checkpoint used by the serving pipeline. Preprocessing matches
the validated serving profile: gray -> 64x64 INTER_LINEAR -> /255 -> [-1, 1].
"""

from __future__ import annotations

import time

import numpy as np

from emolab.inference import FaceEmotionAdapter, ModelMetadata, ModelResult

FER7 = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class MiniXceptionAdapter(FaceEmotionAdapter):
    """mini-Xception (oarriaga face_classification) on FER2013."""

    name = "mini-xception"
    version = "1.0"
    input_shape = (64, 64, 1)
    native_labels = tuple(FER7)
    source = "oarriaga face_classification"
    licence = "MIT"
    calibrated = False

    def __init__(self) -> None:
        self._w = None
        self._predict_fer = None

    def _load(self):
        if self._w is None:
            try:
                from backend.app.face.fer_model import load_fer, predict_fer
            except ImportError:  # allows import outside the repo layout
                from fer_model import load_fer, predict_fer  # type: ignore[no-redef]

            self._w = load_fer()
            self._predict_fer = predict_fer
        return self._w, self._predict_fer

    def predict_proba(self, face_rgb: np.ndarray) -> ModelResult:
        w, predict_fer = self._load()
        from backend.app.face.preprocessing import tensor_from_crop

        t0 = time.perf_counter()
        tensor = tensor_from_crop(face_rgb)
        p7 = [float(v) for v in predict_fer(tensor, w)]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        probs = {name: float(p) for name, p in zip(FER7, p7)}
        top1_idx = int(np.argmax(p7))
        return ModelResult(
            probabilities=probs,
            top1=FER7[top1_idx],
            confidence=float(p7[top1_idx]),
            uncertainty=float(1.0 - p7[top1_idx]),
            latency_ms=latency_ms,
            metadata=ModelMetadata(
                model_id=self.name,
                version=self.version,
                input_shape=self.input_shape,
                class_mapping={i: l for i, l in enumerate(FER7)},
                calibrated=self.calibrated,
                supported_classes=frozenset(FER7),
                provenance={
                    "source": self.source,
                    "licence": self.licence,
                    "weights": "models/fer_mini_xception_h5.hdf5",
                },
            ),
        )
