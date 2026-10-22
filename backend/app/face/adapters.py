"""Modular face-emotion adapters: mini-Xception (default), ViT, HSEmotion-stub.

All adapters return FER7-ordered probabilities. Default model changes only via
docs/validation-gates.md criteria — never silently.
"""

import numpy as np

FER7 = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
DEFAULT_MODEL = "mini-xception"
VIT_REVISION = "747cf16692eea925b54b7b543cf436848128b68d"


class FaceEmotionAdapter:
    name = "base"
    native_labels = list(FER7)
    source = "base"

    def predict_proba(self, face_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MiniXceptionAdapter(FaceEmotionAdapter):
    name = "mini-xception"
    native_labels = list(FER7)
    source = "trained-minixception-fer2013"

    def __init__(self):
        self._w = None

    def _load(self):
        if self._w is None:
            try:
                from .fer_model import load_fer
            except ImportError:
                from fer_model import load_fer
            self._w = load_fer()
        return self._w

    def predict_proba(self, face_bgr: np.ndarray) -> np.ndarray:
        try:
            from .fer_model import preprocess, predict_fer
        except ImportError:
            from fer_model import preprocess, predict_fer
        return np.asarray(predict_fer(preprocess(face_bgr), self._load()), float)


class ViTAdapter(FaceEmotionAdapter):
    name = "vit"
    source = "hf-vit-dima806"

    def __init__(
        self,
        model_id: str = "dima806/facial_emotions_image_detection",
        id2label=None,
        forward_fn=None,
        revision=VIT_REVISION,
    ):
        self.model_id = model_id
        self.revision = revision
        self._id2label = (
            [str(id2label[i]).lower() for i in range(len(id2label))]
            if id2label is not None
            else None
        )
        self._forward_fn = forward_fn
        self._proc = self._model = None

    @property
    def native_labels(self):
        if self._id2label is None:
            self._ensure_loaded()
        return list(self._id2label)

    def _ensure_loaded(self):
        if self._forward_fn is not None and self._id2label is not None:
            return
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            import torch
        except ImportError as e:
            raise ImportError(f"ViT needs torch+transformers: {e}")
        self._proc = AutoImageProcessor.from_pretrained(
            self.model_id, revision=self.revision
        )
        self._model = AutoModelForImageClassification.from_pretrained(
            self.model_id, revision=self.revision
        ).eval()
        cfg = self._model.config
        self._id2label = [
            str(cfg.id2label[i]).lower() for i in range(len(cfg.id2label))
        ]
        import torch as _t

        def _fwd(rgb: np.ndarray) -> np.ndarray:
            with _t.no_grad():
                logits = self._model(
                    **self._proc(images=rgb, return_tensors="pt")
                ).logits[0]
                return _t.softmax(logits, -1).numpy()

        self._forward_fn = _fwd

    def predict_proba(self, face_bgr: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        import cv2

        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        p_native = np.asarray(self._forward_fn(rgb), float)
        out = np.zeros(7)
        for hi, lab in enumerate(self._id2label):
            out[FER7.index(lab)] += p_native[hi]
        s = out.sum()
        return out / s if s > 0 else np.full(7, 1 / 7)


class HSEmotionAdapter(FaceEmotionAdapter):
    name = "hsemotion"
    source = "hsemotion-separate-app"

    def predict_proba(self, face_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "HSEmotion lives in the separate Claude Code app; merge pending "
            "independent validation (see docs/validation-gates.md)."
        )


_REGISTRY = {}


def register_adapter(name: str, cls) -> None:
    if not (isinstance(cls, type) and issubclass(cls, FaceEmotionAdapter)):
        raise ValueError("adapter must subclass FaceEmotionAdapter")
    _REGISTRY[name] = cls


def get_adapter(name: str) -> FaceEmotionAdapter:
    table = {
        "mini-xception": MiniXceptionAdapter,
        "vit": ViTAdapter,
        "hsemotion": HSEmotionAdapter,
    }
    table.update(_REGISTRY)
    if name not in table:
        raise ValueError(f"unknown adapter {name!r}; choices {sorted(table)}")
    return table[name]()
