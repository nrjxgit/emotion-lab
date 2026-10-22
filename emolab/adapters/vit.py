"""ViT adapter wrapping the pinned dima806 checkpoint via the backend adapter.

Weights: dima806/facial_emotions_image_detection, revision
747cf16692eea925b54b7b543cf436848128b68d, Apache-2.0, base
google/vit-base-patch16-224-in21k. Preprocessing: RGB 224x224 bilinear,
/255, mean/std 0.5 ([-1, 1]) via the HuggingFace image processor.

CHANGE: the legacy training.evaluate_models FER ViT route now requires an
explicit pinned local directory; ViTAdapter() fails closed without one.
No hub/network fallback is available.
"""

from __future__ import annotations

import time
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from emolab.inference import FaceEmotionAdapter, ModelMetadata, ModelResult

FER7 = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

VIT_REVISION = "747cf16692eea925b54b7b543cf436848128b68d"


class ViTAdapter(FaceEmotionAdapter):
    """ViT image-classification head mapped to FER7."""

    name = "vit"
    version = "1.0"
    input_shape = (224, 224, 3)
    native_labels = tuple(FER7)
    source = "dima806/facial_emotions_image_detection"
    licence = "Apache-2.0"
    calibrated = False

    def __init__(
        self,
        model_id: str = "dima806/facial_emotions_image_detection",
        id2label: tuple[str, ...] | None = None,
        forward_fn: Any = None,
        revision: str = VIT_REVISION,
        *, local_path: str | None = None, asset_sha256: dict[str, str] | None = None,
    ) -> None:
        self.local_path = Path(local_path) if local_path is not None else None
        self.asset_sha256 = dict(asset_sha256 or {})
        self._verified_assets = {}
        self.model_id = model_id
        self.revision = revision
        self._id2label = (
            tuple(str(id2label[i]).lower() for i in range(len(id2label)))
            if id2label is not None
            else None
        )
        self._forward_fn = forward_fn
        self._proc = None
        self._model = None

    def _labels(self, labels):
        aliases={"anger":"angry","happiness":"happy","sadness":"sad"}
        labels=tuple(aliases.get(str(x).lower(),str(x).lower()) for x in labels)
        if len(labels)!=7 or len(set(labels))!=7 or set(labels)!=set(FER7):
            raise ValueError("ViT requires exactly the seven named FER7 labels")
        return labels

    def _ensure_loaded(self) -> None:
        if self._forward_fn is not None:
            if self._id2label is None:raise ValueError("explicit fixture labels required")
            self._id2label=self._labels(self._id2label)
            return
        from training.cremad_development import safe_path
        if self.local_path is None:raise ValueError("explicit pinned local ViT directory required; no network fallback")
        root=safe_path(self.local_path)
        required={"config.json","preprocessor_config.json","model.safetensors"}
        if set(self.asset_sha256)!=required or {p.name for p in root.iterdir()}!=required:
            raise ValueError("exact local model/config/processor pin set required")
        for name in sorted(required):
            expected=self.asset_sha256[name]
            if len(expected)!=64 or any(c not in "0123456789abcdef" for c in expected):raise ValueError("full authoritative SHA256 required")
            path=safe_path(root/name);h=hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda:f.read(1048576),b""):h.update(chunk)
            if h.hexdigest()!=expected:raise ValueError("ViT asset checksum mismatch")
            self._verified_assets[name]=expected
        cfg=json.loads((root/"preprocessor_config.json").read_bytes())
        size=cfg.get("size")
        if not (size==224 or size=={"height":224,"width":224}) or cfg.get("resample")!=2 or cfg.get("do_resize") is not True or cfg.get("do_rescale") is not True or not np.isclose(cfg.get("rescale_factor",0),1/255) or cfg.get("do_normalize") is not True or cfg.get("image_mean")!=[.5]*3 or cfg.get("image_std")!=[.5]*3:
            raise ValueError("processor does not match preregistered RGB/bilinear/224/normalization")
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            import torch
        except ImportError as exc:
            raise ImportError(f"ViT needs preinstalled offline torch+transformers: {exc}") from exc
        proc=AutoImageProcessor.from_pretrained(str(root),local_files_only=True,trust_remote_code=False)
        model=AutoModelForImageClassification.from_pretrained(str(root),local_files_only=True,trust_remote_code=False,use_safetensors=True).to("cpu").eval()
        labels=self._labels(tuple(model.config.id2label[i] for i in range(len(model.config.id2label))))
        def forward(rgb):
            with torch.inference_mode():
                inputs={k:v.to("cpu") for k,v in proc(images=rgb,return_tensors="pt").items()}
                logits=model(**inputs).logits[0]
                return torch.softmax(logits,-1).detach().cpu().numpy()
        self._proc,self._model,self._id2label,self._forward_fn=proc,model,labels,forward

    def predict_proba(self, face_rgb: np.ndarray) -> ModelResult:
        self._ensure_loaded()
        import cv2

        t0 = time.perf_counter()
        rgb = cv2.cvtColor(face_rgb, cv2.COLOR_BGR2RGB)
        p_native = np.asarray(self._forward_fn(rgb), float)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if p_native.shape != (7,) or not np.isfinite(p_native).all() or (p_native < 0).any() or not np.isclose(p_native.sum(), 1, atol=1e-6, rtol=0):
            raise ValueError("invalid ViT probability vector; no uniform fallback")
        out = np.zeros(len(FER7), dtype=float)
        for hi, lab in enumerate(self._id2label):
            if lab in FER7:
                out[FER7.index(lab)] += float(p_native[hi])
        total = float(out.sum())
        if total <= 0:
            raise ValueError("zero ViT mass; no uniform fallback")
        else:
            out = out / total

        probs = {name: float(v) for name, v in zip(FER7, out)}
        top1_idx = int(np.argmax(out))
        return ModelResult(
            probabilities=probs,
            top1=FER7[top1_idx],
            confidence=float(out[top1_idx]),
            uncertainty=float(1.0 - out[top1_idx]),
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
                    "revision": self.revision,
                    "weights": str(self.local_path) if self.local_path else "SYNTHETIC forward fixture",
                    "verified_assets": dict(self._verified_assets),
                    "device": "cpu",
                },
            ),
        )
