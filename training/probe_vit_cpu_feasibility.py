"""ViT CPU feasibility probe — bounded synthetic-input inference only.

OC-ADDITIONAL-20261019-A — ViT frozen-backbone/head-only CPU feasibility.

This probe:
  * Loads the pinned local ViT checkpoint (safetensors) via transformers
  * Runs ONE bounded synthetic-input CPU inference timing/memory probe
  * No backward/optimizer steps, no trained candidate, no images or labels
  * Whole probe (load + warm-up + forwards) has a hard 60s cap via parent-process
    subprocess timeout
  * Strictly offline/local-files-only: never fetches missing assets, never installs

EXIT CODES:
  0  = probe completed within cap (timings/memory printed)
  1  = dependency missing (transformers/safetensors not installed)
  2  = probe exceeded 60s hard cap (cancelled by parent)
  3  = other load/probe failure

ENVIRONMENT REQUIREMENTS:
  * torch (CPU), transformers, safetensors must be importable
  * Checkpoint: models/vit/747cf16692eea925b54b7b543cf436848128b68d/
"""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

CHECKPOINT_DIR = Path("models/vit/747cf16692eea925b54b7b543cf436848128b68d")
REVISION = "747cf16692eea925b54b7b543cf436848128b68d"
HARD_CAP_SECONDS = 60
NUM_FORWARDS = 5  # bounded number of forwards after warm-up


def _dep_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _timeout_handler(signum, frame):
    raise TimeoutError("probe exceeded 60s hard cap")


def run_probe() -> dict:
    result = {
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "revision": REVISION,
        "deps_available": {},
        "load_seconds": None,
        "probe_seconds": None,
        "total_seconds": None,
        "hard_cap_seconds": HARD_CAP_SECONDS,
        "num_forwards": NUM_FORWARDS,
        "forward_times_ms": [],
        "peak_rss_mb": None,
        "native_labels": None,
        "embedding_dim": None,
        "head_out_features": None,
        "trainable_params_total": None,
        "trainable_params_backbone": None,
        "trainable_params_head": None,
        "frozen_backbone": True,
        "trainable_head_only": False,
        "status": None,
        "error": None,
    }

    # --- Dependency check ---
    for dep in ["torch", "transformers", "safetensors"]:
        result["deps_available"][dep] = _dep_available(dep)

    missing = [d for d, ok in result["deps_available"].items() if not ok]
    if missing:
        result["status"] = "LIMITED/UNMEASURED"
        result["error"] = f"missing dependencies: {', '.join(missing)}"
        return result

    import torch
    import transformers
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    result["torch_version"] = torch.__version__
    result["transformers_version"] = transformers.__version__
    result["cuda_available"] = bool(torch.cuda.is_available())
    result["cpu_threads"] = torch.get_num_threads()

    # --- Load checkpoint (timed) ---
    load_t0 = time.perf_counter()
    try:
        proc = AutoImageProcessor.from_pretrained(str(CHECKPOINT_DIR))
        model = AutoModelForImageClassification.from_pretrained(
            str(CHECKPOINT_DIR)
        ).eval()
    except Exception as exc:
        result["status"] = "LIMITED/UNMEASURED"
        result["error"] = f"checkpoint load failed: {exc}"
        return result
    load_t1 = time.perf_counter()
    result["load_seconds"] = load_t1 - load_t0

    # --- Model introspection ---
    cfg = model.config
    result["native_labels"] = [cfg.id2label[i] for i in range(len(cfg.id2label))]
    result["embedding_dim"] = cfg.hidden_size
    result["head_out_features"] = cfg.num_labels

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Determine frozen vs trainable
    # ViT backbone = vit.embeddings + vit.encoder + vit.layernorm + vit.pooler
    # Head = classifier
    backbone_params = 0
    head_params = 0
    backbone_trainable = 0
    head_trainable = 0
    for name, p in model.named_parameters():
        is_head = name.startswith("classifier")
        if is_head:
            head_params += p.numel()
            if p.requires_grad:
                head_trainable += p.numel()
        else:
            backbone_params += p.numel()
            if p.requires_grad:
                backbone_trainable += p.numel()

    result["total_params"] = total_params
    result["trainable_params_total"] = trainable_total
    result["backbone_params"] = backbone_params
    result["backbone_trainable_params"] = backbone_trainable
    result["head_params"] = head_params
    result["head_trainable_params"] = head_trainable
    result["frozen_backbone"] = backbone_trainable == 0
    result["trainable_head_only"] = (
        head_trainable == head_params and backbone_trainable == 0
    )

    # --- Synthetic probe (timed, bounded forwards) ---
    probe_t0 = time.perf_counter()

    import resource

    # Warm-up: 1 forward
    dummy = np.zeros((1, 224, 224, 3), dtype=np.uint8)
    try:
        with torch.no_grad():
            inputs = proc(images=dummy, return_tensors="pt")
            _ = model(**inputs).logits
    except Exception as exc:
        result["status"] = "LIMITED/UNMEASURED"
        result["error"] = f"warm-up forward failed: {exc}"
        return result

    # Timed forwards: NUM_FORWARDS
    forward_times = []
    for _ in range(NUM_FORWARDS):
        fwd_t0 = time.perf_counter()
        with torch.no_grad():
            inputs = proc(images=dummy, return_tensors="pt")
            logits = model(**inputs).logits
        fwd_t1 = time.perf_counter()
        forward_times.append((fwd_t1 - fwd_t0) * 1000.0)

    probe_t1 = time.perf_counter()
    result["probe_seconds"] = probe_t1 - probe_t0
    result["forward_times_ms"] = forward_times
    result["total_seconds"] = probe_t1 - load_t0

    # Peak RSS (approx, via resource module on Linux; may not work on Windows)
    try:
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On Linux ru_maxrss is in KB; on macOS it's in bytes
        if peak_kb > 1 << 20:  # likely bytes (macOS)
            result["peak_rss_mb"] = peak_kb / (1 << 20)
        else:  # likely KB (Linux)
            result["peak_rss_mb"] = peak_kb / 1024.0
    except Exception:
        result["peak_rss_mb"] = None

    result["status"] = "READY_FOR_GATED_PILOT"
    return result


if __name__ == "__main__":
    # Parent-process hard cap via signal alarm (Unix only); on Windows, rely on
    # the caller's subprocess timeout instead.
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(HARD_CAP_SECONDS)

    try:
        result = run_probe()
    except TimeoutError:
        print(json.dumps({"status": "BLOCKED", "error": "60s hard cap exceeded"}))
        sys.exit(2)
    except Exception as exc:
        print(json.dumps({"status": "LIMITED/UNMEASURED", "error": str(exc)}))
        sys.exit(3)
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)

    print(json.dumps(result, indent=2))
    sys.exit(0 if result["status"] == "READY_FOR_GATED_PILOT" else 1)
