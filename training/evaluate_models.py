"""Common-model evaluation entry point with injectable loaders and factories.

Synthetic contract tests inject all data/model dependencies. Importing this
module does not import a model runtime, construct an adapter, or open a dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from emolab.benchmark.runner import FER7, BenchmarkContractError, run_benchmark

DEFAULT_MODELS = ("mini-xception", "vit")
MODEL_MODULES = {
    "mini-xception": ("emolab.adapters.mini_xception", "MiniXceptionAdapter"),
    "vit": ("emolab.adapters.vit", "ViTAdapter"),
    "claude-hsemotion": ("emolab.adapters.claude_hsemotion", "HSEmotionAdapter"),
    "claude-combined": ("emolab.adapters.claude_combined", "ClaudeCombinedAdapter"),
}


def _construct_adapter(name):
    from importlib import import_module
    module, class_name = MODEL_MODULES[name]
    return getattr(import_module(module), class_name)()


def _default_factories():
    return {name: (lambda name=name: _construct_adapter(name)) for name in MODEL_MODULES}


def _load_fer2013_test(n_per_class, seed=505, *, loader=None,
                      source_dir="eval_data/fer2013/test"):
    if loader is None:
        from training.fer_images import load_split
        loader = load_split
    X, y, files = loader(source_dir)
    y = np.asarray(y)
    if len(X) != len(y) or len(files) != len(y):
        raise BenchmarkContractError("loader image/label/file lengths differ")
    if n_per_class is not None and n_per_class <= 0:
        raise BenchmarkContractError("per-class subset must be positive")
    rng = np.random.default_rng(seed)
    chosen = []
    per_class = {}
    for class_index, class_name in enumerate(FER7):
        candidates = np.flatnonzero(y == class_index)
        take = len(candidates) if n_per_class is None else min(n_per_class, len(candidates))
        chosen.extend(rng.choice(candidates, take, replace=False).tolist())
        per_class[class_name] = {"available": len(candidates), "selected": take,
                                 "requested": n_per_class}
    chosen = sorted(chosen)
    selected_set = set(chosen)
    # The existing image loader returns basenames; labels supply the relative
    # class directory. No file content/hash is inspected here.
    ids = [f"{FER7[int(label)]}/{filename}" for label, filename in zip(y, files)]
    return {
        "images": [np.stack([X[i]] * 3, -1) for i in chosen],
        "labels": [int(y[i]) for i in chosen], "sample_ids": [ids[i] for i in chosen],
        "requested_count": len(chosen) if n_per_class is None else n_per_class * len(FER7),
        "skipped_samples": [{"sample_id": ids[i], "reason": "outside predefined seeded subset"}
                            for i in range(len(ids)) if i not in selected_set],
        "configuration": {"per_class_counts": per_class, "subset_unit": "per class"},
    }


def _write_report(path, report):
    # Serialize first and create exclusively: invalid/nonfinite output cannot
    # clobber a previous result, including when a destination appears in a race.
    payload = json.dumps(report, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload + "\n")


def main(argv=None, *, adapter_factories=None, dataset_loaders=None,
         clock=None, model_references=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", default=None)
    ap.add_argument("--dataset", default="fer2013-test")
    ap.add_argument("--subset", type=int, default=350, help="requested count per FER7 class")
    ap.add_argument("--seed", type=int, default=505)
    ap.add_argument("--out", default=None, help="new output file; existing files are refused")
    a = ap.parse_args(argv)
    requested = list(a.model) if a.model is not None else list(DEFAULT_MODELS)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(a.out) if a.out else Path("eval_results/runner-contract-v1") / (stamp + "-" + uuid.uuid4().hex) / "report.json"
    if out_path.exists():
        print(f"refusing to overwrite existing result: {out_path}", file=sys.stderr)
        return 2
    factories = _default_factories() if adapter_factories is None else adapter_factories
    loaders = {"fer2013-test": lambda subset, seed: _load_fer2013_test(subset, seed)} if dataset_loaders is None else dataset_loaders
    cohort = None
    try:
        if not requested or len(set(requested)) != len(requested):
            raise BenchmarkContractError("requested models must be nonempty and unique")
        unknown = set(requested) - set(factories)
        if unknown:
            raise BenchmarkContractError(f"unknown requested models: {sorted(unknown)}")
        if a.dataset not in loaders:
            raise BenchmarkContractError(
                f"dataset route {a.dataset!r} lacks the strict common-ID loader contract; "
                "cached CREMA-D outputs cannot substitute for requested-model execution")
        cohort = loaders[a.dataset](subset=a.subset, seed=a.seed)
        adapters, factory_errors = [], {}
        for name in requested:
            try:
                adapters.append(factories[name]())
            except Exception as exc:
                factory_errors[name] = f"factory {type(exc).__name__}: {exc}"
        report = run_benchmark(
            adapters, cohort["images"], cohort["labels"], dataset=a.dataset,
            sample_ids=cohort["sample_ids"], requested_count=cohort["requested_count"],
            requested_models=requested, skipped_samples=cohort.get("skipped_samples", ()),
            configuration={**cohort.get("configuration", {}), "seed": a.seed,
                           "requested_subset_per_class": a.subset},
            model_references=model_references, synthetic=cohort.get("synthetic_only", False),
            clock=clock, return_report=True)
        for name, error in factory_errors.items():
            report["results"][name]["error"] = error
    except Exception as exc:
        candidate_ids = [] if cohort is None else list(cohort.get("sample_ids", []))
        unique_count = (len(set(candidate_ids)) if all(isinstance(x, str) for x in candidate_ids)
                        else None)
        report = {"schema": "emolab.benchmark-execution/1", "status": "failed", "complete": False,
                  "dataset": a.dataset, "requested_models": requested,
                  "synthetic_only": False if cohort is None else bool(cohort.get("synthetic_only", False)),
                  "error": f"{type(exc).__name__}: {exc}",
                  "requested_count": None if cohort is None else cohort.get("requested_count"),
                  "actual_unique_count": None if cohort is None else unique_count,
                  "sample_ids": candidate_ids,
                  "skipped_samples": [] if cohort is None else cohort.get("skipped_samples", []),
                  "configuration": {**({} if cohort is None else cohort.get("configuration", {})),
                                    "seed": a.seed, "requested_subset_per_class": a.subset}}
    report["output_path"] = str(out_path)
    _write_report(out_path, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
