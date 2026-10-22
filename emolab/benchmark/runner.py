"""Shared benchmark execution with an explicit sample and completion contract.

The legacy mapping return is retained. ``return_report=True`` includes the
immutable sample manifest, timing boundaries, provenance and failure status.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from typing import Callable, Optional

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from emolab.inference import FaceEmotionAdapter

FER7 = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class BenchmarkContractError(ValueError):
    """Input cannot define the promised common evaluation cohort."""


class BenchmarkExecutionError(RuntimeError):
    def __init__(self, report):
        super().__init__("benchmark execution incomplete; inspect the attached report")
        self.report = report


def expected_calibration_error(proba, y_true, n_bins=10):
    proba = np.asarray(proba, dtype=float)
    y_true = np.asarray(y_true)
    if len(y_true) == 0:
        return 0.0
    conf, pred = proba.max(axis=1), proba.argmax(axis=1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (conf >= lo) & ((conf < hi) if b < n_bins - 1 else (conf <= 1.0))
        if m.any():
            ece += m.mean() * abs(float((pred[m] == y_true[m]).mean()) - float(conf[m].mean()))
    return float(ece)


def _unique_names(values, label):
    names = tuple(values)
    if not names or any(not isinstance(x, str) or not x.strip() for x in names):
        raise BenchmarkContractError(f"{label} must be nonempty strings")
    if len(set(names)) != len(names):
        raise BenchmarkContractError(f"duplicate {label}")
    return names


def _duration_ms(start, end):
    duration = (end - start) * 1000.0
    if not math.isfinite(duration) or duration < 0:
        raise BenchmarkContractError("clock must be finite and monotonic")
    return float(duration)


def _probs_array(result):
    supported = set(result.metadata.supported_classes)
    if not set(FER7).issubset(supported):
        raise BenchmarkContractError("model does not support the common FER7 class space")
    # Sparse zeros are valid only for classes declared supported by the model.
    if any(c not in FER7 and float(value) != 0 for c, value in result.probabilities.items()):
        raise BenchmarkContractError("non-FER7 probability mass cannot be silently discarded")
    p = np.array([float(result.probabilities.get(c, 0.0)) for c in FER7], dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1, atol=1e-6, rtol=0):
        raise BenchmarkContractError("invalid common-class probability distribution")
    return p


def _metrics(probs, y, name, dataset, observed_latencies):
    detected = np.array([p is not None for p in probs])
    n = len(y)
    base = {
        "adapter": name, "dataset": dataset, "n": n,
        "n_detected": int(detected.sum()), "coverage": float(detected.mean()),
        "abstention_coverage": float(1 - detected.mean()),
        "latency_ms": float(np.mean(observed_latencies)),
        "accuracy": None, "macro_f1": None, "selective_accuracy": None,
        "per_class_precision": {c: None for c in FER7},
        "per_class_recall": {c: None for c in FER7},
        "per_class_f1": {c: None for c in FER7},
        "confusion": [[0] * len(FER7) for _ in FER7], "ece": None,
    }
    if detected.any():
        truth = y[detected]
        p = np.asarray([x for x in probs if x is not None], dtype=float)
        predicted = p.argmax(axis=1)
        # Preserve the existing metric definitions; this change concerns execution.
        base["accuracy"] = float(accuracy_score(truth, predicted))
        base["macro_f1"] = float(f1_score(truth, predicted, average="macro", zero_division=0))
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, predicted, labels=list(range(len(FER7))), average=None, zero_division=0)
        for field, values in (("per_class_precision", precision),
                              ("per_class_recall", recall), ("per_class_f1", f1)):
            base[field] = dict(zip(FER7, map(float, values)))
        base["confusion"] = confusion_matrix(truth, predicted,
                                              labels=list(range(len(FER7)))).tolist()
        base["ece"] = expected_calibration_error(p, truth)
        base["selective_accuracy"] = base["accuracy"]
    return base


def run_benchmark(adapters: list[FaceEmotionAdapter], X, y, dataset="eval",
                  crop_fn: Optional[Callable] = None, *, sample_ids=None,
                  requested_count=None, requested_models=None, clock=None,
                  return_report=False, configuration=None, model_references=None,
                  skipped_samples=(), synthetic=False):
    """Run one fixed cohort through every requested adapter.

    Inputs/crops are copied once and made read-only before any model executes.
    A None model result is an explicit abstention, not a missing execution.
    Exceptions or absent models produce an incomplete structured report; the
    legacy mapping interface raises BenchmarkExecutionError in those cases.

    Observed item timing surrounds predict_proba only. Model timing surrounds
    its prediction loop (not scoring). Total timing includes common preparation,
    prediction and scoring, excluding argument validation and output writing.
    Adapter-reported timing is retained separately and never relabelled.
    """
    images = tuple(X)
    truth = np.array(y, copy=True)
    n = len(images)
    if n == 0 or truth.ndim != 1 or len(truth) != n:
        raise BenchmarkContractError("nonempty image/label lists must have equal lengths")
    if not np.issubdtype(truth.dtype, np.integer) or (truth < 0).any() or (truth >= len(FER7)).any():
        raise BenchmarkContractError("labels must be integer FER7 indices")
    if return_report and sample_ids is None:
        raise BenchmarkContractError("explicit sample IDs are required for execution reports")
    ids = _unique_names(sample_ids if sample_ids is not None else
                        (f"position:{i}" for i in range(n)), "sample IDs")
    if len(ids) != n:
        raise BenchmarkContractError("sample-ID/image/label lengths differ")
    count = n if requested_count is None else requested_count
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0 or count != n:
        raise BenchmarkContractError(f"requested count {count!r} != actual unique count {n}")
    adapters = tuple(adapters)
    names = _unique_names([a.name for a in adapters], "adapter names") if adapters else ()
    requested = _unique_names(requested_models if requested_models is not None else names,
                              "requested models")
    extra = set(names) - set(requested)
    if extra:
        raise BenchmarkContractError(f"unexpected adapters: {sorted(extra)}")
    by_name = dict(zip(names, adapters))
    skipped = copy.deepcopy(list(skipped_samples))
    skipped_ids = []
    for item in skipped:
        if (not isinstance(item, dict) or not isinstance(item.get("reason"), str)
                or not item["reason"].strip() or not isinstance(item.get("sample_id"), str)):
            raise BenchmarkContractError("every skipped sample needs a string ID and reason")
        skipped_ids.append(item["sample_id"])
    if skipped_ids:
        _unique_names(skipped_ids, "skipped sample IDs")
        if set(ids) & set(skipped_ids):
            raise BenchmarkContractError("skipped IDs overlap the selected cohort")
    config = copy.deepcopy(configuration or {})
    references = copy.deepcopy(model_references or {})
    json.dumps([config, references, skipped], allow_nan=False)
    now = clock or time.perf_counter
    total_start = now()
    prepared = []
    for image in images:
        crop = image if crop_fn is None else crop_fn(np.array(image, copy=True))
        if crop is None:
            raise BenchmarkContractError("common crop missing; define exclusions before execution")
        crop = np.asarray(crop)
        if crop.dtype.hasobject:
            raise BenchmarkContractError("common crops must contain numeric pixels")
        # Bytes-backed storage cannot be made writable again by an adapter.
        crop = np.frombuffer(crop.tobytes(), dtype=crop.dtype).reshape(crop.shape)
        prepared.append(crop)
    prepared = tuple(prepared)
    manifest_hash = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    report = {
        "schema": "emolab.benchmark-execution/1", "dataset": dataset,
        "synthetic_only": bool(synthetic), "complete": False, "status": "incomplete",
        "requested_models": list(requested), "requested_count": count,
        "actual_unique_count": n, "sample_ids": list(ids),
        "sample_id_source": "provided" if sample_ids is not None else "positional",
        "sample_manifest_sha256": manifest_hash, "skipped_samples": skipped,
        "configuration": config, "model_references": {}, "results": {},
    }
    for name in requested:
        adapter = by_name.get(name)
        report["model_references"][name] = references.get(name, {
            "model_id": name, "version": getattr(adapter, "version", None),
            "checkpoint_reference": getattr(adapter, "model_path", None),
            "checkpoint_checksum": None, "checkpoint_verified": False,
        })
        if adapter is None:
            report["results"][name] = {"status": "missing", "complete": False,
                "error": "requested adapter unavailable", "sample_ids": list(ids),
                "attempted_sample_ids": [], "unprocessed_sample_ids": list(ids), "items": []}
            continue
        items, probabilities, latencies = [], [], []
        model_start = now()
        error = None
        for sid, crop in zip(ids, prepared):
            item_start = now()
            item_end = None
            try:
                result = adapter.predict_proba(crop)
                item_end = now()
                duration = _duration_ms(item_start, item_end)
                vector = None if result is None else _probs_array(result)
                reported = None if result is None else float(result.latency_ms)
                if reported is not None and (not math.isfinite(reported) or reported < 0):
                    raise BenchmarkContractError("invalid adapter-reported latency")
                probabilities.append(vector)
                latencies.append(duration)
                items.append({"sample_id": sid, "status": "abstained" if result is None else "predicted",
                              "observed_inference_ms": duration,
                              "adapter_reported_inference_ms": reported})
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                duration = _duration_ms(item_start, now() if item_end is None else item_end)
                latencies.append(duration)
                items.append({"sample_id": sid, "status": "failed", "error": error,
                              "observed_inference_ms": duration,
                              "adapter_reported_inference_ms": None})
                break
        model_end = now()
        complete = error is None and len(items) == n
        result_report = {
            "status": "complete" if complete else "failed", "complete": complete,
            "sample_ids": list(ids), "attempted_sample_ids": [x["sample_id"] for x in items],
            "unprocessed_sample_ids": list(ids[len(items):]),
            "abstained_sample_ids": [x["sample_id"] for x in items if x["status"] == "abstained"],
            "items": items, "model_prediction_wall_ms": _duration_ms(model_start, model_end),
            "observed_inference_sum_ms": float(sum(latencies)),
        }
        if complete:
            result_report.update(_metrics(probabilities, truth, name, dataset, latencies))
        else:
            result_report["error"] = error or "incomplete predictions"
            result_report["n"] = n
            result_report["n_processed"] = len(probabilities)
        report["results"][name] = result_report
    report["total_wall_ms"] = _duration_ms(total_start, now())
    report["complete"] = all(report["results"][name]["complete"] for name in requested)
    report["status"] = "complete" if report["complete"] else "incomplete"
    if return_report:
        return report
    if not report["complete"]:
        raise BenchmarkExecutionError(report)
    return report["results"]


# Opt-in DEVELOPMENT path. Legacy FER execution/metrics above are unchanged.
DEVELOPMENT_TRUTHS = tuple(c for c in FER7 if c != "surprise")
DEVELOPMENT_EXPOSURE = "DEVELOPMENT / RETROSPECTIVE; upstream CREMA-D exposure UNKNOWN"

def development_metrics(items):
    if not items: raise BenchmarkContractError("empty development cohort")
    confusion=np.zeros((6,8),dtype=int);latencies=[]
    for row in items:
        if row["label"] not in DEVELOPMENT_TRUTHS:raise BenchmarkContractError("unknown intended class")
        truth=DEVELOPMENT_TRUTHS.index(row["label"])
        p=row.get("probabilities")
        if p is None:
            if row.get("status") not in ("no_face","decode_failure"):raise BenchmarkContractError("missing prediction status")
            predicted=7
        else:
            a=np.asarray(p,dtype=float)
            if a.shape!=(7,) or not np.isfinite(a).all() or (a<0).any() or not np.isclose(a.sum(),1,atol=1e-6,rtol=0):raise BenchmarkContractError("invalid seven-vector")
            predicted=int(a.argmax())
        confusion[truth,predicted]+=1
        latencies.extend(row.get("observed_inference_ms",[]))
    per={};correct=0
    for name in FER7:
        col=FER7.index(name);support=int(confusion[DEVELOPMENT_TRUTHS.index(name)].sum()) if name in DEVELOPMENT_TRUTHS else 0
        tp=int(confusion[DEVELOPMENT_TRUTHS.index(name),col]) if support else 0
        fp=int(confusion[:,col].sum())-tp;fn=support-tp;correct+=tp
        per[name]={"support":support,"precision":tp/(tp+fp) if tp+fp else 0.,
                   "recall":tp/support if support else None,"f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}
    if any(not math.isfinite(x) or x<0 for x in latencies):raise BenchmarkContractError("invalid timing")
    return {"exposure":DEVELOPMENT_EXPOSURE,"n":len(items),"accuracy":correct/len(items),
        "macro_f1":sum(per[c]["f1"] for c in DEVELOPMENT_TRUTHS)/6,"per_class":per,
        "coverage":1-float(confusion[:,7].sum())/len(items),"confusion":confusion.tolist(),
        "truth_order":list(DEVELOPMENT_TRUTHS),"prediction_order":FER7+["no_face/decode_failure"],
        "ece":None,"ece_reason":"no authorized validation split/probability endpoint",
        "timing_ms":{"count":len(latencies),"total":sum(latencies),"mean":float(np.mean(latencies)) if latencies else None,
                     "p50":float(np.percentile(latencies,50)) if latencies else None,"p95":float(np.percentile(latencies,95)) if latencies else None}}

def development_bootstrap(model_items, *, seed=505):
    """1000 paired actor-cluster draws. Multiplicity preserved, never frames."""
    names=list(model_items)
    if not names:raise BenchmarkContractError("no models for paired bootstrap")
    reference=model_items[names[0]]
    identity=lambda rows:[(r["id"],r["actor"],r["label"]) for r in rows]
    if any(identity(model_items[n])!=identity(reference) for n in names):raise BenchmarkContractError("paired cohort mismatch")
    actors=sorted({r["actor"] for r in reference});rng=np.random.default_rng(seed)
    draws=rng.choice(actors,size=(1000,len(actors)),replace=True)
    # Validate each clip once, cache actor confusion, then batch all draws.
    weights=np.stack([(draws==actor).sum(axis=1) for actor in actors],axis=1)
    values={};undefined={};recalls={}
    cols=np.array([FER7.index(label) for label in DEVELOPMENT_TRUTHS])
    for name in names:
        clusters={actor:[] for actor in actors}
        for row in model_items[name]:clusters[row["actor"]].append(row)
        cache=np.array([development_metrics(clusters[actor])["confusion"] for actor in actors])
        confusion=np.einsum("da,aij->dij",weights,cache)
        support=confusion.sum(axis=2)
        tp=confusion[:,np.arange(6),cols]
        fp=confusion[:,:,cols].sum(axis=1)-tp
        denominator=support+tp+fp
        f1=np.divide(2*tp,denominator,out=np.zeros_like(tp,dtype=float),where=denominator!=0)
        values[name]=f1.mean(axis=1)
        recall=np.divide(tp,support,out=np.zeros_like(tp,dtype=float),where=support!=0)
        undefined[name]={label:int((support[:,i]==0).sum()) for i,label in enumerate(DEVELOPMENT_TRUTHS)}
        recalls[name]={label:recall[support[:,i]!=0,i] for i,label in enumerate(DEVELOPMENT_TRUTHS)}
    ci=lambda xs:list(map(float,np.percentile(xs,[2.5,97.5]))) if len(xs) else None
    return {"exposure":DEVELOPMENT_EXPOSURE,"seed":seed,"replicates":len(draws),"actor_count":len(actors),"draw_shape":list(draws.shape),
            "draws_sha256":hashlib.sha256(draws.tobytes()).hexdigest(),"macro_f1_ci":{n:ci(values[n]) for n in names},
            "recall_ci":{n:{c:ci(v) for c,v in recalls[n].items()} for n in names},"undefined_recall_draws":undefined,
            "paired_macro_f1_difference_ci":{a+" minus "+b:ci(values[a]-values[b]) for i,a in enumerate(names) for b in names[i+1:]}}


def run_development_clips(adapters, manifest, prepare_clip, *, clock=time.perf_counter, progress=None, timing_only=False):
    """Independent opt-in clip route; no model loading or real IO here.

    prepare_clip returns identical readonly BGR crops and metadata per clip.
    Decoder failures are explicit; detector/adapter errors invalidate full model.
    Caller supervises each model separately and audits crop hashes across workers.
    timing_only never retains prediction vectors or computes scores (pilot).
    """
    names=[a.name for a in adapters]
    if not names or len(set(names))!=len(names) or not set(names)<={"mini-xception","vit"}:raise BenchmarkContractError("explicit primary roster required")
    ids=[r["id"] for r in manifest["rows"]]
    if len(set(ids))!=len(ids) or not ids:raise BenchmarkContractError("unique clip manifest required")
    report={"schema":"emolab.development-clips/1","exposure":DEVELOPMENT_EXPOSURE,"complete":False,
      "protocol_sha256":manifest["protocol_sha256"],"results":{},"blocked_roster":{"hse":"asset authority unavailable; no substitution","combined":"identity/access unresolved; retrospective only"}}
    for adapter in adapters:
        items=[];failure=None
        for row in manifest["rows"]:
            item={**row,"probabilities":None,"status":"no_face","observed_inference_ms":[],"adapter_reported_inference_ms":[],"frame_provenance":[]}
            try:
                crops,meta=prepare_clip(row);item.update(meta);vectors=[];crop_hashes=[]
                for crop in crops:
                    crop=np.asarray(crop);crop=np.frombuffer(crop.tobytes(),dtype=crop.dtype).reshape(crop.shape)
                    crop_hashes.append(hashlib.sha256(crop.tobytes()).hexdigest())
                    start=clock();result=adapter.predict_proba(crop);elapsed=_duration_ms(start,clock())
                    if result is None:raise BenchmarkContractError("adapter failure/abstention not authorized")
                    vector=_probs_array(result);vectors.append(vector)
                    reported=float(result.latency_ms)
                    if not math.isfinite(reported) or reported<0:raise BenchmarkContractError("invalid reported timing")
                    item["observed_inference_ms"].append(elapsed);item["adapter_reported_inference_ms"].append(reported)
                    item["frame_provenance"].append(copy.deepcopy(result.metadata.provenance))
                item["crop_sha256"]=crop_hashes;item["face_frames"]=len(vectors)
                if vectors:
                    if timing_only:item["status"]="processed"
                    else:item.update(status="predicted",probabilities=np.mean(vectors,axis=0).tolist())
                elif item.get("decoded_frames")==0:item["status"]="decode_failure"
            except Exception as exc:
                item.update(status="failed",error=type(exc).__name__+": "+str(exc));failure=item["error"]
            if timing_only:
                item.pop("probabilities",None)
                if item["status"]=="predicted":item["status"]="processed"
            items.append(item)
            if progress:progress(items,ids[len(items):])
            if failure:break
        complete=failure is None and len(items)==len(ids)
        report["results"][adapter.name]={"complete":complete,"items":items,"error":failure,
            "attempted_ids":[x["id"] for x in items],"unprocessed_ids":ids[len(items):],
            "metrics":development_metrics(items) if complete and not timing_only else None}
    report["complete"]=all(x["complete"] for x in report["results"].values())
    report["primary_roster_complete"]=report["complete"] and set(names)=={"mini-xception","vit"}
    return report
