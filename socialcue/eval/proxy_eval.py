"""Proxy-label evaluation of HB-1 over a DatasetAdapter (EL-006 S3). Research prototype.

Pure Python, no file or network access. Data enters only through a DatasetAdapter
after the rights gate. Honest non-synthetic declarations are refused even if
granted. Only the two exact reviewed synthetic adapter classes are accepted, using one
decision end-to-end. Unreviewed in-process impersonation is outside the boundary.

Proxy endpoints (cycle order section 2): P1 "affect consistent with nervous-like"
and P2 "negative-arousal affect". They are proxies from acted/manufactured labels;
they do not measure nervousness, uncertainty or need for help.

KNOWN LIMITATIONS (visible in every result and report):
- TODO(before any real run, gates G1/G2): "FINAL is opened once" and "protocol frozen and hashed before the run"
  (cycle order section 5) are NOT enforced here. ``evaluate()`` computes DEV, VAL and FINAL on every call and
  records ``final_opened_once_enforced: false`` and ``protocol_hash: null`` in its output. Build the frozen
  protocol hash and the one-time FINAL opening in a separately reviewed real runner;
  this entry point never accepts non-synthetic declarations.
- AUROC and ECE are not computed (predictions are binary).
- ``fer7_only`` is NOT a pure no-temporal ablation: it shares HB-1's window, coverage, quality and abstention
  gates (it abstains whenever HB-1 abstains). Only its positive decision is FER7-only. See METHOD_NOTES.

Items must provide ``records()`` (a chronological FrameRecord list) and a raw
label in ``label`` (or ``scenario``); the adapter's ``label_map()`` turns that
into a proxy class and ``group_id(item)`` gives the actor.
"""
from __future__ import annotations

import math
import random

from .. import state as hb1
from ..data.provenance import manifest_from_decision
from ..data.rights import DatasetAdapter, RightsRefusal, authorize
from ..data.synthetic_adapter import SyntheticAdapter
from ..data.synthetic_commercial_adapter import SyntheticCommercialCleanAdapter
from .report import CAVEAT, PROXY_LABEL, SYNTHETIC_LABEL

ROLES = ("DEV", "VAL", "FINAL")
SEED = 505
N_DRAWS = 1000
METHODS = ("hb1", "always_no_cue", "fer7_only")
METRICS = ("precision", "recall", "f1", "balanced_accuracy")
METHOD_NOTES = {
    "hb1": "HB-1 heuristic (nervous-like; P2 also unsure-like) at the end of each clip.",
    "always_no_cue": "Always predicts 'no cue'; never abstains.",
    "fer7_only": ("FER7-only decision: positive when the mean negative-arousal probability in the last window exceeds HB-1's own "
                  "threshold, ignoring head motion, arousal and hesitation. NOT a pure no-temporal ablation: it shares HB-1's window, "
                  "coverage, quality and abstention gates, so it abstains whenever HB-1 abstains."),
}
USE = "research_evaluation"
EXCLUDED = "excluded"  # sentinel class the adapters use for clips outside every proxy class: not a confuser

# proxy classes on each side of an endpoint; everything else is a confuser
ENDPOINTS = {
    "P1": {
        "positive": ("proxy_positive", "fear"),
        "negative": ("proxy_negative", "neutral", "calm"),
        "hb1_states": ("nervous-like",),
    },
    "P2": {
        "positive": ("proxy_positive", "fear", "sad", "surprise"),
        "negative": ("proxy_negative", "neutral", "calm", "happy"),
        "hb1_states": ("nervous-like", "unsure-like"),
    },
}


# --- splits ---------------------------------------------------------------------

def actor_disjoint_split(actors, seed=SEED, fractions=(0.5, 0.25, 0.25)):
    """Deterministic role -> sorted actor list, with every actor in exactly one role.

    Sizes use the largest remainder method: each role first gets floor(n * fraction), the leftover actors go to the
    roles with the largest fractional parts, and ties go to FINAL first, then VAL, then DEV. Every role keeps at
    least one actor. With the default fractions VAL and FINAL differ by at most one actor and DEV is never the smaller.
    """
    ids = sorted(set(actors))
    if len(ids) < len(ROLES):
        raise ValueError(f"need at least {len(ROLES)} actors, got {len(ids)}")
    random.Random(seed).shuffle(ids)
    n = len(ids)
    quotas = [n * f for f in fractions]
    sizes = [int(q) for q in quotas]
    priority = {"FINAL": 0, "VAL": 1, "DEV": 2}
    order = sorted(range(len(ROLES)), key=lambda i: (-(quotas[i] - sizes[i]), priority[ROLES[i]]))
    for i in order[:n - sum(sizes)]:
        sizes[i] += 1
    for i in range(len(sizes)):  # keep every role non-empty, taking from the largest role
        while sizes[i] < 1:
            sizes[sizes.index(max(sizes))] -= 1
            sizes[i] += 1
    cut1, cut2 = sizes[0], sizes[0] + sizes[1]
    split = {"DEV": sorted(ids[:cut1]), "VAL": sorted(ids[cut1:cut2]), "FINAL": sorted(ids[cut2:])}
    assert_actor_disjoint(split, expected=ids)
    return split


def assert_actor_disjoint(split, expected=None):
    seen = {}
    for role, actors in split.items():
        for a in actors:
            if a in seen:
                raise ValueError(f"actor {a!r} appears in both {seen[a]} and {role}")
            seen[a] = role
    if expected is not None and set(seen) != set(expected):
        raise ValueError("split does not cover exactly the expected actors")
    if any(not actors for actors in split.values()):
        raise ValueError("every role needs at least one actor")


# --- metrics --------------------------------------------------------------------

def confusion(outcomes, mode="full_cohort"):
    """outcomes: (actor, truth_positive, pred) with pred in pos|neg|abstain.

    full_cohort counts an abstention as a failure (wrong on both classes);
    answered_only drops abstentions.
    """
    tp = fp = tn = fn = ab = 0
    for _, truth, pred in outcomes:
        if pred == "abstain":
            ab += 1
            if mode == "answered_only":
                continue
            pred = "neg" if truth else "pos"
        if truth and pred == "pos":
            tp += 1
        elif truth:
            fn += 1
        elif pred == "pos":
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "abstained": ab, "n": len(outcomes)}


def _div(a, b):
    return a / b if b else None


def metrics(c):
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    recall, spec = _div(tp, tp + fn), _div(tn, tn + fp)
    return {
        "precision": _div(tp, tp + fp),
        "recall": recall,
        "specificity": spec,
        "f1": _div(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": None if recall is None or spec is None else (recall + spec) / 2,
        "abstention_rate": _div(c["abstained"], c["n"]),
    }


def _percentile(sorted_vals, q):
    k = (len(sorted_vals) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def bootstrap_ci(outcomes, mode="full_cohort", n_draws=N_DRAWS, seed=SEED, names=METRICS):
    """95% actor-grouped percentile bootstrap. Draws where a metric is undefined are skipped (n_valid)."""
    by_actor = {}
    for o in outcomes:
        by_actor.setdefault(o[0], []).append(o)
    actors = sorted(by_actor)
    point = metrics(confusion(outcomes, mode))
    out = {n: {"estimate": point[n], "lo": None, "hi": None, "n_valid": 0} for n in names}
    if not actors:
        return out
    rng = random.Random(seed)
    draws = {n: [] for n in names}
    for _ in range(n_draws):
        sample = []
        for a in (actors[rng.randrange(len(actors))] for _ in actors):
            sample.extend(by_actor[a])
        m = metrics(confusion(sample, mode))
        for n in names:
            if m[n] is not None:
                draws[n].append(m[n])
    for n in names:
        v = sorted(draws[n])
        out[n]["n_valid"] = len(v)
        if v:
            out[n]["lo"], out[n]["hi"] = _percentile(v, 0.025), _percentile(v, 0.975)
    return out


# --- predictions ----------------------------------------------------------------

def predict_clip(records, cfg=None):
    """Per-clip HB-1 result and the FER7-only ablation. Returns {'hb1': {...}, 'abstain': bool, 'fer7_pos': bool}."""
    cfg = hb1.HB1_CONFIG if cfg is None else cfg
    result = hb1.analyze(records, cfg)
    abstain = bool(result["insufficient"]["flag"])
    fer7_pos = False
    if not abstain:
        fer7_pos = hb1.extract_features(records, cfg)["p_neg_arousal"] > cfg["nervous_like"]["p_neg_arousal_thr"]
    return {"states": {s["name"] for s in result["states"]}, "abstain": abstain, "fer7_pos": fer7_pos}


def _raw_label(item):
    label = getattr(item, "label", None)
    return label if label is not None else getattr(item, "scenario", None)


def _outcome(method, endpoint, pred, actor, truth):
    if method == "always_no_cue":
        return (actor, truth, "neg")
    if pred["abstain"]:
        return (actor, truth, "abstain")
    if method == "fer7_only":
        return (actor, truth, "pos" if pred["fer7_pos"] else "neg")
    hit = bool(pred["states"] & set(ENDPOINTS[endpoint]["hb1_states"]))
    return (actor, truth, "pos" if hit else "neg")


# --- harness --------------------------------------------------------------------

def evaluate(adapter, run_scope, seed=SEED, n_draws=N_DRAWS, cfg=None, use=USE):
    """Run HB-1 and both comparators on ``adapter`` after the rights gate; actor-disjoint DEV/VAL/FINAL.

    Raises RightsRefusal for blocked, unknown or forbidden adapters and TypeError for
    anything that is not a DatasetAdapter (paths and file handles are not accepted).
    """
    if not isinstance(adapter, DatasetAdapter):
        raise TypeError("evaluate() takes a DatasetAdapter only")
    decision = authorize(adapter, use, run_scope)
    # Honest non-synthetic declarations are refused even under a future grant.
    # authorize() above may execute subclass-authored declaration properties before
    # the exact-class pin refuses: subclass code under a refusal-bound call.
    # Repo code reads no dataset payload here; label_map/_items/group_id wait below.
    reviewed = {
        "synthetic": (SyntheticAdapter, "synthetic"),
        "synthetic_commercial_clean": (SyntheticCommercialCleanAdapter, "commercial_clean"),
    }
    expected = reviewed.get(decision.adapter)
    if expected is None or decision.source_class != "synthetic" or decision.data_scope != expected[1]:
        raise RightsRefusal("NON_SYNTHETIC_EVALUATION_DISABLED", "legacy evaluate cannot open real records")
    reviewed_class = expected[0]
    if (type(adapter) is not reviewed_class or adapter.name != decision.adapter
            or set(vars(adapter)) != {"_n", "_seed", "_scenarios"}):
        raise RightsRefusal("UNREVIEWED_EVALUATOR_ADAPTER", "exact reviewed synthetic adapter classes only")
    # Consume reviewed metadata methods under this decision; items() would re-authorize.
    label_map = reviewed_class.label_map(adapter)
    items = reviewed_class._items(adapter)
    split = actor_disjoint_split([reviewed_class.group_id(adapter, i) for i in items], seed)
    role_of = {a: r for r, actors in split.items() for a in actors}

    clips = []
    for it in items:
        raw = _raw_label(it)
        clips.append((reviewed_class.group_id(adapter, it), label_map.get(raw, EXCLUDED), predict_clip(it.records(), cfg)))

    endpoints = {}
    confusers = {}
    for ep, spec in ENDPOINTS.items():
        endpoints[ep] = {}
        for role in ROLES:
            res = {}
            for method in METHODS:
                outs = []
                for actor, proxy, pred in clips:
                    if role_of[actor] != role or proxy not in spec["positive"] + spec["negative"]:
                        continue
                    outs.append(_outcome(method, ep, pred, actor, proxy in spec["positive"]))
                full, ans = confusion(outs, "full_cohort"), confusion(outs, "answered_only")
                res[method] = {
                    "n_clips": len(outs),
                    "full_cohort": {"confusion": full, **metrics(full)},
                    "answered_only": {"confusion": ans, **metrics(ans)},
                    "ci_full_cohort": bootstrap_ci(outs, "full_cohort", n_draws, seed),
                }
            endpoints[ep][role] = res
        # confusers: clips in neither class, reported separately with the rate HB-1 says positive
        conf = [(a, p, pr) for a, p, pr in clips if p not in spec["positive"] + spec["negative"] and p != EXCLUDED]
        n_abstained = sum(1 for _, _, pr in conf if pr["abstain"])
        n_answered = len(conf) - n_abstained
        positive_n = sum(1 for _, _, pr in conf if not pr["abstain"] and pr["states"] & set(spec["hb1_states"]))
        confusers[ep] = {
            "n_clips": len(conf),
            "excluded_n": sum(1 for _, p, _ in clips if p == EXCLUDED),
            "classes": sorted({p for _, p, _ in conf}),
            "n_answered": n_answered,
            "n_abstained": n_abstained,
            "hb1_positive_n": positive_n,
            "hb1_positive_rate": _div(positive_n, n_answered),
            "abstention_rate": _div(n_abstained, len(conf)),
            "rate_definition": "hb1_positive_rate = hb1_positive_n / n_answered (abstentions excluded); abstention_rate = n_abstained / n_clips",
            "note": "excluded_n counts clips the adapter mapped to no proxy class (excluded by the adapter); they are not confusers.",
        }

    manifest = manifest_from_decision(decision, {"actors": len(role_of), "clips": len(clips)},
                                      config_sha256=hb1.config_sha256(hb1.HB1_CONFIG if cfg is None else cfg))
    synthetic = decision.source_class == "synthetic"
    return {
        "schema": "socialcue-proxy-eval-v1",
        "banner": SYNTHETIC_LABEL if synthetic else PROXY_LABEL,
        "caveat": CAVEAT,
        "final_opened_once_enforced": False,
        "protocol_hash": None,
        "method_notes": dict(METHOD_NOTES),
        "adapter": {"name": decision.adapter, "source_class": decision.source_class, "scope": decision.data_scope},
        "run_scope": run_scope,
        "seed": seed,
        "n_draws": n_draws,
        "splits": split,
        "endpoints": endpoints,
        "confusers": confusers,
        "manifest": manifest,
        "config_sha256": manifest["hb1_config_sha256"],
        "notes": [
            "Abstentions/insufficient windows count as failures in full_cohort; answered_only is shown separately.",
            "P1 and P2 coincide when the adapter only provides proxy_positive/proxy_negative classes.",
            "AUROC and calibration (ECE) are not computed in this harness version.",
            "FINAL-opened-once and a frozen protocol hash are NOT enforced (final_opened_once_enforced=false, protocol_hash=null).",
            "fer7_only shares HB-1's window, coverage, quality and abstention gates: it is not a pure no-temporal ablation.",
        ],
    }
