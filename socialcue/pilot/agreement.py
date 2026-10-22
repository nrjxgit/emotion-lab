"""Inter-annotator agreement from the definitions (EL-006 S15; pilot protocol section 4.5). Pure stdlib.

Cohen's kappa (two raters), Krippendorff's alpha (nominal, missing values allowed), a participant-level bootstrap
helper (seed 505, 1000 draws) and a per-label summary with the protocol's small-N rules. Degenerate inputs return
``None`` with a reason; a statistic is never reported as 1.0 when it is undefined.
"""
from __future__ import annotations

import math
import random

SEED = 505
N_DRAWS = 1000
MIN_POSITIVES_FOR_INTERPRETATION = 20  # protocol 4.5: below this, counts and raw agreement only


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def cohen_kappa(a, b) -> dict:
    """Cohen's kappa for two raters over the same items. Items where either rater is missing are dropped."""
    try:
        a, b = list(a), list(b)
    except TypeError:
        return _res(None, "not_sequences", 0)
    if len(a) != len(b):
        return _res(None, "length_mismatch", 0)
    pairs = [(x, y) for x, y in zip(a, b) if not _missing(x) and not _missing(y)]
    n = len(pairs)
    if n < 2:
        return _res(None, "too_few_items", n)
    try:
        cats = sorted({c for p in pairs for c in p}, key=repr)
    except TypeError:
        return _res(None, "unhashable_values", n)
    po = sum(1 for x, y in pairs if x == y) / n
    pe = sum((sum(1 for x, _ in pairs if x == c) / n) * (sum(1 for _, y in pairs if y == c) / n) for c in cats)
    out = _res(None, None, n)
    out.update(observed=po, expected=pe, dropped_missing=len(a) - n)
    if pe >= 1.0 - 1e-12:
        out["reason"] = "no_variation"  # kappa is undefined, never 1.0
        return out
    out["kappa"] = (po - pe) / (1.0 - pe)
    return out


def _res(kappa, reason, n):
    return {"kappa": kappa, "reason": reason, "n": n, "observed": None, "expected": None, "dropped_missing": 0}


def krippendorff_alpha_nominal(units) -> dict:
    """Krippendorff's alpha (nominal). ``units`` is a list of units, each a list of rater values (None = missing).

    alpha = 1 - (n - 1) * sum_{c != k} o_ck / sum_{c != k} n_c * n_k, with the coincidence matrix built from every
    ordered pair of values in a unit weighted 1 / (m_u - 1). Units with fewer than two values are not pairable.
    """
    try:
        units = [[v for v in u if not _missing(v)] for u in units]
    except TypeError:
        return {"alpha": None, "reason": "not_sequences", "n_pairable": 0}
    units = [u for u in units if len(u) >= 2]
    n = sum(len(u) for u in units)
    if n < 2:
        return {"alpha": None, "reason": "too_few_pairable_values", "n_pairable": n}
    o = {}
    for u in units:
        w = 1.0 / (len(u) - 1)
        for i, c in enumerate(u):
            for j, k in enumerate(u):
                if i != j:
                    o[(c, k)] = o.get((c, k), 0.0) + w
    cats = sorted({c for c, _ in o} | {k for _, k in o}, key=repr)
    nc = {c: sum(o.get((c, k), 0.0) for k in cats) for c in cats}
    disagree = sum(v for (c, k), v in o.items() if c != k)
    expected = sum(nc[c] * nc[k] for c in cats for k in cats if c != k)
    if expected <= 1e-12:
        return {"alpha": None, "reason": "no_variation", "n_pairable": n}  # undefined, never 1.0
    return {"alpha": 1.0 - (n - 1) * disagree / expected, "reason": None, "n_pairable": n}


def _percentile(sorted_vals, q):
    k = (len(sorted_vals) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def bootstrap_by_participant(items_by_participant: dict, stat_fn, n_draws: int = N_DRAWS, seed: int = SEED) -> dict:
    """95% percentile bootstrap that resamples PARTICIPANTS (never windows) and recomputes ``stat_fn(items)``.

    ``items_by_participant`` maps a participant token to a list of items; ``stat_fn`` gets the concatenated items
    and returns a float or None. Draws where the statistic is undefined are skipped and counted.
    """
    out = {"estimate": None, "lo": None, "hi": None, "n_valid": 0, "n_participants": 0, "reason": None}
    if not isinstance(items_by_participant, dict) or not callable(stat_fn):
        out["reason"] = "invalid_arguments"
        return out
    try:
        parts = sorted(items_by_participant, key=str)
        data = {p: list(items_by_participant[p]) for p in parts}
    except TypeError:
        out["reason"] = "invalid_arguments"
        return out

    def safe(items):
        try:
            v = stat_fn(items)
        except Exception:  # a raising statistic is undefined, not fatal
            return None
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None

    out["estimate"] = safe([it for p in parts for it in data[p]])
    out["n_participants"] = len(parts)
    if len(parts) < 2:
        out["reason"] = "fewer_than_2_participants"
        return out
    rng = random.Random(seed)
    vals = []
    for _ in range(n_draws):
        items = []
        for _ in parts:
            items.extend(data[parts[rng.randrange(len(parts))]])
        v = safe(items)
        if v is not None:
            vals.append(v)
    out["n_valid"] = len(vals)
    if vals:
        vals.sort()
        out["lo"], out["hi"] = _percentile(vals, 0.025), _percentile(vals, 0.975)
    else:
        out["reason"] = "statistic_undefined_in_every_draw"
    return out


def band(kappa):
    """Pre-registered reading (conventional rules of thumb, not a standard)."""
    if kappa is None:
        return None
    return "substantial_enough_to_use" if kappa >= 0.60 else "moderate_report_do_not_rely" if kappa >= 0.40 else "unreliable_drop_from_primary"


WITHHELD = "fewer_than_20_positive_windows"


def gate_alpha(res: dict, positives_total: int) -> dict:
    """Withhold alpha below 20 positive windows (protocol 4.5); keep the raw value under a clearly named key."""
    out = dict(res)
    out["positives_total"] = positives_total
    if positives_total < MIN_POSITIVES_FOR_INTERPRETATION:
        out["raw_alpha_do_not_interpret"] = res.get("alpha")
        out["raw_alpha_reason"] = res.get("reason")
        out["alpha"], out["reason"] = None, WITHHELD
    return out


def _empty_agreement(reason) -> dict:
    return {"n": 0, "table": {"both_present": 0, "only_first": 0, "only_second": 0, "both_absent": 0}, "positives_total": 0,
            "observed_agreement": None, "prevalence_index": None, "bias_index": None, "pabak": None, "kappa": None,
            "kappa_reason": reason, "raw_kappa_do_not_interpret": None, "raw_kappa_reason": reason,
            "interpretation": "undefined", "band": None}


def label_agreement(a, b, positive="present") -> dict:
    """Per-label agreement for two annotators: table, observed %, prevalence, bias, PABAK, kappa, band.

    Below 20 positive windows (summed over annotators) the protocol allows counts and raw agreement only, so ``kappa``
    is None with reason ``fewer_than_20_positive_windows``; the computed value stays under ``raw_kappa_do_not_interpret``.
    Non-sequence input returns an empty summary with a reason instead of raising.

    Binary labels only: complete pairs must contain hashable categories, at
    most two distinct categories, and a two-category set must include positive.
    Otherwise refuse binary_labels_only (unhashable_values for unhashable input).
    All exported statistics use the same complete pairs normalised to booleans;
    use cohen_kappa directly for general multi-category agreement.
    """
    try:
        a, b = list(a), list(b)
    except TypeError:
        return _empty_agreement("not_sequences")
    if len(a) != len(b):
        return _empty_agreement("length_mismatch")
    pairs = [(x, y) for x, y in zip(a, b) if not _missing(x) and not _missing(y)]
    try:
        hash(positive)
        categories = {v for pair in pairs for v in pair}
    except TypeError:
        out = _empty_agreement("unhashable_values")
        out["n"] = len(pairs)
        return out
    if len(categories) > 2 or (len(categories) == 2 and positive not in categories):
        out = _empty_agreement("binary_labels_only")
        out["n"] = len(pairs)
        return out
    try:
        pairs = [(bool(x == positive), bool(y == positive)) for x, y in pairs]
    except Exception:
        return _empty_agreement("incomparable_values")
    k = cohen_kappa([x for x, _ in pairs], [y for _, y in pairs])
    n = len(pairs)
    try:
        tp = sum(1 for x, y in pairs if x and y)
        fn = sum(1 for x, y in pairs if x and not y)
        fp = sum(1 for x, y in pairs if not x and y)
    except Exception:
        return _empty_agreement("incomparable_values")
    tn = n - tp - fn - fp
    positives = 2 * tp + fn + fp  # summed over both annotators
    out = {"n": n, "table": {"both_present": tp, "only_first": fn, "only_second": fp, "both_absent": tn},
           "positives_total": positives,
           "observed_agreement": None if n == 0 else (tp + tn) / n,
           "prevalence_index": None if n == 0 else abs(tp - tn) / n,
           "bias_index": None if n == 0 else abs(fn - fp) / n,
           "pabak": None if n == 0 else 2 * ((tp + tn) / n) - 1,
           "kappa": k["kappa"], "kappa_reason": k["reason"],
           "raw_kappa_do_not_interpret": k["kappa"], "raw_kappa_reason": k["reason"]}
    if positives < MIN_POSITIVES_FOR_INTERPRETATION:
        out["interpretation"] = "counts_only_fewer_than_20_positive_windows"
        out["band"] = None
        out["kappa"], out["kappa_reason"] = None, WITHHELD
    else:
        out["interpretation"] = "interpretable" if k["kappa"] is not None else "undefined"
        out["band"] = band(k["kappa"])
    return out
