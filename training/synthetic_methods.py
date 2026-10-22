"""SYNTHETIC / NOT REAL MODEL PERFORMANCE; in-memory calibration mechanics.

No CLI, file reads/writes, adapters or deployment configuration. Grids, methods,
seed and actor roles are fixed before outcomes. The fit role is reserved (no
model is fitted); only validation selects T; assessment is descriptive only.
Existing calibrate_temp primitives are reused without changing their semantics.
Pure numerical helpers accept caller arrays, not authenticated/admitted data.
"""
import numpy as np
from training import calibrate_temp as calibration

MARKER = "SYNTHETIC / NOT REAL MODEL PERFORMANCE"
SEED = 505
MAX_ROWS = 10000
LABELS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
GRID = (.25, .5, 1., 2., 4.)
METHODS = ("identity_T1", "validation_nll_grid", "validation_brier_grid", "legacy_validation_nll_200_grid")
ECE10_CONVENTION = "10 equal-width bins (lower, upper]; lower exclusive, upper inclusive. This reuse convention differs from the DESIGN protocol left-closed/right-open bins (last includes 1)."
LIMITATION = ("This manufactured overconfidence experiment checks numerical selection and held-out scoring mechanics under one generator. "
              "It cannot establish real model accuracy, calibration, fairness, camera robustness or an appropriate real temperature. "
              "Scalar positive temperature preserves top-1 except numerical/tie artifacts; no winner is deployed. " + ECE10_CONVENTION)


def marked(**values):
    return {"synthetic_only": True, "exposure": MARKER, "seed": SEED, **values}


def bounded_matrix(value, *, allow_empty=False):
    try:
        raw = np.asarray(value)
        if any(isinstance(x, (bool, np.bool_)) for x in np.asarray(value, dtype=object).flat):
            raise ValueError("booleans are not numeric scores")
        if raw.dtype.kind not in "iuf": raise ValueError("real numeric matrix required")
        result = raw.astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("real numeric matrix required") from exc
    if (result.ndim != 2 or not 2 <= result.shape[1] <= 7 or
            not (0 if allow_empty else 1) <= len(result) <= MAX_ROWS or not np.isfinite(result).all()):
        raise ValueError("finite bounded matrix required")
    return result


def validated_probabilities(value, *, allow_empty=False):
    p = bounded_matrix(value, allow_empty=allow_empty)
    if np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(1), 1, rtol=0, atol=1e-6):
        raise ValueError("unit probabilities required; no renormalisation")
    return p


def validated_labels(value, n, width):
    raw = np.asarray(value)
    if raw.ndim != 1 or len(raw) != n or (n and raw.dtype.kind not in "iu"):
        raise ValueError("integer labels matching rows required")
    if n and (np.any(raw < 0) or np.any(raw >= width)):
        raise ValueError("label outside class space")
    return raw.astype(int)


def probabilities(logits, temperature=1.):
    z = bounded_matrix(logits)
    if isinstance(temperature, (bool, np.bool_)) or not isinstance(temperature, (int, float, np.floating)) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("positive finite temperature required")
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = z / temperature
    if not np.isfinite(scaled).all(): raise ValueError("scaled logits overflow")
    with np.errstate(over="ignore", invalid="ignore"):
        p = calibration.apply_temperature(z, temperature)
    return validated_probabilities(p)


def metrics(proba, labels):
    """Multiclass Brier=sum of squared class errors; NLL clips at 1e-12.

    ECE reuses the existing ten-bin (lower exclusive, upper inclusive) helper.
    Empty selected cohorts are explicitly null, unlike legacy abstention_curve.
    """
    p = validated_probabilities(proba, allow_empty=True)
    y = validated_labels(labels, len(p), p.shape[1])
    if not len(p): return marked(n=0, nll=None, brier=None, ece10=None, error=None)
    targets = np.eye(p.shape[1])[y]
    return marked(n=len(p), nll=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
                  brier=float(np.square(p - targets).sum(1).mean()),
                  ece10=calibration.ece_from_proba(p, y, n_bins=10),
                  error=float((p.argmax(1) != y).mean()))


def select_temperatures(validation_logits, validation_labels):
    """Only these validation arrays are passed to either objective/fitter.

    Common-grid ties pick the first declared temperature. Legacy fitter has its
    OWN .05..5/200 grid and log(p+1e-12) objective, explicitly named separately.
    """
    z = bounded_matrix(validation_logits)
    y = validated_labels(validation_labels, len(z), z.shape[1])
    candidates = [(t, metrics(probabilities(z, t), y)) for t in GRID]
    nll = min(candidates, key=lambda item: item[1]["nll"])[0]
    brier = min(candidates, key=lambda item: item[1]["brier"])[0]
    legacy = calibration.fit_temperature(z, y)
    return marked(common_grid=list(GRID), common_grid_tie_rule="first declared temperature",
                  legacy_grid="np.linspace(.05,5,200); existing log(p+1e-12) NLL",
                  rows=[marked(method=name, temperature=float(t)) for name, t in
                        zip(METHODS, (1., nll, brier, legacy))])


def assign_roles():
    """Split actor IDs BEFORE logits/labels, without outcome stratification."""
    actors = np.random.default_rng(np.random.SeedSequence(SEED).spawn(2)[0]).permutation(60)
    return {role: tuple(int(x) for x in actors[start:start + 20])
            for role, start in (("fit", 0), ("validation", 20), ("assessment", 40))}


def generate_cohort():
    roles = assign_roles()  # roles fixed before generating ANY outcome
    rng = np.random.default_rng(np.random.SeedSequence(SEED).spawn(2)[1])
    latent_actor = rng.normal(0, .7, (60, 7))
    latent = np.repeat(latent_actor, 24, axis=0) + rng.normal(0, .8, (1440, 7))
    true_p = probabilities(latent)
    uniforms = rng.random(1440)
    labels = (uniforms[:, None] > true_p.cumsum(1)).sum(1).clip(max=6)
    observed = 3 * latent  # known manufactured overconfidence
    actor_ids = np.repeat(np.arange(60), 24)
    return marked(roles={role: marked(actor_ids=list(ids), logits=observed[np.isin(actor_ids, ids)].copy(),
                                    labels=labels[np.isin(actor_ids, ids)].copy()) for role, ids in roles.items()})


def run_experiment():
    cohort = generate_cohort()
    validation = cohort["roles"]["validation"]
    selected = select_temperatures(validation["logits"], validation["labels"])
    assessment = cohort["roles"]["assessment"]
    rows = [marked(method=item["method"], temperature=item["temperature"],
                   metrics=metrics(probabilities(assessment["logits"], item["temperature"]), assessment["labels"]))
            for item in selected["rows"]]
    return marked(experiment="E3 synthetic calibration comparison", total_rows=sum(len(role["labels"]) for role in cohort["roles"].values()),
                  generator="60 latent actors x24 categorical rows; logits scaled by3; independent child streams of root seed505; role split before outcomes",
                  roles=marked(rows=[marked(role=role, actor_ids=data["actor_ids"], rows=len(data["labels"]))
                                     for role, data in cohort["roles"].items()]),
                  selection=selected, assessment=marked(rows=rows), ece10_convention=ECE10_CONVENTION, bootstrap_replicates=0,
                  limitation=LIMITATION)
