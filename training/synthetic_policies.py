"""SYNTHETIC / NOT REAL MODEL PERFORMANCE; fixed causal policy experiments.

In-memory events only. No file/model/UI imports or production setting changes.
No fitting: alpha=.5, raw/EMA and thresholds off/.5/.7/.9 are predeclared.
Coverage uses active frames including missing faces; selective error is null
when nothing is accepted. Unsupported scores display None, never supported zero.
Flicker is not counted while settling after a truth step (until first accepted
correct decision); if settling never ends, all such switches remain uncounted.
Abstention breaks adjacency, so flicker must not be compared across thresholds.
settling_never_ended_count counts censored truth-step segments, not frames, and
is an explicit alias of unresolved_transition_count. Truth-aware metrics do not
change causal predictions or EMA.
"""
import numpy as np
from training.synthetic_methods import LABELS, MARKER, SEED, MAX_ROWS, marked, validated_probabilities

ALPHA = .5
SMOOTHING = ("none", "ema_alpha_0.5")
THRESHOLDS = ("off", .5, .7, .9)
FLICKER_CONVENTION = ("Flicker is not counted while settling after a truth step until the first accepted correct decision; "
                      "settling may never end. Abstention breaks adjacency. Do not compare flicker across thresholds. "
                      "settling_never_ended_count counts truth-step segments without an accepted correct decision before censoring, not frames.")
DISTRIBUTION_CONSTRUCTION = "holds by construction (confidence tied to correctness)"
LIMITATION = ("These manufactured streams/distributions measure causal smoothing, resets, flicker, transition lag, coverage and selective-error tradeoffs for fixed policies. "
              "They cannot show real accuracy, calibration, fairness, camera robustness or an appropriate real decision threshold. "
              "Missing/identity/run/stop/support resets are simulation rules only; no production evidence threshold or policy is changed. "
              + FLICKER_CONVENTION + " Distribution result " + DISTRIBUTION_CONSTRUCTION + "; it is not evidence that confidence predicts correctness on real data.")


def _checked_events(events):
    if type(events) is not list or not 1 <= len(events) <= MAX_ROWS:
        raise ValueError("bounded explicit event list required")
    checked = []
    for event in events:
        if type(event) is not dict or set(event) != {"run", "identity", "active", "probabilities", "truth", "supported_classes"}:
            raise ValueError("exact synthetic event structure required")
        for key in ("run", "identity"):
            if not isinstance(event[key], str) or not event[key].startswith("SYNTHETIC-") or len(event[key]) > 160:
                raise ValueError("bounded manufactured run/identity required")
        if type(event["active"]) is not bool or event["truth"] not in LABELS:
            raise ValueError("active boolean and named synthetic truth required")
        support = event["supported_classes"]
        if (type(support) is not list or not support or len(support) > 7 or
                any(type(x) is not str or x not in LABELS for x in support) or len(set(support)) != len(support)):
            raise ValueError("unique named supported classes required")
        raw = event["probabilities"]
        if not event["active"] and raw is not None: raise ValueError("stopped event cannot carry scores")
        p = None if raw is None else validated_probabilities([raw])[0]
        if p is not None and (len(p) != 7 or any(p[i] != 0 for i, name in enumerate(LABELS) if name not in support)):
            raise ValueError("FER7 width and zero mass on unsupported classes required")
        checked.append((event, p, (event["run"], event["identity"], tuple(sorted(support)))))
    return checked


def simulate(events, *, smoothing="none", threshold="off"):
    """Causal decisions. First-index LABELS ties match numpy.argmax.

    EMA is reset on missing face, identity/run/support change or stop. Rejected
    scores do not reset EMA; they do interrupt the consecutive-decision flicker
    measure. Initial catch-up after each truth step is excluded from flicker
    until the first accepted correct decision (that catch-up is measured as lag).
    These truth-aware metrics never affect predictions or EMA. Lag is an offline
    descriptive metric ONLY, never used by decisions:
    frames from a truth step until the first accepted correct decision, censored
    by next truth step/missing face/reset/end. Unreached steps have null lag.
    """
    if smoothing not in SMOOTHING: raise ValueError("undeclared smoothing policy")
    if not (threshold == "off" if isinstance(threshold, str) else type(threshold) in (int, float) and threshold in THRESHOLDS[1:]):
        raise ValueError("undeclared decision threshold")
    checked = _checked_events(events)
    state = None; last_key = None; previous = None; last_truth = None; settling = False
    rows = []; accepted = mistakes = flicker = active = missing = 0
    for index, (event, p, key) in enumerate(checked):
        active += int(event["active"])
        if key != last_key or p is None or not event["active"]:
            state = None; previous = None; last_truth = None; settling = False
        last_key = key
        if p is None or not event["active"]:
            missing += int(event["active"])
            rows.append(marked(index=index, state="missing-face" if event["active"] else "stopped",
                               prediction=None, confidence=None, scores=None))
            continue
        if last_truth is not None and last_truth != event["truth"]: settling = True
        last_truth = event["truth"]
        state = p.copy() if state is None or smoothing == "none" else ALPHA * p + (1 - ALPHA) * state
        prediction = LABELS[int(state.argmax())]
        confidence = float(state.max())
        accept = threshold == "off" or confidence >= threshold
        if accept:
            accepted += 1; mistakes += int(prediction != event["truth"])
            if previous is not None and previous[1] == event["truth"] and not settling:
                flicker += int(previous[0] != prediction)
            previous = (prediction, event["truth"])
            if prediction == event["truth"]: settling = False
        else: previous = None
        rows.append(marked(index=index, state="accepted" if accept else "abstained",
                           prediction=prediction if accept else None, confidence=confidence,
                           scores=marked(rows=[marked(label=name, value=float(state[i]) if name in event["supported_classes"] else None,
                                                      support="supported" if name in event["supported_classes"] else "unsupported") for i, name in enumerate(LABELS)])))
    transitions = []
    for i in range(1, len(checked)):
        event, p, key = checked[i]; before, before_p, before_key = checked[i - 1]
        if (p is None or before_p is None or not event["active"] or not before["active"]
                or key != before_key or event["truth"] == before["truth"]): continue
        lag = None
        for j in range(i, len(checked)):
            later, later_p, later_key = checked[j]
            if later_key != key or later_p is None or not later["active"] or later["truth"] != event["truth"]: break
            if rows[j]["prediction"] == event["truth"]:
                lag = j - i; break
        transitions.append(marked(index=i, lag_frames=lag))
    reached = [r["lag_frames"] for r in transitions if r["lag_frames"] is not None]
    return marked(smoothing=smoothing, alpha=ALPHA if smoothing != "none" else None, threshold=threshold,
                  tie_rule="first class in LABELS via np.argmax", rows=rows,
                  metrics=marked(active_denominator=active, accepted=accepted, mistakes=mistakes,
                                 coverage=accepted / active if active else None,
                                 selective_error=mistakes / accepted if accepted else None,
                                 missing_face=missing, stopped=len(events) - active, flicker_count=flicker,
                                 transition_count=len(transitions), reached_transition_count=len(reached),
                                 unresolved_transition_count=len(transitions) - len(reached),
                                 settling_never_ended_count=len(transitions) - len(reached),
                                 mean_reached_transition_lag=float(np.mean(reached)) if reached else None),
                  transitions=marked(rows=transitions))


def _event(p, truth="angry", *, run="SYNTHETIC-runA", identity="SYNTHETIC-actorA", active=True, support=None):
    return {"run": run, "identity": identity, "active": active, "probabilities": p,
            "truth": truth, "supported_classes": list(LABELS if support is None else support)}


def generate_stream():
    """Fixed scenario list is declared before RNG505 draws, not selected by result."""
    rng = np.random.default_rng(SEED)
    events = []
    # Stable truth with manufactured score jitter, then a step, then each reset.
    for _ in range(40):
        p = float(rng.uniform(.35, .9)); events.append(_event([p, 0, 0, 1-p, 0, 0, 0]))
    for _ in range(20): events.append(_event([.1, 0, 0, .9, 0, 0, 0], "happy"))
    events.extend([_event(None, "happy"), _event([.2, 0, 0, .8, 0, 0, 0], "happy"),
                   _event([.9, 0, 0, .1, 0, 0, 0], identity="SYNTHETIC-actorB"),
                   _event(None, active=False, identity="SYNTHETIC-actorB"),
                   _event([.1, 0, 0, .9, 0, 0, 0], "happy", run="SYNTHETIC-runB"),
                   _event([.8, 0, 0, .2, 0, 0, 0], support=["angry", "happy"])])
    return events


def generate_distribution():
    """400 manufactured independent identities: 200 correct .95, 200 wrong .85.

    With threshold .9 and no smoothing, selective error is exactly0, coverage.5;
    off/.5/.7 retain every wrong score. Identity changes prohibit EMA leakage.
    """
    rng = np.random.default_rng(SEED)
    kinds = rng.permutation(np.repeat([0, 1], 200))
    return [_event([.95, 0, 0, .05, 0, 0, 0] if kind == 0 else [.15, 0, 0, .85, 0, 0, 0],
                   identity=f"SYNTHETIC-independent-{i}") for i, kind in enumerate(kinds)]


def run_experiment():
    # Policy grid constants were fixed at import, before either generator.
    stream = generate_stream(); distribution = generate_distribution()
    tables = []
    for name, events in (("stream", stream), ("distribution", distribution)):
        rows = []
        for smoothing in SMOOTHING:
            for threshold in THRESHOLDS:
                result = simulate(events, smoothing=smoothing, threshold=threshold)
                rows.append(marked(smoothing=smoothing, threshold=threshold, metrics=result["metrics"], transitions=result["transitions"]))
        tables.append(marked(scenario=name, input_rows=len(events), rows=rows,
                             limitation=FLICKER_CONVENTION if name == "stream" else DISTRIBUTION_CONSTRUCTION))
    return marked(experiment="E4 synthetic temporal/decision/abstention policies", total_rows=len(stream)+len(distribution),
                  policies=marked(smoothing=list(SMOOTHING), alpha=ALPHA, thresholds=list(THRESHOLDS)),
                  comparison=marked(rows=tables), bootstrap_replicates=0, limitation=LIMITATION)
