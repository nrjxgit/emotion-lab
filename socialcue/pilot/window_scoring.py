"""Align HB-1 windows to annotated windows and score them (EL-006 S15; pilot protocol sections 4-5). Pure stdlib.

``score_pilot(adapter, ...)`` runs the rights gate on the ADAPTER's declarations only. The streams and annotations are
supplied by the caller (the synthetic adapter has no pilot items to hand over), so the gate does not by itself bind the
data. Two guards stand in for that, and neither is a proof: (1) under the synthetic adapter every participant code must
be in the reserved synthetic form (``PT-5eed`` + 4 hex), so real-looking codes are refused and cannot receive the
SYNTHETIC banner; (2) the results carry a hash of the exact data scored, bound to the provenance manifest hash. The
shipped ``pilot_consented`` adapter is a blocked declaration that also needs an ``ethics_approval_token`` string; a
string is not an approval and nothing here verifies one, so the pilot cannot be scored without the operator's decision.
Today only the synthetic adapter passes the gate, and every report from it is labelled SYNTHETIC / NOT REAL MODEL PERFORMANCE.

KNOWN LIMITATIONS: FINAL-opened-once and a frozen protocol hash are not enforced (recorded as
``final_opened_once_enforced: false`` / ``protocol_hash: null``); AUROC, ECE and the interaction-log comparator are not
implemented; the state mapping below is a hypothesis registered in code, not a validated construct.
"""
from __future__ import annotations

import copy
import math

from .. import state as hb1
from ..data.provenance import build_manifest
from ..data.rights import DatasetAdapter, authorize, sha256_of
from ..eval.proxy_eval import actor_disjoint_split, assert_actor_disjoint, confusion, metrics
from ..eval.report import ALLOWED_CLAIMS as EVAL_ALLOWED_CLAIMS
from ..eval.report import CAVEAT, SYNTHETIC_LABEL, ReportIncomplete
from . import agreement as G
from . import annotations as A

PILOT_LABEL = "PILOT-DRAFT"
USE = "research_evaluation"
MIN_PARTICIPANTS_FOR_SPLIT = 24
MIN_POS_PARTICIPANTS, MIN_POS_WINDOWS = 10, 30
HB_WINDOW, HB_HOP = 3.0, 1.0
MAX_DENSE_ENDPOINTS = 20000  # above this many scheduled endpoints the bounded sparse rule below applies

# Analysis-time mapping from protocol 4.2: registered before any scoring; a hypothesis, not a validated construct.
STATE_MAP = {
    "nervous-like": ("fidget", "look_away"),
    "unsure-like": ("pause", "look_away"),
    "needs-help-like": ("asks_help", "or sustained pause+look_away over >= 2 consecutive windows"),
}
LABEL_STATES = {"fidget": ("nervous-like",), "look_away": ("nervous-like", "unsure-like"), "pause": ("unsure-like",),
                "asks_help": ("needs-help-like",)}
TARGETS = tuple(f"raw:{l}" for l in A.LABELS) + tuple(f"state:{s}" for s in STATE_MAP)
ALIGNMENT_RULE = ("Each HB-1 window (t1-3, t1] (hop 1 s) inherits the annotated 3 s block that contains at least 50% of it "
                  "(ties go to the earlier block); windows with no such block are dropped. Unscorable blocks are excluded and counted.")

PILOT_LIMITATIONS = (EVAL_ALLOWED_CLAIMS[0] + " Volunteer sample, scripted mock tasks, small N, single site. "
                     "The rights gate authorises the adapter's declaration, not the data handed to the scorer; only the synthetic-code check "
                     "and the data hash stand in for that binding. Kappa and alpha are withheld below 20 positive windows.")
PILOT_ALLOWED_CLAIMS = (
    "In a pilot of N consenting adult volunteers on mock kiosk/reception tasks, an unvalidated heuristic (HB-1) agreed with "
    "independent human annotations of observable behaviour at [metric, 95% participant-bootstrap CI]; inter-annotator agreement "
    "was [kappa, CI].",
    "Descriptive base rates and pipeline floors from the pilot.",
)
PILOT_DISALLOWED_CLAIMS = (
    "Any claim that HB-1 detects or reads nervousness, uncertainty, need for help, feelings or intent.",
    "Any real-world (POS/reception) efficacy claim, and any population or fairness statement.",
    "Pooling tier A and tier B without separate rows; any commercial use of the data or models.",
    "Comparing to acted-dataset numbers as if equivalent; dropping unfavourable labels; presenting self-report as a label.",
    "Any statement about an individual participant.",
)


class PilotInputError(ValueError):
    """Malformed arguments to score_pilot (documented error type; ``codes`` lists the reasons)."""

    def __init__(self, codes):
        super().__init__("; ".join(codes))
        self.codes = list(codes)


def _data_hash(streams: dict, recs: list) -> str:
    """Hash of the exact data scored (participant streams and validated annotations), independent of dict order."""
    body = {"streams": {p: [(r.t, r.quality, r.arousal, r.valence, repr(r.box), repr(r.landmarks), repr(r.face_count),
                             repr(sorted(r.expression.items())) if isinstance(r.expression, dict) else repr(r.expression))
                            for r in sorted((x for x in v if isinstance(x, hb1.FrameRecord)), key=lambda x: repr(x.t))] for p, v in sorted(streams.items())},
            "annotations": sorted(repr(sorted((k, repr(v)) for k, v in r.items())) for r in recs)}
    return sha256_of(repr(body))


def _hb_windows(records, cfg=None, blocks=None):
    """Scheduled HB-1 outputs on (t0, t1], hop 1 s; retain earlier records for dwell.

    Endpoints are first+W, first+W+hop, ... up to the last record. Ordinary streams (at most MAX_DENSE_ENDPOINTS scheduled
    endpoints) use that dense schedule unchanged, so their results are byte-identical to the pre-S58 scheduler. A stream
    spanning more endpoints (for example one stray record at t=1e6) would otherwise cost time linear in the span, so the
    schedule is bounded to the endpoints that can matter: (a) endpoints whose window (t1-W, t1] holds at least one record
    (evidence), and (b) endpoints inside the alignment reach of an annotated block (t1 from block start + W/2 to block end
    + W/2, where a window overlaps the block by at least half), which are still emitted and abstain when empty. Work is
    then bounded by 3 x records + 4 x blocks. Endpoints outside both sets cannot align to any block, so scoring is unaffected.
    The sparse endpoint values are first + W + k x hop computed directly.
    """
    recs = sorted((r for r in records if isinstance(r, hb1.FrameRecord) and isinstance(r.t, (int, float)) and math.isfinite(r.t)),
                  key=lambda r: r.t)
    if not recs:
        return []
    base, last = recs[0].t + HB_WINDOW, recs[-1].t
    n_dense = math.floor((last + 1e-9 - base) / HB_HOP) + 1 if last + 1e-9 >= base else 0
    if n_dense <= MAX_DENSE_ENDPOINTS:
        out, t1 = [], base
        while t1 <= last + 1e-9:
            res = hb1.analyze([r for r in recs if r.t <= t1], cfg, evaluation_time=t1)
            out.append((t1 - HB_WINDOW, t1, frozenset(s["name"] for s in res["states"]), bool(res["insufficient"]["flag"])))
            t1 += HB_HOP
        return out
    ks = set()
    for r in recs:  # evidence: t1 in [t, t + W)
        lo, hi = math.ceil((r.t - base) / HB_HOP), math.ceil((r.t + HB_WINDOW - base) / HB_HOP) - 1
        ks.update(range(max(lo, 0), min(hi, n_dense - 1) + 1))
    half = 0.5 * HB_WINDOW
    for b in blocks or ():  # alignment reach of each annotated block
        lo, hi = math.ceil((b["window_start"] + half - 1e-9 - base) / HB_HOP), math.floor((b["window_end"] + half + 1e-9 - base) / HB_HOP)
        ks.update(range(max(lo, 0), min(hi, n_dense - 1) + 1))
    out = []
    for k in sorted(ks):
        t1 = base + k * HB_HOP
        res = hb1.analyze([r for r in recs if r.t <= t1], cfg, evaluation_time=t1)
        out.append((t1 - HB_WINDOW, t1, frozenset(s["name"] for s in res["states"]), bool(res["insufficient"]["flag"])))
    return out


def align(windows, blocks):
    """Pair each HB-1 window with the annotated block holding >= 50% of it. Returns [(window, block_index)]."""
    pairs = []
    for w in windows:
        best = None
        for i, b in enumerate(blocks):
            ov = max(0.0, min(w[1], b["window_end"]) - max(w[0], b["window_start"]))
            if ov >= 0.5 * HB_WINDOW - 1e-9:
                best = i
                break
        if best is not None:
            pairs.append((w, best))
    return pairs


def _truth(target, blocks, i):
    """Truth for a raw label or a mapped state in block i; None if the block is unscorable."""
    b = blocks[i]
    if b["unscorable"]:
        return None
    kind, name = target.split(":")
    lab = b["labels"]
    if kind == "raw":
        return lab[name] == A.PRESENT
    if name == "nervous-like":
        return lab["fidget"] == A.PRESENT or lab["look_away"] == A.PRESENT
    if name == "unsure-like":
        return lab["pause"] == A.PRESENT or lab["look_away"] == A.PRESENT
    if lab["asks_help"] == A.PRESENT:
        return True
    both = lambda x: (not x["unscorable"]) and x["labels"]["pause"] == A.PRESENT and x["labels"]["look_away"] == A.PRESENT  # noqa: E731
    adjacent = lambda x, y: abs(x["window_end"] - y["window_start"]) < 1e-6  # noqa: E731
    if both(b):
        if i > 0 and adjacent(blocks[i - 1], b) and both(blocks[i - 1]):
            return True
        if i + 1 < len(blocks) and adjacent(b, blocks[i + 1]) and both(blocks[i + 1]):
            return True
    return False


def _pred(target, states):
    kind, name = target.split(":")
    if kind == "state":
        return name in states
    return any(s in states for s in LABEL_STATES[name])


def _bacc(items):
    return metrics(confusion(items))["balanced_accuracy"]


def _score_view(outcomes, n_draws, seed):
    """outcomes: [(participant, truth, 'pos'|'neg'|'abstain')] -> metrics, counts, CI, evaluability, no-cue comparator."""
    positives = [o for o in outcomes if o[1]]
    pos_participants = len({o[0] for o in positives})
    evaluable = pos_participants >= MIN_POS_PARTICIPANTS and len(positives) >= MIN_POS_WINDOWS
    full, ans = confusion(outcomes), confusion(outcomes, "answered_only")
    no_cue = [(p, t, "neg") for p, t, _ in outcomes]
    by_p = {}
    for o in outcomes:
        by_p.setdefault(o[0], []).append(o)
    return {
        "n_windows": len(outcomes), "n_positive_windows": len(positives), "n_participants": len(by_p),
        "n_participants_with_positive": pos_participants, "n_abstained": full["abstained"],
        "evaluable": evaluable, "not_evaluable_reason": None if evaluable else "not_evaluable_at_this_n",
        "hb1": {"full_cohort": {"confusion": full, **metrics(full)}, "answered_only": {"confusion": ans, **metrics(ans)},
                "ci_balanced_accuracy_by_participant": G.bootstrap_by_participant(by_p, _bacc, n_draws, seed)},
        "always_no_cue": {"full_cohort": {"confusion": confusion(no_cue), **metrics(confusion(no_cue))}},
    }


def score_pilot(adapter, run_scope, streams, annotations, *, seed=G.SEED, n_draws=G.N_DRAWS, cfg=None, use=USE) -> dict:
    """Score HB-1 against window annotations after the rights gate. ``streams``: {participant: [FrameRecord]}."""
    if not isinstance(adapter, DatasetAdapter):
        raise TypeError("score_pilot() takes a DatasetAdapter only")
    authorize(adapter, use, run_scope)
    codes = []
    if not isinstance(streams, dict):
        codes.append("streams_must_be_a_dict")
    elif any(not isinstance(v, (list, tuple)) for v in streams.values()):
        codes.append("each_stream_must_be_a_list")
    if annotations is None or isinstance(annotations, (str, bytes, dict)):
        codes.append("annotations_must_be_a_list")
    if codes:
        raise PilotInputError(codes)
    recs = A.validate_batch(annotations)
    synthetic = adapter.source_class == "synthetic"
    seen = {r["participant"] for r in recs} | {p for p in streams if isinstance(p, str)}
    if synthetic and not all(A.is_synthetic_participant(p) for p in seen):
        raise PilotInputError(["real_looking_participant_codes_under_synthetic_adapter"])
    if not synthetic and any(A.is_synthetic_participant(p) for p in seen):
        raise PilotInputError(["synthetic_participant_codes_under_non_synthetic_adapter"])
    annotators = sorted({r["annotator"] for r in recs})
    participants = sorted({r["participant"] for r in recs})
    scored = [p for p in participants if p in streams]
    if len(scored) >= MIN_PARTICIPANTS_FOR_SPLIT:
        split = actor_disjoint_split(scored, seed)
        assert_actor_disjoint(split, expected=scored)
        exploratory = False
    else:
        split, exploratory = {"ALL": list(scored)}, True
    role_of = {p: r for r, ps in split.items() for p in ps}
    blocks = {(r["annotator"], r["participant"]): [] for r in recs}
    for r in recs:
        blocks[(r["annotator"], r["participant"])].append(r)
    for v in blocks.values():
        v.sort(key=lambda x: x["window_start"])
    hbw = {p: _hb_windows(streams[p], cfg, [b for (_, pp), bl in blocks.items() if pp == p for b in bl]) for p in scored}

    # HB-1 outputs aligned to each annotator's blocks
    aligned = {(a, p): align(hbw[p], blocks[(a, p)]) for (a, p) in blocks if p in hbw}
    unscorable = {a: sum(1 for r in recs if r["annotator"] == a and r["unscorable"]) for a in annotators}

    def view_outcomes(target, view, role):
        out = []
        for p in split[role]:
            if view == "consensus":
                if len(annotators) < 2 or any((a, p) not in aligned for a in annotators):
                    continue
                lead = aligned[(annotators[0], p)]
                for w, i in lead:
                    truths = []
                    for a in annotators:
                        bi = next((j for ww, j in aligned[(a, p)] if ww[0] == w[0]), None)
                        truths.append(None if bi is None else _truth(target, blocks[(a, p)], bi))
                    if any(t is None for t in truths) or len(set(truths)) != 1:
                        continue
                    out.append((p, truths[0], "abstain" if w[3] else ("pos" if _pred(target, w[2]) else "neg")))
            elif (view, p) in aligned:
                for w, i in aligned[(view, p)]:
                    t = _truth(target, blocks[(view, p)], i)
                    if t is not None:
                        out.append((p, t, "abstain" if w[3] else ("pos" if _pred(target, w[2]) else "neg")))
        return out

    views = list(annotators) + (["consensus"] if len(annotators) >= 2 else [])
    scoring = {t: {v: {role: _score_view(view_outcomes(t, v, role), n_draws, seed) for role in split} for v in views} for t in TARGETS}

    # First-two agreement is independent of later raters; alpha retains missing ratings.
    agreement = {}
    for lab in A.LABELS:
        entry = {"n_annotators": len(annotators)}
        keys, pair_keys, by_p = [], [], {}
        for p in participants:
            by_start = {}
            for a in annotators:
                for b in blocks.get((a, p), []):
                    by_start.setdefault(round(b["window_start"], 6), {})[a] = b
            for st, row in sorted(by_start.items()):
                values = [row[a]["labels"][lab] if a in row and not row[a]["unscorable"] else None for a in annotators]
                if sum(v is not None for v in values) >= 2:
                    keys.append((p, st, values))
                if len(values) >= 2 and values[0] is not None and values[1] is not None:
                    pair_keys.append((p, st, values[:2]))
        if len(annotators) >= 2:
            first = [k[2][0] for k in pair_keys]
            second = [k[2][1] for k in pair_keys]
            entry["pair"] = G.label_agreement(first, second)
            for p in participants:
                by_p[p] = [(k[2][0], k[2][1]) for k in pair_keys if k[0] == p]
            by_p = {p: values for p, values in by_p.items() if values}
            if entry["pair"]["positives_total"] < G.MIN_POSITIVES_FOR_INTERPRETATION:  # a CI is an interpretation too
                entry["kappa_ci_by_participant"] = {"estimate": None, "lo": None, "hi": None, "n_valid": 0, "n_participants": len(by_p),
                                                    "reason": G.WITHHELD}
            else:
                entry["kappa_ci_by_participant"] = G.bootstrap_by_participant(
                    {p: v for p, v in by_p.items() if v}, lambda items: G.cohen_kappa([x for x, _ in items], [y for _, y in items])["kappa"], n_draws, seed)
        positives = sum(1 for k in keys for v in k[2] if v == A.PRESENT)
        entry["alpha"] = G.gate_alpha(G.krippendorff_alpha_nominal([k[2] for k in keys]), positives)
        agreement[lab] = entry

    counts = {"participants": len(participants), "participants_scored": len(scored), "annotators": len(annotators),
              "blocks": len(recs), "unscorable_blocks": sum(unscorable.values())}
    manifest = build_manifest(adapter, use, run_scope, counts, config_sha256=hb1.config_sha256(hb1.HB1_CONFIG if cfg is None else cfg))
    data_sha256 = _data_hash(streams, recs)
    return {
        "schema": "socialcue-pilot-scoring-v1",
        "data_sha256": data_sha256,
        "bound_manifest_sha256": sha256_of({"manifest": manifest["manifest_sha256"], "data": data_sha256}),
        "banner": SYNTHETIC_LABEL if synthetic else PILOT_LABEL,
        "caveat": CAVEAT,
        "adapter": {"name": adapter.name, "source_class": adapter.source_class, "scope": adapter.scope},
        "run_scope": run_scope, "seed": seed, "n_draws": n_draws,
        "exploratory": exploratory, "splits": split,
        "mapping": {k: list(v) for k, v in STATE_MAP.items()}, "alignment_rule": ALIGNMENT_RULE,
        "counts": counts, "unscorable_by_annotator": unscorable, "views": views,
        "scoring": scoring, "agreement": agreement,
        "final_opened_once_enforced": False, "protocol_hash": None,
        "notes": ["Abstentions count as failures in full_cohort; answered_only is shown separately.",
                  "Windows within a participant are not independent: the effective sample is closer to the number of participants.",
                  "Labels with fewer than 20 positive windows show counts and raw agreement only; scoring targets need >= 10 participants "
                  "with a positive and >= 30 positive windows to be evaluable.",
                  "FINAL-opened-once and a frozen protocol hash are NOT enforced; AUROC, ECE and the interaction-log comparator are not implemented."],
        "manifest": manifest, "config_sha256": manifest["hb1_config_sha256"],
    }


# --- report (conventions of socialcue.eval.report) -------------------------------------------------------------

REQUIRED = ("status", "caveat", "limitations", "allowed_claims", "disallowed_claims", "manifest_sha256", "hb1_config_sha256")
WITHHELD_PREFIXES = ("raw_kappa", "raw_alpha")  # computed but withheld below 20 positives: never serialised in a report


def _strip_withheld(obj):
    """Deep copy without the withheld raw agreement values (keys starting raw_kappa / raw_alpha)."""
    if isinstance(obj, dict):
        return {k: _strip_withheld(v) for k, v in obj.items() if not (isinstance(k, str) and k.startswith(WITHHELD_PREFIXES))}
    if isinstance(obj, (list, tuple)):
        return [_strip_withheld(v) for v in obj]
    return obj


def _has_withheld(obj) -> bool:
    if isinstance(obj, dict):
        return any((isinstance(k, str) and k.startswith(WITHHELD_PREFIXES)) or _has_withheld(v) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return any(_has_withheld(v) for v in obj)
    return False


def build_pilot_report(results: dict) -> dict:
    """Build the report. The in-memory scoring results keep the withheld raw kappa/alpha under ``raw_*_do_not_interpret``
    keys for debugging; they are removed here, so no report output (JSON or Markdown) contains them."""
    results = _strip_withheld(copy.deepcopy(results))
    synthetic = results["adapter"]["source_class"] == "synthetic"
    body = {
        "schema": "socialcue-pilot-report-v1",
        "status": SYNTHETIC_LABEL if synthetic else PILOT_LABEL,
        "caveat": CAVEAT,
        "limitations": PILOT_LIMITATIONS,
        "allowed_claims": list(PILOT_ALLOWED_CLAIMS),
        "disallowed_claims": list(PILOT_DISALLOWED_CLAIMS),
        "manifest_sha256": results["manifest"]["manifest_sha256"],
        "hb1_config_sha256": results["config_sha256"],
        "results": results,
    }
    return {**body, "report_sha256": sha256_of(body)}


def validate_pilot_report(report: dict) -> None:
    missing = [k for k in REQUIRED if not report.get(k)]
    if missing:
        raise ReportIncomplete(f"pilot report is missing: {', '.join(missing)}")
    if report["caveat"] != CAVEAT or report["limitations"] != PILOT_LIMITATIONS \
            or tuple(report["allowed_claims"]) != PILOT_ALLOWED_CLAIMS or tuple(report["disallowed_claims"]) != PILOT_DISALLOWED_CLAIMS:
        raise ReportIncomplete("caveat, limitations and claims blocks must be the verbatim text")
    res = report["results"]
    if _has_withheld(res):
        raise ReportIncomplete("withheld raw agreement values (raw_kappa* / raw_alpha*) must not be serialised in a report")
    synthetic = res["adapter"]["source_class"] == "synthetic"
    want = SYNTHETIC_LABEL if synthetic else PILOT_LABEL
    if report["status"] != want or res.get("banner") != want or res.get("caveat") != CAVEAT:
        raise ReportIncomplete(f"status and banner must be {want!r}")
    if res.get("final_opened_once_enforced") is not False or "protocol_hash" not in res:
        raise ReportIncomplete("results must record the FINAL-once / protocol-hash limitation")
    if res["manifest"]["manifest_sha256"] != report["manifest_sha256"]:
        raise ReportIncomplete("manifest hash does not match the results")
    if res.get("bound_manifest_sha256") != sha256_of({"manifest": res["manifest"]["manifest_sha256"], "data": res.get("data_sha256")}):
        raise ReportIncomplete("the data hash is not bound to the manifest hash")
    if report.get("report_sha256") != sha256_of({k: v for k, v in report.items() if k != "report_sha256"}):
        raise ReportIncomplete("report hash mismatch (edited after build)")


def _f(v):
    return "n/a" if v is None else f"{v:.3f}"


def render_pilot_markdown(report: dict) -> str:
    validate_pilot_report(report)
    r = report["results"]
    lines = [
        f"# {report['status']}", "",
        f"Adapter {r['adapter']['name']} ({r['adapter']['source_class']}); seed {r['seed']}, {r['n_draws']} participant-level bootstrap draws. "
        f"Exploratory: {str(r['exploratory']).lower()}. Participants {r['counts']['participants_scored']}/{r['counts']['participants']} scored, "
        f"{r['counts']['blocks']} annotated blocks, {r['counts']['unscorable_blocks']} unscorable.",
        f"Provenance manifest sha256: `{report['manifest_sha256']}`  ", f"Scored-data sha256: `{r['data_sha256']}` (bound: `{r['bound_manifest_sha256']}`)  ",
        f"HB-1 config sha256: `{report['hb1_config_sha256']}`  ",
        f"Report sha256: `{report['report_sha256']}`", "",
        "## Label-validity caveat", report["caveat"], "",
        "## Required limitations", report["limitations"], "",
        "## Allowed claims", *[f"- {c}" for c in report["allowed_claims"]], "",
        "## Disallowed claims", *[f"- {c}" for c in report["disallowed_claims"]], "",
        "## Registered mapping and alignment (a hypothesis, not a validated construct)",
        *[f"- {k}: {' / '.join(v)}" for k, v in r["mapping"].items()], f"- Alignment: {r['alignment_rule']}", "",
        "## Inter-annotator agreement (first two annotators; counts first)",
        "| label | n | both present | positives | observed | kappa | CI | interpretation |", "|---|---|---|---|---|---|---|---|",
    ]
    for lab, e in r["agreement"].items():
        p = e.get("pair")
        ci = e.get("kappa_ci_by_participant") or {}
        if p:
            lines.append(f"| {lab} | {p['n']} | {p['table']['both_present']} | {p['positives_total']} | {_f(p['observed_agreement'])} | "
                         f"{_f(p['kappa'])} | [{_f(ci.get('lo'))}, {_f(ci.get('hi'))}] | {p['interpretation']} |")
        else:
            lines.append(f"| {lab} | - | - | - | - | - | - | needs two annotators |")
    lines += ["", "## HB-1 versus annotations (balanced accuracy, abstentions counted as failures)", "",
              "| target | view | role | windows | positives | abstained | evaluable | bal. acc. [participant CI] | always-no-cue |", "|---|---|---|---|---|---|---|---|---|"]
    for t, views in r["scoring"].items():
        for v, roles in views.items():
            for role, d in roles.items():
                ci = d["hb1"]["ci_balanced_accuracy_by_participant"]
                lines.append(f"| {t} | {v} | {role} | {d['n_windows']} | {d['n_positive_windows']} | {d['n_abstained']} | {str(d['evaluable']).lower()} | "
                             f"{_f(d['hb1']['full_cohort']['balanced_accuracy'])} [{_f(ci['lo'])}, {_f(ci['hi'])}] | "
                             f"{_f(d['always_no_cue']['full_cohort']['balanced_accuracy'])} |")
    lines += ["", "## Notes", *[f"- {n}" for n in r["notes"]],
              f"- final_opened_once_enforced: {str(r['final_opened_once_enforced']).lower()}; protocol_hash: {r['protocol_hash']}", ""]
    return "\n".join(lines)


def write_pilot_report(report: dict, directory: str) -> tuple:
    """Write report.json and report.md into an explicit existing directory; never overwrites."""
    import json
    import os

    md = render_pilot_markdown(report)
    jp, mp = os.path.join(directory, "pilot_report.json"), os.path.join(directory, "pilot_report.md")
    with open(jp, "x", encoding="utf-8") as fh:
        fh.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    with open(mp, "x", encoding="utf-8") as fh:
        fh.write(md)
    return jp, mp
