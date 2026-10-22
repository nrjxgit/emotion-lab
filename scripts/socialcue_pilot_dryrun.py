"""Fabricated annotation reconciliation; no capture, consent or model performance.

Usage: python -B scripts/socialcue_pilot_dryrun.py [OUT.txt]
Only OUT.txt is written. Fixtures stay in memory; importing this module does no IO.
"""
import copy
import json
import pathlib
import sys

HEADER = "SYNTHETIC / NOT REAL MODEL PERFORMANCE / no real consent"
PROTOCOL = "receipts/EL-006-socialcue/S9pilot/pilot-protocol-v4.1.md"
# Verified at 2d2a8bf against protocol v4.1. Tests check anchors and absence
# markers; these are discrepancy records, never implementation claims.
GAPS = (
    {"id": "tiers", "title": "Distinct tiers A/B", "status": "unimplemented",
     "protocol_ref": f"{PROTOCOL}:24", "code_ref": "socialcue/pilot/annotations.py:30",
     "expected_substring": 'ALLOWED_KEYS = frozenset({"schema_version", "annotator", "participant",',
     "detail": "closed schema has no tier; scoring has no tier population",
     "absent_constructs": [{"file": "socialcue/pilot/annotations.py", "pattern": r"^tier$"}]},
    {"id": "log_labels", "title": "Log-derived label handling", "status": "unimplemented",
     "protocol_ref": f"{PROTOCOL}:40", "code_ref": "socialcue/pilot/window_scoring.py:265",
     "expected_substring": "for lab in A.LABELS:",
     "detail": "all four labels enter agreement, including pause/asks_help"},
    {"id": "log_comparator", "title": "Interaction-log comparator", "status": "unimplemented",
     "protocol_ref": f"{PROTOCOL}:53", "code_ref": "socialcue/pilot/window_scoring.py:13",
     "expected_substring": "AUROC, ECE and the interaction-log comparator are not",
     "detail": "interaction-log comparator absent; only always-no-cue baseline is built at window_scoring.py:192",
     "absent_constructs": [{"file": "socialcue/pilot/window_scoring.py", "pattern": r"^(?:interaction_log|log_only|log_comparator)$"}]},
    {"id": "pooled_primary", "title": "Pooled primary analysis within each tier", "status": "unimplemented",
     "protocol_ref": f"{PROTOCOL}:53", "code_ref": "socialcue/pilot/window_scoring.py:220",
     "expected_substring": "if len(scored) >= MIN_PARTICIPANTS_FOR_SPLIT:",
     "detail": "automatic DEV/VAL/FINAL split or ALL below the split threshold, no within-tier primary"},
    {"id": "recorder", "title": "Capture/recording tool", "status": "unimplemented",
     "protocol_ref": f"{PROTOCOL}:31", "code_ref": "socialcue/pilot/window_scoring.py:196",
     "expected_substring": "def score_pilot(adapter, run_scope, streams, annotations,",
     "detail": "scorer accepts caller-supplied streams/annotations; pilot package contains no recorder",
     "absent_constructs": [{"file": "socialcue/pilot/window_scoring.py", "pattern": r"^(?:recorder|capture|start_recording)$"}]},
    {"id": "final_hash", "title": "FINAL-once/protocol hash", "status": "unimplemented",
     "protocol_ref": f"{PROTOCOL}:270", "code_ref": "socialcue/pilot/window_scoring.py:312",
     "expected_substring": '"final_opened_once_enforced": False, "protocol_hash": None',
     "detail": "false / null, no enforcement"},
    {"id": "block_floor", "title": "Distinct positive-block floor", "status": "unimplemented",
     "protocol_ref": f"{PROTOCOL}:51", "code_ref": "socialcue/pilot/window_scoring.py:180",
     "expected_substring": "evaluable = pos_participants >= MIN_POS_PARTICIPANTS and len(positives) >= MIN_POS_WINDOWS",
     "detail": "floor counts aligned HB outcomes, not distinct annotation blocks"},
    {"id": "ambiguous", "title": "Ambiguous flag rename", "status": "unimplemented",
     "protocol_ref": f"{PROTOCOL}:212", "code_ref": "socialcue/pilot/annotations.py:25",
     "expected_substring": 'FLAGS = ("unscorable", "unsure_rate", "confound_flag")',
     "detail": "unsure_rate remains; ambiguous is not accepted"},
    {"id": "report_claims", "title": "Outcome-dependent pilot report", "status": "unimplemented",
     "protocol_ref": f"{PROTOCOL}:58", "code_ref": "socialcue/pilot/window_scoring.py:49",
     "expected_substring": "PILOT_LIMITATIONS = (EVAL_ALLOWED_CLAIMS[0]",
     "detail": "proxy limitations prefix and fixed allowed-claim text remain"},
    {"id": "practice", "title": "Practice-insufficiency gate / support-driven practice set", "status": "unimplemented",
     "protocol_ref": f"{PROTOCOL}:44", "code_ref": "socialcue/pilot/agreement.py:163",
     "expected_substring": "def label_agreement(a, b, positive=",
     "detail": "generic agreement only; no practice-set support gate or practice-insufficient decision",
     "absent_constructs": [{"file": f"socialcue/pilot/{name}.py", "pattern": r"practice"}
                           for name in ("annotations", "agreement", "window_scoring")]},
    {"id": "tier_sufficiency", "title": "Tier-specific not-estimable / descriptive / primary populations", "status": "unimplemented",
     "protocol_ref": f"{PROTOCOL}:51", "code_ref": "socialcue/pilot/window_scoring.py:180",
     "expected_substring": "evaluable = pos_participants >= MIN_POS_PARTICIPANTS and len(positives) >= MIN_POS_WINDOWS",
     "detail": "no tier population gate: protocol tier <10 counts-only, 10-29 descriptive, primary >=30; ten-person normal case is only exploratory under protocol",
     "absent_constructs": [{"file": "socialcue/pilot/window_scoring.py", "pattern": r"^(?:tier|primary|descriptive|not_estimable)$"}]},
    {"id": "block_counts", "title": "Distinct positive-block counts alongside aligned windows", "status": "unimplemented",
     "protocol_ref": f"{PROTOCOL}:51", "code_ref": "socialcue/pilot/window_scoring.py:187",
     "expected_substring": '"n_windows": len(outcomes), "n_positive_windows": len(positives)',
     "detail": "per-target positive counts are aligned windows; aggregate blocks count is not a distinct positive-block count",
     "absent_constructs": [{"file": "socialcue/pilot/window_scoring.py", "pattern": r"^(?:n_positive_blocks|n_distinct_positive_blocks)$"}]},
    {"id": "paired_fer7", "title": "Same-draw paired comparisons and FER7-only comparator", "status": "unimplemented",
     "protocol_ref": f"{PROTOCOL}:272", "code_ref": "socialcue/pilot/window_scoring.py:192",
     "expected_substring": '"always_no_cue":',
     "detail": "only always-no-cue comparator; no FER7-only comparator or paired differences on shared participant draws",
     "absent_constructs": [{"file": f"socialcue/pilot/{name}.py", "pattern": r"(?:fer7|paired|shared_draw)"}
                           for name in ("window_scoring", "agreement")]},
    {"id": "mixed_rows", "title": "Mixed video/log state-row labels", "status": "unimplemented",
     "protocol_ref": f"{PROTOCOL}:40", "code_ref": "socialcue/pilot/window_scoring.py:45",
     "expected_substring": 'TARGETS = tuple(f"raw:{l}"',
     "detail": "raw/state target names have no mixed video/log source label",
     "absent_constructs": [{"file": "socialcue/pilot/window_scoring.py", "pattern": r"\bmixed\b"}]},
    {"id": "adjudication", "title": "Adjudication as a separate view", "status": "unimplemented",
     "protocol_ref": f"{PROTOCOL}:274", "code_ref": "socialcue/pilot/window_scoring.py:260",
     "expected_substring": 'views = list(annotators) + (["consensus"]',
     "detail": "individual annotators and agreement-only consensus; no separate adjudicated view",
     "absent_constructs": [{"file": "socialcue/pilot/window_scoring.py", "pattern": r"adjudicat"}]},
    {"id": "outcome_validator", "title": "Neutral failure/undefined wording and outcome-specific null-safe validator", "status": "unimplemented",
     "protocol_ref": f"{PROTOCOL}:57", "code_ref": "socialcue/pilot/window_scoring.py:369",
     "expected_substring": 'or tuple(report["allowed_claims"]) != PILOT_ALLOWED_CLAIMS',
     "detail": "validator enforces verbatim fixed claims, not outcome-specific neutral conclusions; existing null withholding and raw-value stripping do work",
     "absent_constructs": [{"file": "socialcue/pilot/window_scoring.py", "pattern": r"^(?:outcome_conclusion|registered_criterion|neutral_conclusion)$"}]},
)


def fixtures():
    """Reserved synthetic participant/annotator codes only, including rejected cases."""
    def record(p, a, start, positive=True):
        return dict(schema_version=1, participant=f"PT-5eed{p:04x}", annotator=f"ANN-{a:06x}",
                    window_start=start, window_end=start + 3,
                    labels={k: "present" if positive else "absent" for k in ("look_away", "pause", "asks_help", "fidget")},
                    unscorable=False, unsure_rate=False, confound_flag=False)

    normal = [record(p, a, s, s < 9) for p in range(1, 11) for a in (1, 2) for s in (0, 3, 6, 9)]
    below = [record(p, a, s, s == 0) for p in (1, 2) for a in (1, 2) for s in (0, 3)]
    third = copy.deepcopy(normal) + [record(1, 3, 0), record(1, 3, 9, False)]
    missing = [record(1, 1, 0), record(1, 1, 3, False), record(1, 2, 0)]
    missing[0]["unscorable"] = True
    pii = [record(1, 1, 0)]
    pii[0]["email"] = "fabricated@example.invalid"
    return {"normal_multi_participant": normal, "below_floor": below,
            "third_rater_partial": third, "unscorable_missing_block": missing,
            "pii_shaped_field": pii, "overlapping_blocks": [record(1, 1, 0), record(1, 1, 2)],
            "non_grid_start": [record(1, a, .5) for a in (1, 2)]}


def run_dryrun():
    # Lazy imports preserve a side-effect-free module import, including sys.path.
    from socialcue import synthetic
    from socialcue.data.synthetic_adapter import SyntheticAdapter
    from socialcue.pilot import annotations as A, window_scoring as W

    cases = {}
    for name, records in fixtures().items():
        # Round-trip manufactured annotation records through a file-format encoding,
        # without creating incidental fixture files on disk.
        records = json.loads(json.dumps(records, sort_keys=True))
        try:
            accepted = A.validate_batch(records)
        except A.AnnotationError as exc:
            cases[name] = {"accepted": False, "reasons": exc.codes, "input_records": len(records)}
            continue
        streams = {p: synthetic.generate("unsure-like" if i % 2 else "calm", duration=13, seed=505)
                   for i, p in enumerate(sorted({r["participant"] for r in accepted}))}
        result = W.score_pilot(SyntheticAdapter(), "synthetic", streams, accepted)
        report = W.build_pilot_report(result)
        W.validate_pilot_report(report)
        cases[name] = {"accepted": True, "reasons": ["closed schema and non-overlap validation passed"],
                       "report": report, "markdown": W.render_pilot_markdown(report)}
    return cases


def render(cases):
    from socialcue.pilot.window_scoring import MIN_POS_PARTICIPANTS, MIN_POS_WINDOWS
    from socialcue.pilot.agreement import MIN_POSITIVES_FOR_INTERPRETATION

    lines = [HEADER, "RECONCILIATION - fabricated annotation plumbing only", "",
             "No real participants, independent judgments, consent, capture or practice certification.",
             f"Existing code floor: >={MIN_POS_PARTICIPANTS} participants WITH positives and >={MIN_POS_WINDOWS} positive aligned HB windows.",
             "Protocol instead requires distinct positive annotation blocks within each tier; NOT implemented.",
             f"Agreement: <{MIN_POSITIVES_FOR_INTERPRETATION} positives summed across raters => kappa/alpha and kappa CI withheld.",
             "Report-template allowed claims below are conditional boilerplate, not findings of this dry-run.", ""]
    for name, case in cases.items():
        lines += [f"CASE {name}: {'ACCEPTED' if case['accepted'] else 'REJECTED'}", "; ".join(case["reasons"])]
        if not case["accepted"]:
            lines += [f"Input records: {case['input_records']}; no scoring or statistics (validation rejected).", ""]
            continue
        r = case["report"]["results"]
        lines += ["Support: " + json.dumps(r["counts"], sort_keys=True),
                  "Agreement including withholding reasons (sanitized by existing report builder):",
                  json.dumps(r["agreement"], sort_keys=True, allow_nan=False)]
        for target, views in r["scoring"].items():
            for view, roles in views.items():
                for role, row in roles.items():
                    lines.append(f"Support {target}/{view}/{role}: participants={row['n_participants']}; "
                                 f"positive_participants={row['n_participants_with_positive']}; "
                                 f"positive_aligned_windows={row['n_positive_windows']}; "
                                 f"evaluable={row['evaluable']}; reason={row['not_evaluable_reason']}")
        lines += [case["markdown"], ""]
    lines += ["Non-grid starts: documented convention, NOT enforced; overlap IS rejected.",
              "Third-rater partial coverage: first-two agreement retains its own pairs; consensus requires all raters.",
              "Missing blocks are not imputed; unscorable blocks excluded. See per-view support above.", "",
              "KNOWN DESIGN GAPS (verified at 2d2a8bf; all remain unimplemented):"]
    for gap in GAPS:
        lines.append(f"- [{gap['id']}] {gap['title']} ({gap['status']}): {gap['detail']}. "
                     f"Code {gap['code_ref']}; protocol {gap['protocol_ref']}; anchor {gap['expected_substring']!r}.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) > 1:
        raise SystemExit("usage: socialcue_pilot_dryrun.py [OUT.txt]")
    text = render(run_dryrun())
    if args:
        with pathlib.Path(args[0]).open("x", encoding="utf-8", newline="\n") as output:
            output.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    main()
