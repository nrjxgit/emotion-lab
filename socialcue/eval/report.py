"""JSON + Markdown report writer for the proxy evaluation (EL-006 S3).

Every report carries the status label, the proxy-label validity caveat, the
allowed and disallowed claims, the provenance manifest hash and the HB-1 config
hash. A report missing any of them is refused at render/write time.
"""
from __future__ import annotations

import json
import os

from ..data.rights import canonical_json, sha256_of

SYNTHETIC_LABEL = "SYNTHETIC / NOT REAL MODEL PERFORMANCE"
PROXY_LABEL = "PROXY-LABEL RESEARCH RUN / NOT CUE-DETECTION ACCURACY"

CAVEAT = (
    "Proxy-label validity: the labels come from acted or manufactured basic-emotion data. Acted affect is not "
    "nervousness, uncertainty or need for help, and facial movements do not reliably map to inner states "
    "(Barrett, Adolphs, Marsella, Martinez & Pollak 2019, Psychological Science in the Public Interest 20(1):1-68, "
    "doi 10.1177/1529100619832930). These numbers only say whether an unvalidated heuristic separates acted "
    "proxy classes. P1/P2 are proxies, not the three target states."
)
ALLOWED_CLAIMS = (
    "Research prototype. On held-out actors of [dataset], an unvalidated heuristic separated acted [proxy positive] "
    "from [proxy negative] clips with balanced accuracy X [CI]. Proxy labels from acted basic emotions; does not "
    "measure nervousness, uncertainty or need for help; not for decisions about people; no commercial claim.",
)
DISALLOWED_CLAIMS = (
    "Any accuracy of nervous / unsure / needs-help detection.",
    "Any real-world efficacy claim.",
    "Pooling RAVDESS and CREMA-D without per-dataset rows.",
)
REQUIRED = ("status", "caveat", "allowed_claims", "disallowed_claims", "manifest_sha256", "hb1_config_sha256")


class ReportIncomplete(ValueError):
    """Raised when a report lacks a mandatory label, caveat, claims block or hash."""


def build_report(results: dict) -> dict:
    manifest = results["manifest"]
    synthetic = results["adapter"]["source_class"] == "synthetic"
    report = {
        "schema": "socialcue-proxy-report-v1",
        "status": SYNTHETIC_LABEL if synthetic else PROXY_LABEL,
        "licence_scope": manifest["licence_banner"],
        "caveat": CAVEAT,
        "allowed_claims": list(ALLOWED_CLAIMS),
        "disallowed_claims": list(DISALLOWED_CLAIMS),
        "manifest_sha256": manifest["manifest_sha256"],
        "hb1_config_sha256": results["config_sha256"],
        "results": results,
    }
    return {**report, "report_sha256": sha256_of(report)}


def validate(report: dict) -> None:
    missing = [k for k in REQUIRED if not report.get(k)]
    if missing:
        raise ReportIncomplete(f"report is missing: {', '.join(missing)}")
    if report["caveat"] != CAVEAT or tuple(report["allowed_claims"]) != ALLOWED_CLAIMS \
            or tuple(report["disallowed_claims"]) != DISALLOWED_CLAIMS:
        raise ReportIncomplete("caveat/claims blocks must be the verbatim text")
    synthetic = report["results"]["adapter"]["source_class"] == "synthetic"
    if synthetic and report["status"] != SYNTHETIC_LABEL:
        raise ReportIncomplete("a synthetic run must carry the SYNTHETIC label")
    if not synthetic and report["status"] != PROXY_LABEL:
        raise ReportIncomplete("a non-synthetic run must carry the PROXY-LABEL status")
    if report["results"].get("banner") != report["status"] or report["results"].get("caveat") != CAVEAT:
        raise ReportIncomplete("results must carry the same banner and the verbatim caveat as the report")
    if report["results"].get("final_opened_once_enforced") is not False or "protocol_hash" not in report["results"]:
        raise ReportIncomplete("results must record the FINAL-once / protocol-hash limitation")
    if report["results"]["manifest"]["manifest_sha256"] != report["manifest_sha256"]:
        raise ReportIncomplete("manifest hash does not match the results")
    body = {k: v for k, v in report.items() if k != "report_sha256"}
    if report.get("report_sha256") != sha256_of(body):
        raise ReportIncomplete("report hash mismatch (edited after build)")


def _fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


def render_markdown(report: dict) -> str:
    validate(report)
    r = report["results"]
    lines = [
        f"# {report['status']}",
        "",
        f"Licence scope: {report['licence_scope']}. Adapter: {r['adapter']['name']} ({r['adapter']['source_class']}). "
        f"Seed {r['seed']}, {r['n_draws']} actor-grouped bootstrap draws (95% percentile).",
        f"Provenance manifest sha256: `{report['manifest_sha256']}`  ",
        f"HB-1 config sha256: `{report['hb1_config_sha256']}`  ",
        f"Report sha256: `{report['report_sha256']}`",
        "",
        "## Proxy-label caveat",
        report["caveat"],
        "",
        "## Known limitations",
        f"- final_opened_once_enforced: {str(r['final_opened_once_enforced']).lower()}; protocol_hash: {r['protocol_hash']}",
        *[f"- {k}: {v}" for k, v in r["method_notes"].items()],
        "",
        "## Allowed claims",
        *[f"- {c}" for c in report["allowed_claims"]],
        "",
        "## Disallowed claims",
        *[f"- {c}" for c in report["disallowed_claims"]],
        "",
        "## Splits (actor-disjoint)",
        *[f"- {role}: {len(actors)} actors" for role, actors in r["splits"].items()],
        "",
    ]
    for ep, roles in r["endpoints"].items():
        for role, methods in roles.items():
            lines += [f"## {ep} / {role}", "",
                      "| method | clips | balanced acc [95% CI] | recall | precision | F1 | abstention |",
                      "|---|---|---|---|---|---|---|"]
            for m, d in methods.items():
                fc, ci = d["full_cohort"], d["ci_full_cohort"]["balanced_accuracy"]
                lines.append(f"| {m} | {d['n_clips']} | {_fmt(fc['balanced_accuracy'])} [{_fmt(ci['lo'])}, {_fmt(ci['hi'])}] "
                             f"| {_fmt(fc['recall'])} | {_fmt(fc['precision'])} | {_fmt(fc['f1'])} | {_fmt(fc['abstention_rate'])} |")
            lines.append("")
        c = r["confusers"][ep]
        lines += [f"Confusers for {ep}: {c['n_clips']} clips ({', '.join(c['classes']) or 'none'}); excluded by the adapter: {c['excluded_n']}; "
                  f"HB-1 positive rate {_fmt(c['hb1_positive_rate'])} ({c['hb1_positive_n']} of {c['n_answered']} answered), "
                  f"abstention {_fmt(c['abstention_rate'])} ({c['n_abstained']} of {c['n_clips']}).", ""]
    lines += ["## Notes", *[f"- {n}" for n in r["notes"]], ""]
    return "\n".join(lines)


def write_report(report: dict, directory: str) -> tuple:
    """Write report.json and report.md into an explicit existing directory; never overwrites."""
    md = render_markdown(report)  # validates first
    jp, mp = os.path.join(directory, "report.json"), os.path.join(directory, "report.md")
    with open(jp, "x", encoding="utf-8") as fh:
        fh.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    with open(mp, "x", encoding="utf-8") as fh:
        fh.write(md)
    return jp, mp


__all__ = ["build_report", "render_markdown", "write_report", "validate", "ReportIncomplete", "canonical_json"]
