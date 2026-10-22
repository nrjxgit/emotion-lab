"""Opt-in calibration sessions: AI predictions vs participant self-reports.

Local JSON only. Disagreement is reported as an agreement rate — never as
the participant being wrong.
"""

import json
import os
import time


class CalibrationSession:
    def __init__(self, store: str = "configs/calibration_sessions.json"):
        self.store = store
        self.session = None

    def start(self, consent: bool, participant: str = "anon") -> dict:
        if not consent:
            raise PermissionError("explicit opt-in consent required")
        self.session = {
            "participant": participant,
            "consent": True,
            "t0": time.time(),
            "records": [],
        }
        return {"ok": True, "participant": participant}

    def record(
        self, ai_label: str, self_report: str, probs: dict | None = None
    ) -> dict:
        if not self.session:
            raise RuntimeError("no active session: call start(consent=True) first")
        self.session["records"].append(
            {
                "ai": ai_label,
                "self": self_report,
                "probs": probs or {},
                "t": time.time(),
            }
        )
        return {"n": len(self.session["records"])}

    def report(self) -> dict:
        if not self.session:
            raise RuntimeError("no active session")
        recs = self.session["records"]
        agree = sum(1 for r in recs if r["ai"] == r["self"])
        per = {}
        for r in recs:
            d = per.setdefault(r["self"], {"n": 0, "agree": 0})
            d["n"] += 1
            d["agree"] += r["ai"] == r["self"]
        out = {
            "consent": True,
            "n": len(recs),
            "agreement_rate": (agree / len(recs)) if recs else 0.0,
            "per_expression": {
                k: {"n": v["n"], "agree": v["agree"]} for k, v in per.items()
            },
            "note": "Agreement with self-report is not accuracy: feelings are private; "
            "this measures alignment between visible cues and stated experience.",
        }
        os.makedirs(os.path.dirname(self.store) or ".", exist_ok=True)
        prior = []
        if os.path.exists(self.store):
            try:
                prior = json.load(open(self.store))
            except Exception:
                prior = []
        prior.append(
            self.session | {"report": {k: v for k, v in out.items() if k != "note"}}
        )
        json.dump(prior, open(self.store, "w"))
        return out
