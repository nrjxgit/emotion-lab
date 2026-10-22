"""Tests for the DDAMFN++ evaluation gate.

The gate checks whether DDAMFN++ weights and licence are available before the
adapter is allowed to run. Returns (eligible: bool, reason: str) — never
silently integrates unlicensed weights.
"""

import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))


def test_ddamfn_gate_rejects_missing_weights():
    from training.ddamfn_gate import evaluate_gate

    eligible, reason = evaluate_gate("/nonexistent/path")
    assert eligible is False
    assert isinstance(reason, str) and len(reason) > 0


def test_ddamfn_gate_accepts_valid_setup(tmp_path, monkeypatch):
    from training.ddamfn_gate import evaluate_gate

    d = tmp_path / "ddamfn"
    d.mkdir()
    (d / "model.pt").write_bytes(b"fake-weights")
    (d / "LICENSE").write_text("MIT\n")
    eligible, reason = evaluate_gate(str(d))
    assert eligible is True
    assert "mit" in reason.lower()


def test_ddamfn_gate_rejects_proprietary_licence(tmp_path):
    from training.ddamfn_gate import evaluate_gate

    d = tmp_path / "ddamfn2"
    d.mkdir()
    (d / "model.pt").write_bytes(b"fake-weights")
    (d / "LICENSE").write_text("All rights reserved.\n")
    eligible, reason = evaluate_gate(str(d))
    assert eligible is False
    assert "proprietary" in reason.lower() or "all rights" in reason.lower()
