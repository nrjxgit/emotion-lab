"""Synthetic in-process route rehearsal with inert module stubs installed BEFORE app import.

Usage: python -B scripts/socialcue_rehearsal.py [OUT.txt]
Print a transcript, or explicitly write it to OUT.txt. No real model/camera/media is used.
The audit hook does not cover native file IO (cv2/h5py). Zero counters show only
absence of observed audited events, not complete isolation or a native-IO guarantee.
Each frame compares the exact retained-snapshot reference (must match, else failure)
and a reconstructed full-history reference (ring definition effect). Comparisons
reuse the route's body inputs/deltas; they do not independently validate body science.
Reports total mismatch count plus the first 25 details per reference. HEAD is read
at runtime and identifies checkout HEAD only, not dirty/untracked source.
"""
import dataclasses
import importlib.abc
import json
import os
import pathlib
import subprocess
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
HEADER_LINE = "SYNTHETIC / NOT REAL MODEL PERFORMANCE - inert module stubs precede app import; audit hook does not cover native file IO (cv2/h5py); zero counters show only absence of observed audited events"
SID = "rehearsal-session-0001-abcdefgh"
STEP = 0.125  # 8 fps
BLOCKED = {"torch", "tensorflow", "transformers", "onnxruntime", "mediapipe"}
FER7_TO_CLASS = {"happy": "happiness", "sad": "sadness", "angry": "anger", "surprise": "surprise",
                 "fear": "fear", "disgust": "disgust", "neutral": "neutral"}
FACE_SHIFT_Y = 100  # aligns the synthetic face with the manufactured skeleton, as tests/test_socialcue_body_api.py does


class IsolationError(RuntimeError):
    pass


class _Guard:
    """Blocks selected imports and observed audited events; native file IO is not covered."""

    def __init__(self):
        self.active = False
        self.attempts, self.model_opens, self.sockets, self.selfpipe = [], [], [], 0
        self.finder = None

    def install(self):
        if getattr(sys, "_socialcue_rehearsal_guard", None) is not None:
            sys._socialcue_rehearsal_guard.reset()
            return sys._socialcue_rehearsal_guard
        loaded = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
        if loaded:
            raise IsolationError(f"heavy frameworks already imported, refusing to run: {loaded}")
        guard = self
        outer = self

        class NoHeavy(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in BLOCKED:
                    guard.attempts.append(fullname)
                    raise ImportError("rehearsal: heavy-framework import disabled")

        self.finder = NoHeavy()
        sys.meta_path.insert(0, self.finder)
        import socket
        fallback = socket._fallback_socketpair.__code__
        models = ROOT / "models"

        def audit(event, args):
            if not outer.active:
                return
            if event == "socket.connect":
                frame, via_selfpipe = sys._getframe(1), False
                while frame is not None:
                    if frame.f_code is fallback:
                        via_selfpipe = True
                    frame = frame.f_back
                addr = args[1] if len(args) > 1 else None
                if via_selfpipe and isinstance(addr, tuple) and addr[0] in ("127.0.0.1", "::1") and addr[1] != 8765:
                    outer.selfpipe += 1  # asyncio's in-process wake-up pipe on Windows, not a network connection
                    return
                outer.sockets.append(str(addr))
                raise RuntimeError("rehearsal: network connection forbidden")
            if event == "open" and args and isinstance(args[0], (str, bytes)):
                try:
                    p = pathlib.Path(os.path.abspath(os.fsdecode(args[0])))
                except Exception:
                    return
                if p == models or models in p.parents:
                    outer.model_opens.append(str(p))
                    raise RuntimeError("rehearsal: models/ is off limits")

        sys.addaudithook(audit)
        sys._socialcue_rehearsal_guard = self
        return self

    def reset(self):
        self.attempts, self.model_opens, self.sockets, self.selfpipe = [], [], [], 0


def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _scripts(syn, sk):
    """name -> (records, skeleton name). Expected qualitative story is documented per scenario."""
    calm = syn.build(duration=8.0, fps=8.0)
    hesitation = syn.build(duration=12.0, fps=8.0, uns=lambda t: 1.0 if t >= 3.0 else 0.0)
    escalation = syn.build(duration=22.0, fps=8.0, nerv=lambda t: 1.0 if 2.0 <= t < 6.0 else 0.0,
                           uns=lambda t: 1.0 if t >= 6.0 else 0.0, drop=lambda t, u: t >= 17.0)
    return {
        "calm": (calm, "calm"),
        "hesitation": (hesitation, "calm"),
        "escalation_with_absence_tail": (escalation, "fidget"),
    }


def _summary(block):
    body = block.get("body") or {}
    cues = body.get("cues") or []
    return {
        "states": [[s["name"], s["confidence"]] for s in block["states"]],
        "insufficient": block["insufficient"]["flag"], "reasons": block["insufficient"]["reasons"],
        "window": block["window"],
        "body": {"status": body.get("status"), "reason": body.get("reason"), "bystander_checked": body.get("bystander_checked"),
                 "frames": (body.get("window") or {}).get("frames"), "cues_present": [c["cue"] for c in cues if c.get("present")]},
    }


def _install_inert_modules():
    """Replace model-bearing module boundaries before importing main; no asset lookup."""
    modules = {}
    detector = types.ModuleType("face.detector")
    detector.detect_faces = lambda image: []
    detector.model_available = lambda: False  # YuNet asset check is inert too
    expression = types.ModuleType("face.expression")
    class InertPredictor:
        PROFILES = ("corrected", "legacy")
        TRAINED_SOURCE = "SYNTHETIC-rehearsal"
        def __init__(self, profile="corrected"):
            self.source = "SYNTHETIC-rehearsal"
        def predict(self, *args, **kwargs):
            raise IsolationError("scenario predictor was not installed")
    expression.ExpressionPredictor = InertPredictor
    expression.CLASSES = list(FER7_TO_CLASS.values()) + ["unknown"]
    expression.observable_feature_analysis = lambda aus: {"source": "SYNTHETIC-inert"}
    pose = types.ModuleType("body.pose")
    pose.estimate_pose = lambda image: {"visible": False, "keypoints": [], "quality": 0., "estimator_available": False}
    for module in (detector, expression, pose):
        module.__rehearsal_inert__ = True
        modules[module.__name__] = sys.modules.get(module.__name__)
        sys.modules[module.__name__] = module
    return modules


def run_rehearsal():
    """Run all three scenarios in-process; returns the transcript as a dict. Raises IsolationError if isolation fails."""
    guard = _Guard().install()
    guard.active = True
    saved_modules = _install_inert_modules()
    sys.path[:0] = [p for p in (str(ROOT / "backend" / "app"), str(ROOT)) if p not in sys.path]
    try:
        import cv2
        import numpy as np
        from fastapi.testclient import TestClient
        import main
        import socialcue_session as scs
        from socialcue import body as bodymod, state as hb1, synthetic as syn, synthetic_skeleton as sk

        loaded = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
        if loaded:
            raise IsolationError(f"heavy frameworks loaded by the app import: {loaded}")
        ok, buf = cv2.imencode(".jpg", np.full((480, 640, 3), 128, dtype=np.uint8))
        jpg = buf.tobytes()

        class Clock:
            t = 1000.0

            def __call__(self):
                return self.t

        saved = {k: getattr(main, k) for k in ("detect_faces", "get_predictor", "estimate_pose", "yunet_ok", "_track")}
        saved_store = scs.STORE
        out = {"header": HEADER_LINE, "git_head": _git_head(), "hb1_config_sha256": hb1.CONFIG_SHA256,
               "body_config_sha256": bodymod.BODY_CONFIG_SHA256, "fps": 8.0, "session_header": scs.HEADER, "scenarios": {}}
        try:
            for name, (records, skel_name) in _scripts(syn, sk).items():
                clock = Clock()
                class CapturingStore(scs.SocialCueSessions):
                    def add(self, session_id, make_record, make_pose=None, **kwargs):
                        result = super().add(session_id, make_record, make_pose, **kwargs)
                        self.snapshot = result[0] if isinstance(result, tuple) else result
                        self.poses = result[1] if isinstance(result, tuple) else []
                        self.current_pose = make_pose(self.snapshot.current.t, self.snapshot.current) if make_pose else None
                        return result
                store = CapturingStore(clock=clock)
                scs.STORE = store
                seq = sk.build(skel_name, fs=8.0, n=len(records))
                names = ("eye_R", "eye_L", "nose", "mouth_R", "mouth_L")
                cur = {"i": -1}

                def detect(img):
                    r = records[cur["i"]]
                    if r.box is None:
                        return []
                    x, y, w, h = r.box
                    return [{"x": int(x), "y": int(y) - FACE_SHIFT_Y, "w": int(w), "h": int(h), "confidence": r.quality,
                             "landmarks5": {n: (p[0], p[1] - FACE_SHIFT_Y) for n, p in zip(names, r.landmarks)}}]

                class Predictor:
                    source = "SYNTHETIC-rehearsal"

                    def predict(self, crop, aus, calibrated=False):
                        r = records[cur["i"]]
                        probs = {FER7_TO_CLASS[k]: v for k, v in r.expression.items()}
                        probs["unknown"] = 0.0
                        return {"probs": probs, "valence": 0.0, "arousal": r.arousal, "expressiveness": 0.0,
                                "confidence": max(probs.values()), "uncertainty": 1 - max(probs.values()),
                                "source": "fake", "profile": "corrected", "calibrated": False}

                predictor = Predictor()

                def pose(img):
                    if records[cur["i"]].box is None:
                        return {"visible": False, "keypoints": [], "quality": 0.0, "estimator_available": True}
                    return {"visible": True, "keypoints": seq.lm[min(cur["i"], len(seq.lm) - 1)], "quality": 0.9}

                main.detect_faces, main.get_predictor, main.estimate_pose = detect, (lambda profile: predictor), pose
                main.yunet_ok, main._track = (lambda: False), []
                client = TestClient(main.app)
                frames, pure_records = [], []
                mismatches = {label: [] for label in ("retained-snapshot reference", "full-history reference")}
                for i, rec in enumerate(records):
                    cur["i"] = i
                    clock.t += STEP
                    resp = client.post("/api/infer?socialcue=1", files={"image": ("synthetic.jpg", jpg, "image/jpeg")},
                                       headers={scs.HEADER: SID})
                    line = {"t": round(clock.t, 3), "scenario": name, "frame": i, "http": resp.status_code}
                    if resp.status_code == 200:
                        block = resp.json()["socialcue"]
                        line.update(_summary(block))
                        shifted = rec if rec.box is None else dataclasses.replace(
                            rec, box=(int(rec.box[0]), int(rec.box[1]) - FACE_SHIFT_Y, int(rec.box[2]), int(rec.box[3])),
                            landmarks=tuple((p[0], p[1] - FACE_SHIFT_Y) for p in rec.landmarks))
                        pure_records.append(dataclasses.replace(shifted, t=clock.t))
                        pure = hb1.analyze(list(pure_records), evaluation_time=clock.t)
                        route_states = {s["name"]: s["confidence"] for s in block["states"]}
                        deltas = {s["name"]: (s.get("body_support") or {}).get("delta", 0.0) for s in block["states"]}
                        pure_states = {s["name"]: round(min(0.8, max(0.0, s["confidence"] + deltas.get(s["name"], 0.0))), 4)
                                       for s in pure["states"]}  # body support re-applied to the pure HB-1 value, as attach_body does
                        snapshot = store.snapshot
                        retained = hb1.analyze(snapshot, evaluation_time=clock.t)
                        retained = scs._with_body(retained, store.poses, store.current_pose, True)
                        retained["banner"] = scs.BANNER
                        # Exact whole-response equality, not only states/flags or a numeric tolerance.
                        if retained != block:
                            raise AssertionError(f"retained-snapshot reference differs: {name} frame {i}")
                        line["route_record"] = dataclasses.asdict(snapshot.current)
                        line["retained_times"] = [r.t for r in snapshot]
                        line["bin_origin"] = snapshot.bin_origin
                        line["body_deltas"] = sorted(deltas.items())
                        line["references"] = {
                            "retained-snapshot reference": _summary(retained),
                            "full-history reference": {"states": [[k, v] for k, v in sorted(pure_states.items())],
                                                       "insufficient": pure["insufficient"]["flag"]},
                        }
                        for label, reference in line["references"].items():
                            reference_states = dict(reference["states"])
                            same_names = route_states.keys() == reference_states.keys()
                            diff = max([abs(route_states[k] - reference_states[k]) for k in route_states.keys() & reference_states.keys()] or [0.])
                            flag_same = block["insufficient"]["flag"] == reference["insufficient"]
                            if not (same_names and flag_same and diff <= .0005):
                                mismatches[label].append({"frame": i, "t": line["t"], "route": sorted(route_states.items()),
                                                          "reference": reference["states"], "effect": "ring definition effect"})
                    frames.append(line)
                ok_frames = [f for f in frames if f["http"] == 200]
                first = {}
                for f in ok_frames:
                    if not f["insufficient"]:
                        for st, _ in f["states"]:
                            first.setdefault(st, f["t"])
                out["scenarios"][name] = {
                    "frames_total": len(frames), "frames_http_200": len(ok_frames),
                    "first_seen_t": first, "last_frame": frames[-1],
                    "compare": {label: {"frames_compared": len(ok_frames), "mismatch_count": len(details),
                                        "mismatches": details[:25], "rule": ("exact retained snapshot and explicit evaluation time; any difference fails"
                                            if label == "retained-snapshot reference" else
                                            "ring definition effect: reconstructed full history, route body delta, confidence tolerance 0.0005")}
                                for label, details in mismatches.items()},
                    "frames": frames,
                }
        finally:
            for k, v in saved.items():
                setattr(main, k, v)
            scs.STORE = saved_store
        out["isolation"] = {"preimport_stub_modules": sorted(saved_modules), "audit_limit": "native file IO (cv2/h5py) is not covered; zero counters show only absence of observed audited events", "heavy_import_attempts": list(guard.attempts), "models_opens": list(guard.model_opens),
                            "network_connects": list(guard.sockets), "asyncio_selfpipe_connects": guard.selfpipe}
        if guard.model_opens or guard.sockets:
            raise IsolationError(f"isolation violated: {out['isolation']}")
        return out
    finally:
        guard.active = False
        for name, previous in saved_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def render(t):
    lines = [t["header"], f"git_head: {t['git_head']}", f"hb1_config_sha256: {t['hb1_config_sha256']}",
             f"body_config_sha256: {t['body_config_sha256']}", f"fps: {t['fps']}  header: {t['session_header']}",
             "isolation: " + json.dumps(t["isolation"], sort_keys=True), ""]
    for name, sc in t["scenarios"].items():
        lines.append(f"## scenario {name}: {sc['frames_http_200']}/{sc['frames_total']} frames HTTP 200; first_seen_t {json.dumps(sc['first_seen_t'], sort_keys=True)}")
        for f in sc["frames"]:
            lines.append(json.dumps(f, sort_keys=True, separators=(",", ":")))
        for label, c in sc["compare"].items():
            lines.append(f"## compare {name} / {label}: total mismatch count {c['mismatch_count']} in {c['frames_compared']} frames; first {len(c['mismatches'])} details (cap 25); rule: {c['rule']}")
            lines += [json.dumps(m, sort_keys=True, separators=(",", ":")) for m in c["mismatches"]]
        lines.append("")
    lines.append("SYNTHETIC / NOT REAL MODEL PERFORMANCE")
    return "\n".join(lines) + "\n"


def main_cli(argv):
    text = render(run_rehearsal())
    if len(argv) > 1:
        with open(argv[1], "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main_cli(sys.argv))
