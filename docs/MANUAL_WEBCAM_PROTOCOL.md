# Manual webcam evaluation protocol (no camera in this build environment)

Run on a machine with a camera. Each condition: one consenting adult participant
at a time; visible recording indicator; guardian consent + presence for minors
(infant protocol stays experimental — observe only, no state claims).

1. Start backend: `run_backend.bat`; open `frontend-standalone/index.html`.
2. Conditions matrix (record AI top-1 + confidence + uncertainty per cell):
   - Expressions (posed, then spontaneous): neutral, smile, brow-furrow, surprise, mouth-open.
   - Lighting: overhead, side-window, dim (~1 lux desk lamp).
   - Angles: frontal, ±30° yaw, ±15° pitch.
   - Movement: still, talking, head-shake, hand-gesture.
3. Occlusion checks: glasses on/off, hand-over-mouth, profile (expect Unknown + high uncertainty).
4. Self-report: after each posed block ask "what were you aiming to show / feeling?"
   log via `POST /api/session/record {participant, ai_label, self_report}`.
5. Export: `GET /api/session/report?participant=X` → agreement rate (alignment, not accuracy).
6. PASS criteria: frontal posed smile → happiness top-1; profile/occluded → Unknown or uncertainty>0.6; no crashes on camera drop (unplug mid-run → typed error shown).

`scripts/manual_capture.py` automates frame grabs + /api/infer logging into `eval_results/manual_run_<ts>.json`.
