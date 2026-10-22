# Limitations / NOT done — v0.2 (honest list)

1. **No labeled accuracy numbers.** FER2013 CSV needs a Kaggle account; Aff-Wild2/RAF-DB need requests. Nothing downloaded → confusion/macF1/ECE on real data UNMEASURED. `training/evaluate.py` is implemented and unit-tested (toy data) but has no real inputs yet.
2. **Disgust class ~0 on tested inputs.** Seen on dark/gradient/lena. Could be genuine (rare class) or a subtle forward-pass deviation (e.g. padding). Not resolved without TF cross-check (TF has no Python-3.14 wheel here). Do NOT trust disgust outputs.
3. **AU intensities uncalibrated.** Geometric rules from YuNet-5pt-derived landmarks; lena gave AU15=1.0 on a neutral mouth. Needs DISFA/BP4D validation. Kept separate from trained probs.
4. **No body-emotion model.** Searched: no suitably licensed full-body emotion weights found for this stack (MMpose/OpenPose models need torch + dataset-specific heads). Body stays motion-only.
5. **Landmarks are 5pt + completion**, not 468pt mesh (mediapipe not installed). Brow/chin points are estimated, flagged `_source: yunet-5pt`.
6. **Demographic/infant/anatomy claims: none.** Modules exist as scaffolding; no consented labeled data → no group comparisons run.
7. **oarriaga weights licence:** source repo is GPL-3.0. Weights used locally for research; do NOT redistribute the .hdf5 in a commercial bundle without review.
8. **Frontend dashboard is minimal** (mode buttons + copy); full Recharts dashboard + XAI overlays remain to build. Standalone HTML works against :8000.
9. **Webcam live loop untested here** (no camera in this environment); error path tested, happy path not.
