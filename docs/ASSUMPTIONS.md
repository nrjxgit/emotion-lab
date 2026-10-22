# Assumptions (2026-10-16)
- Local-first; MediaPipe/Torch optional with graceful fallback (OpenCV5 removed Haar → center-box + low quality flag).
- Heuristic AU→emotion clearly labeled `source: heuristic` until genuine ONNX/HF weights downloaded via `models/download_weights.py`.
- Face box fallback confidence 0.2 → fusion caps confidence; never overconfident on missing data.
- Nose-geometry vs AU09 is a hypothesis test (Pearson + CI + confounders), not a claim.
- Infant pipeline always `experimental: true`.
- Demographics for evaluation only, never as inference inputs.
