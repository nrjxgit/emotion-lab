# API

- `GET /api/health` → `{status, mediapipe, torch, onnx}`
- `POST /api/infer` multipart `image` → `{face, body, fusion, ethics_notice}`
  - face: `{box, aus[{code,name,intensity,confidence}], expression{probs,valence,arousal,expressiveness,uncertainty,source}, geometry{...}, quality}`
  - body: `{visible, keypoints, movement{energy,...}, probs, uncertainty}`
  - fusion: `{probs, confidence, uncertainty, evidence[], mode}`
- `POST /api/calibrate` `{aus, label?}` opt-in baseline update
- `GET /api/anatomy/stats` → hypothesis-test helpers (nose-geometry vs AU09 etc.)
- `GET /api/fairness/report` → stratified metrics schema

All probs sum to 1.0 across 8 classes: happiness, sadness, anger, surprise, fear, disgust, neutral, unknown.

## v0.5 research model selection (default unchanged)

- `GET /api/models`: model registry, default and experimental status. Does not
  load/download ViT. HSEmotion is explicitly not integrated.
- `POST /api/infer?model=mini-xception|vit`: same multipart image contract. Omit
  `model` for the unchanged legacy mini-Xception path. `selected_model` is added
  to the response. Invalid model => 422; missing ViT dependencies/weights => 503;
  HSEmotion placeholder => 501, never silently falls back.
- ViT is pinned to `747cf16692eea925b54b7b543cf436848128b68d`, experimental,
  uncalibrated; unknown internal feelings remain unknown. It has no trained
  arousal/expressiveness output, so those values are null. The static webcam UI
  has an opt-in model selector. Scores from different models must not be treated
  as equally calibrated confidence.
- Reported FER benchmarks use raw seven-class probabilities, not the legacy
  API's heuristic Unknown-mass augmentation; this distinction is intentional.

Calibration in `eval_results/calibration.json` is a historical train-derived
experiment, not a deployed calibration certificate. Crop selection uses a
separate hashed validation manifest; test variants are diagnostic only.
`source: trained|heuristic` always present. `ethics_notice` always present.

## v0.7 calibrated inference, abstention, consent (additive)

- `POST /api/infer?profile=corrected|legacy` (default corrected): corrected =
  YuNet margin25 crop + [-1,1] upstream normalization; legacy = tight box + [0,1]
  AREA (byte-identical to v0.5 and earlier). Unknown profile => 422.
- `POST /api/infer?calibrated=true`: applies T=1.045, validated ONLY for
  mini-xception + corrected (else 422). Response adds `probs_raw`, `temperature`,
  `calibrated: true` and a plain-language note. Default off; raw scores stay default.
- `POST /api/infer?abstain=0.5|0.7|0.9|0.95`: only validated thresholds accepted
  (else 422). Adds `abstained: bool` + `abstain_threshold`; abstained readings append
  `below-abstention-threshold` to insufficient-evidence reasons. Probabilities are
  still returned for inspection.
- `POST /api/calibrate` now requires `{"consent": true}` to store baseline data.
- `DELETE /api/session?participant=X` removes in-memory session and stored rows.
- Uploads over 5 MB are rejected (413). Server binds 127.0.0.1 in `run_backend.bat`;
  CORS remains open — do not expose beyond localhost without hardening.
