# Capabilities — v0.2 (branch dev/v0.2-real-models, measured 2026-10-16)

## Operating AI models
| Model | Weights | Status | Evidence |
|---|---|---|---|
| YuNet face detector 2023mar | `models/face_detection_yunet_2023mar.onnx` (232KB, Apache-2.0) | OPERATING via cv2.dnn | lena 512px: 1 face conf 0.91; blank 320px: [] |
| mini-Xception FER2013 | `models/fer_mini_xception_h5.hdf5` (oarriaga, 29 layers loaded, shapes verified) | OPERATING via numpy forward (no TF) | dark→neutral .46; gradient→neutral .43/surprise .22; lena crop→neutral .47/fear .23; ~10ms/infer; zeroed-weights ablation→uniform (weights proven active) |
| AU geometry | rule-based from YuNet 5pt | OPERATING, labeled `observable-feature-analysis` | smile/brow tests pass |
| Body motion | frame-diff energy | TRACKING ONLY, labeled `motion-only` | static low / oscillating high tests pass |

## Verified behaviors (30 tests, `pytest -q` → all pass)
- `/api/model-info` reports trained-minixception-fer2013 loaded.
- Gray JPEG → `face_found:false`, fusion unknown>0.4.
- Corrupt upload → 422, never 500.
- No camera (index 99) → typed `WebcamError`.
- No synthetic face boxes anywhere (asserted on source).
- Frontend `npm run build` → dist/ ok (must run via `cmd /c npm…`; PowerShell blocks npm.ps1).

## Training pipeline (real, data-gated)
- `training/fer_csv.py` loads Kaggle FER2013 CSV (instructions in FileNotFoundError); `train_mlp.py` trains sklearn MLP (neural net) when CSV present.
- No FER2013 CSV in repo → NO accuracy claims made. Synthetic smoke (`--synthetic`) is labeled synthetic and only proves pipeline mechanics.
