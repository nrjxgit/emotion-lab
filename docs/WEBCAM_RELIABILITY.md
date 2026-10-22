# Webcam reliability — v0.6 (measured locally + manual procedure)

## Measured pipeline latency (CPU, 20 FER stills upscaled to 192px)
- YuNet detect: p50 3.9 ms, p95 7.9 ms. mini-X FER: p50 13.5 ms, p95 18.2 ms.
- Full frame ≈ 20 ms → ~50 fps headroom; dashboard polls every 800 ms.
- Script: `measure_latency.py` (uses `face/crop.py` clipping; empty crops skipped).

## Robustness behaviors (tested, not simulated)
- Out-of-frame detector boxes are filtered before cropping (`test_out_of_frame_box_never_500`).
- No face / poor quality / low confidence → explicit `insufficient_evidence` reasons.
- Largest face wins; additional people in frame are ignored (documented limitation).
- Camera absent/denied → typed error in UI; backend unreachable → UI message, no stale readings.

## Manual condition matrix (requires consenting participant; NOT yet run live)
See `docs/MANUAL_E2E_v0.5.md` + `docs/MANUAL_WEBCAM_PROTOCOL.md`: normal/dim/backlit
lighting, ±30° yaw, occlusion (glasses/hand/profile), motion, low resolution.
Record detection rate + prediction stability per cell. Prior dark-room probe proved
only the abstention path (0/5 faces, no images kept) — it is not performance evidence.

## Not claimed
Real-camera accuracy, skin-tone/proportion robustness, motion-blur tolerance:
no live-person measurements exist yet. Simulated-still numbers are not camera evidence.
