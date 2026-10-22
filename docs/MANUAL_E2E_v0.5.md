# Consented end-to-end validation, v0.5

**Not yet performed on a live consenting participant in this session.** A prior
camera probe decoded nearly black frames; that is hardware/error-path evidence,
not expression-recognition performance. Do not start capture without opt-in.

1. Close other applications using the camera. Run `run_backend.bat`, check
   `http://127.0.0.1:8000/api/models` (default mini-Xception), then serve/open the
   standalone UI. Browser requires localhost/HTTPS for camera permission.
2. Explain purpose, uncertainty, model limitations and local processing. Obtain
   adult informed consent; clear the frame of bystanders. No raw frames retained
   unless a separate recording consent is documented. No consequential use.
3. Press Start; accept browser permission. Confirm model/source name is visible.
   Press Stop and confirm browser camera indicator turns off. Repeat with camera
   permission denied and backend unavailable; no stale prediction may be read as
   a fresh observation. These are manual pass/fail cases, not claimed automated
   browser-test results.
4. Allocate anonymous participant IDs to training/calibration/validation/test
   **before** recording. Never put frames from one participant in multiple splits.
   Keep all sessions/repetitions of one participant together. Held-out people
   must not contribute to threshold selection or temperature calibration.
5. For each consenting held-out participant, record intended posed expressions
   separately from voluntary feelings reports. Use neutral, smile, sadness-like,
   anger-like, surprise-like, fear-like and disgust-like poses without claiming
   any pose reveals an actual feeling. Record camera distance/resolution.
6. Cross conditions: diffuse frontal / side / dim light; yaw0/+30/-30 degrees;
   still / speaking / gentle head movement; glasses if already worn. Randomize
   order, repeat trials. Do not solicit distress or painful experiences.
7. For each frame/clip record detection success, box, model/checkpoint/profile,
   seven-class scores, acquisition time and condition ID. Run both models on
   the **same** frames/crops. Summarize by clip and person; do not treat thousands
   of adjacent frames as independent participants.
8. Metrics: conditional accuracy/F1/recall on common detected frames, end-to-end
   accuracy/F1 with misses included, detection rate, calibration error, latency
   p50/p95. Bootstrap participants, not individual frames. Keep ambiguous labels
   and abstention reasons visible. Self-report comparisons measure agreement,
   not that the participant is right/wrong about a private feeling.
9. Compare default vs opt-in ViT without changing default. It may download pinned
   weights on first load. Do not compare raw confidence as if equally calibrated.
   Body remains motion-only; no trained body-fusion accuracy can be inferred.
10. Export consented records, inspect error cases, then delete personal records
    if requested. Never publish raw webcam imagery or participant identities.

Infants, children and broad demographic accuracy are **not validated** by the
FER or CREMA-D experiments. Additional consent/developmental protocols required.
