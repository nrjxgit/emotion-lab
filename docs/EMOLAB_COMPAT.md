# emolab.inference/1.0 compatibility (proposed interface for the three systems)

No HSEmotion/Codex repositories are available locally; nothing is merged.
This document specifies the contract any implementation must satisfy so future
evaluation and integration are mechanical.

## Endpoint shape (all systems)
- Input: single BGR uint8 frame (HxWx3) + `frame_id` + `timestamp_ms`.
- Output per face: `box_xywh`, `probs` (named classes), `top1`, `confidence`,
  `uncertainty`, `model_id`, `checkpoint_revision`, `preprocessing_profile`,
  `calibrated` (bool), `inference_ms`, plus `insufficient_evidence[]` reasons.
- Class contract: probabilities keyed by explicit names; adapters MUST publish a
  `native_labels` order and a mapping to the comparison set. No silent remapping.
- Provenance: weights file + SHA256 + source URL + licence must accompany any score.

## Conformance checklist (for HSEmotion / Codex systems when repos arrive)
1. Same clips, same crops, same class definitions, same metrics as `training/compare_clips.py`.
2. Revision-pinned weights; result arrays committed with sample IDs.
3. Calibration/abstention validated on disjoint actors, never the test actors.
4. Temporal aggregation declared (single-frame vs mean-N) before scoring.
5. Report coverage alongside accuracy wherever abstention is used.

## Current status
- This repo conforms for mini-Xception + ViT on FER2013-test and CREMA-D/DFA pilot.
- HSEmotion: explicit `NotImplementedError` adapter; no code copied, no merge.
