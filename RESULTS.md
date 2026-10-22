# Results summary

Every number below comes from a report in [eval_results/](eval_results/). "Real data" means the test ran on real images; none of it is a test on everyday, real-world emotions, and no result here has been checked by anyone outside the project.

## FER2013 (web images labelled by strangers)

| Measurement | Result | Source |
|---|---|---|
| Accuracy, original version, all 7,178 test faces | 56.0% (macro-F1 0.503) | `v0.4-report.md`, `fer_test_baseline.json` |
| Best and worst classes | happy 89% recall; fear 33%, sad 36% | `v0.4-report.md` |
| Preprocessing fix (pixel range and crop), same 350 faces | 49.7% → 65.7% | `v0.6-report.md`, `v0.6/old_vs_new.json` |
| Pixel range alone, same tight crops | +14.9 percentage points | `v0.5-report.md` |
| Corrected version, detected test faces | 65.9% | `v0.6-report.md` |
| Test images that exactly match a training image (decoded pixels) | 568 of 7,178 (7.9%); 67.3% accuracy on those vs 55.0% on the rest | `v0.6/overlap.json` |

## Saying "not sure" (original version, all 7,178 faces, temperature-scaled)

| Answers only when at least… | Share of faces answered | Accuracy on those |
|---|---|---|
| always answers | 100% | 56.0% |
| 50% sure | 59.7% | 69.3% |
| 70% sure | 35.3% | 82.3% |
| 90% sure | 18.8% | 93.3% |

Source: `calibration.json` (temperature T = 1.12, expected calibration error 0.083 → 0.054).

## CREMA-D (91 actors, 546 clips, six acted emotions)

| Measurement | Result | Source |
|---|---|---|
| mini-Xception accuracy | 29.7% (random guessing: 16.7%) | `v0.8-report.md` |
| ViT accuracy | 29.7% | `v0.8-report.md` |
| Clips where both models gave the same answer | 47.8% | `v0.8-report.md` |
| Calibration error, FER2013-fitted temperature | 0.387 → 0.376 | `v0.7-report.md` |
| Calibration error, temperature refitted on 30 separate actors | 0.285 → 0.050 | `v0.8-report.md`, `v0.8/paired_calibration.json` |

CREMA-D differs from FER2013 in more than the people: emotions are acted on instruction, faces come from video frames, and there are six labels instead of seven. The accuracy drop cannot be blamed on unfamiliar faces alone.

## Not yet tested

- Body-movement cues on real people (only synthetic tests so far).
- Personal calibration for one person.
- Any controlled study with consenting participants.

## The ViT comparison

The vision transformer scored 89.7% on a balanced 350-face FER2013 subset, but it could not be confirmed that its training data excludes the FER2013 test set. It is kept as experimental and is never the default. See `docs/VIT_FORENSICS.md`.
