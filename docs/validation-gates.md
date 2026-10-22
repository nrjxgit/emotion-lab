# Validation gates — default-model change criteria

The serving default stays `mini-xception` until a challenger passes ALL gates on
an independent (non-FER2013) labeled set with participant-disjoint splits:

1. Higher macroF1 than baseline with non-overlapping 95% CIs (bootstrap, n=1000).
2. ECE equal or lower after temperature scaling (T fit on validation only).
3. No single class below 0.30 recall (no collapsed classes).
4. Training-data overlap statement published (what the challenger trained on).
5. YuNet-crop eval repeated (deployment path, not pre-cropped stills).

HSEmotion merge additionally requires: license review + adapter conformance tests.
