# Claude Code Emotion Lab integration (emolab.adapters)

The Claude Code Emotion Lab is a separate repository and is NOT merged into
this repo. Its models are accessed through emolab.adapters stubs that raise a
clear error until the repo is available locally and its weights/licence are
verified.

## Expected repository layout (user-provided paths)

The Claude Code Emotion Lab is expected at a user-specified path, e.g.
`/path/to/emotion-lab`. Expected layout:

```
emotion-lab/
├── models/
│   ├── hsemotion/
│   │   ├── weights.h5 / .pt / .safetensors
│   │   ├── config.json
│   │   └── LICENSE
│   ├── fer2013_cnn/
│   │   ├── weights.h5 / .pt
│   │   └── LICENSE
│   └── combined_fer2013_cremad/
│       ├── weights.h5 / .pt
│       └── LICENSE
├── datasets/
│   ├── fer2013/
│   └── cremad/
└── LICENSE.md (repository-level licence)
```

## Integration plan (not yet implemented)

1. `emolab/adapters/hsemotion.py` — wrap HSEmotion pretrained model.
   Currently raises `NotImplementedError` pending licence/weights review.
2. `emolab/adapters/claude_fer2013_cnn.py` — wrap FER2013-trained CNN.
   Raises `NotImplementedError` pending review.
3. `emolab/adapters/claude_combined.py` — wrap FER2013+CREMA-D head.
   Raises `NotImplementedError` pending review.

## Licence requirements

Before any Claude adapter is activated, the following must be verified:
* Weights file exists and is readable.
* Licence file present and permits research use (MIT, Apache-2.0, BSD-3,
  GPL-3.0-only acceptable; proprietary/"all rights reserved" rejected).
* Weight checksum recorded for reproducibility.

No Claude model output is presented in any benchmark result until weights,
licence, and reproducibility are verified.

## Adapter contract

All Claude adapters implement `emolab.inference.FaceEmotionAdapter`:
`predict_proba(face_rgb: np.ndarray) -> ModelResult` returning FER7-ordered
probabilities with provenance metadata. See `emolab.inference.adapter`.
