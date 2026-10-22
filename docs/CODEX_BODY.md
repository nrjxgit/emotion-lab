# Codex body-language integration (emolab.adapters)

The Codex body-language project lives in a separate repository and is NOT
merged into this repo. Its model is accessed through an adapter stub that
raises a clear error until the repo is available locally and its weights/
licence are verified.

## Expected repository layout (user-provided paths)

The Codex body-language repo is expected at a user-specified path, e.g.
`/path/to/codex-body`. Expected layout:

```
codex-body/
├── models/
│   ├── body_language/
│   │   ├── weights.pt / .safetensors
│   │   ├── config.json
│   │   └── LICENSE
│   └── LICENSE (repository-level)
├── datasets/
│   └── body_language/
└── README.md
```

## Integration plan (not yet implemented)

1. `emolab/adapters/codex_body.py` — wrap Codex body-language model.
   Currently raises `NotImplementedError` pending licence/weights review.
2. `emolab/adapters/fusion_stub.py` — face+body side-by-side stub (no fusion
   weights, no improvement claim until matched labelled evaluation
   demonstrates it).

## Licence requirements

Same as Claude: weights file present, licence file permitting research use,
checksum recorded.

## Fusion stance

The fusion stub returns face and body predictions side-by-side. It does NOT
combine them or claim improvement. Any claim that combined face+body analysis
outperforms face-only requires matched labelled evaluation demonstrating it.
