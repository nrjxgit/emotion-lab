# Model adapter/export contract (no application merge)

`backend/app/face/adapters.py` contains lazy, independent model adapters:

```python
import numpy as np
from backend.app.face.adapters import get_adapter, FER7

model = get_adapter('mini-xception')  # legacy default, no Torch needed
face_bgr = np.zeros((64, 64, 3), dtype=np.uint8)  # substitute a REAL detected crop
probabilities = model.predict_proba(face_bgr)   # shape(7,), sum~1
label = FER7[int(probabilities.argmax())]
```

The example array demonstrates the input contract, not a valid face or prediction.
All adapters take **uint8 BGR HxWx3** and return FER7 order:
angry/disgust/fear/happy/sad/surprise/neutral. Unknown is an abstention policy,
not an eighth neuron in these checkpoints. No adapter predicts internal feelings.

- mini-Xception: NumPy/HDF5; unchanged legacy `/255` preprocessing. The corrected
  upstream range and frozen crop profiles are separately available in
  `training.crop_science.preprocess_crop` + `fer_model.predict_fer`. Do not change
  a pipeline's preprocessing without recording the profile.
- ViT: `get_adapter('vit')`, pinned HF revision
  `747cf16692eea925b54b7b543cf436848128b68d`; CPU Torch/Transformers required.
  RGB224 and checkpoint processor; native labels are explicitly remapped.
  Local preserved export: `models/vit/<revision>/` (weights/config/processor).
  Instantiate `ViTAdapter(model_id='models/vit/<revision>')` for an offline copy.
- HSEmotion: explicit `NotImplementedError`, not a dummy/fallback classifier.
  The independent Claude application remains separate. A future adapter must
  document head order, input size/channel/range, source/revision/licence and
  passing cross-dataset tests before consideration for default/merge.

Runtime dependencies: base NumPy/OpenCV/HDF5; optional Torch/Transformers for
ViT; TensorFlow used only for batched research/parity, not required by serving
mini-Xception. Exact measured versions and hashes are in
`eval_results/v0.5/sources/provenance.json`; the TF build was a release candidate.
Use an isolated virtual environment to reproduce it rather than changing another
application's dependencies.

Raw scores are not equally calibrated between adapters. The API default remains
mini-Xception; opt-in ViT is marked experimental and may download weights unless
cached. API comparison/selection and reproducible batch comparisons are separate:
`GET /api/models`, `POST /api/infer?model=vit`, `training.score_vit_full`.

Models and video archives are preserved locally, not all committed as large
binaries. Git records source code, checkpoint identities, metric JSONs, per-image
predictions and validation locks. Do not delete caches/exports while experiments
depend on them. Dataset licences are distinct from checkpoint/source-code terms.
