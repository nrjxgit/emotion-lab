# ViT checkpoint forensics — `dima806/facial_emotions_image_detection`

- Source: Hugging Face, Apache-2.0; base `google/vit-base-patch16-224-in21k`.
- Native head order: sad, disgust, angry, neutral, fear, surprise, happy
  (mapping to FER7 pinned in `adapters.py` + `test_adapters.py`).
- Author's README claims ~91% accuracy on a balanced 25,170-image set
  (~3,596/class) and links a Kaggle ViT notebook — training set is very likely
  FER2013-derived (same source as our test set).
- Overlap status: UNRESOLVED from the card alone. All FER2013-test ViT numbers are
  therefore labeled **in-distribution, possible train overlap** — never
  generalisation evidence. Independent-set eval (CREMA-D visual) required.
- Our subset check (n=350, seed 0): ViT 0.897 vs mini-Xception 0.497 — consistent
  with the author's 0.91 claim, which raises (not lowers) the overlap suspicion.

## v0.5 primary-source audit (supersedes card-only inference above)

Exact revision: `747cf16692eea925b54b7b543cf436848128b68d`.
Safetensors SHA256: `ffad6b9e5cb20d4857f4b532c2543055d7d0e7e1422ab7469087edaab82f8dd8`.
Full sources/hashes/installed versions: `eval_results/v0.5/sources/provenance.json`.
Processor: 224x224 bilinear RGB, /255, channel mean=.5/std=.5 ([-1,1]).

The public notebook API returned version 8, dataset
`samaneheslamifar/facial-emotion-expressions`. Archived code:
`eval_results/v0.5/sources/author_notebook.py.txt`.

1. `glob('*/*/*.*')` pools source folders below `images/images/`.
2. `RandomOverSampler.fit_resample` duplicates paths **before splitting**.
3. `dataset.train_test_split(test_size=0.4, shuffle=True, stratify_by_column='label')`
   creates a new split instead of preserving official FER split membership.

This is direct evidence of a leakage-prone published workflow, not merely a high
accuracy suspicion. Historical checkpoint training IDs and an overlap hash list
were not released. Therefore exact membership remains unverifiable; all our FER
ViT scores are potentially contaminated/in-distribution. No superiority claim.

mini-Xception also has provenance limits: the archived upstream loader reads
every CSV row without filtering `Usage`, then makes a row-order 80/20 split.
For a standard 35,887-row FER CSV in official row order, the first 28,709 rows
are training, so this does **not** by itself prove test images were trained on.
However, the remaining 7,178 rows then serve as checkpoint-selection validation
(ModelCheckpoint/early stopping), overlapping our combined PublicTest+PrivateTest.
The exact historical CSV is not published. Our test split is untouched **by us**,
not proven untouched by either pretrained workflow. Distinguish possible model-
selection reuse (mini-Xception) from oversampling-before-splitting leakage (ViT).
