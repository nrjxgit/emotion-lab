# Dataset verification — v0.3 (checked 2026-10-16)

## Request-gated (not downloaded; forms/identity needed)
- **AffectNet**: request via mohammadmahoor.com/pages/databases/affectnet (largest wild VA+expr set).
- **RAF-DB**: official whdeng.cn request; Kaggle mirrors exist but are unofficial copies — prefer official for licensing clarity.
- **Aff-Wild2**: ABAW challenge request (ibug.doc.ic.ac.uk) — video VA+AU+expr, best future fusion benchmark.
- **JAFFE**: kasrl.org request form (Nakano group); 213 images, 10 Japanese female posers — small but freely licensed for research; good age/gender slice, NOT ethnically diverse alone.
- **BoLD (body)**: cydar.ist.psu.edu/emotionchallenge — ~10k in-the-wild bodily-expression clips + challenge baselines/pretrained models (see FG2021-BoLD repo). The body-emotion integration target: request access, train temporal head on pose tracks.

## Diversity coverage rule
No single dataset covers age×skin-tone×anatomy×gender×ethnicity. Plan: evaluate
per-dataset slices with `stratified_report`, report CIs, and NEVER treat a
demographic attribute as an input feature. Characteristics do not determine emotions.

## Acquired here: none
Sandbox blocks: Kaggle creds absent, kasrl TLS fails, request forms need a human.
`training/labelled_sources.md` holds per-source instructions.
