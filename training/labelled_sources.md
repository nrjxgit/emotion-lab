# Labelled evaluation sources (verified 2026-10-16 from this environment)

| Dataset | Licence | Access | Outcome here |
|---|---|---|---|
| FER2013 | Kaggle ToU (free w/ account) | `pip install kaggle` + API token → `kaggle datasets download -d msambare/fer2013` (or `datablist/fer2013`) | kaggle CLI absent; needs user credentials. `training/fer_csv.py` loads it once present. |
| FER+ (土Barsoum relabels) | research (Microsoft) | github.com/microsoft/FERPlus (`fer2013new.csv` labels; images still from FER2013) | labels public; images gated on FER2013. |
| JAFFE | research request (Nakano/Kasai) | kasrl.org/jaffe.html request form | site unreachable from here (proxy SSL); human should apply in browser. |
| RAF-DB / AffectNet / Aff-Wild2 | research request each | whdeng.cn / michaelmahoor.com / ibug.doc.ic.ac.uk | request-gated; not attempted (identity + forms). |
| oarriaga demo images | repo licence (GPL-3.0) | github raw (verified reachable) | unlabeled — useful for smoke, NOT benchmarking. |

Rule: benchmark dicts always carry `dataset`; anything without one is not accuracy.
