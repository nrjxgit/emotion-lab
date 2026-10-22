"""Dataset inventory — no downloads, only verified licence/access metadata."""

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Manually verified Oct 2026: existence + licence summary. Access URLs in docs.
DATASET_CATALOG = [
    {
        "name": "AffectNet",
        "license": "research-only request",
        "access": "request http://mohammadmahoor.com/affectnet/",
        "labels": ["8 expr", "valence/arousal"],
        "notes": "1M in-the-wild, diverse; manual request",
    },
    {
        "name": "RAF-DB",
        "license": "research request",
        "access": "request http://www.whdeng.cn/RAF/model1.html",
        "labels": ["7 basic + compound"],
        "notes": "~30k Flickr diverse",
    },
    {
        "name": "FER2013",
        "license": "Kaggle ToU (free)",
        "access": "kaggle: challenges/in/sam-fer2013",
        "labels": ["7 expr 48x48"],
        "notes": "35k; noisy labels; baseline only",
    },
    {
        "name": "FER+",
        "license": "research (Microsoft)",
        "access": "github microsoft/FERPlus",
        "labels": ["8 expr relabeled"],
        "notes": "FER2013 relabel, better quality",
    },
    {
        "name": "ExpW",
        "license": "research",
        "access": "http://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html",
        "labels": ["7 expr"],
        "notes": "~92k in-wild",
    },
    {
        "name": "CK+",
        "license": "research request",
        "access": "http://www.jeffcohn.net/Resources/",
        "labels": ["7 expr + AU"],
        "notes": "lab posed sequences",
    },
    {
        "name": "DISFA",
        "license": "research request",
        "access": "http://www.engr.pitt.edu/~emotion/ausystem.htm",
        "labels": ["AU intensity"],
        "notes": "spontaneous AU gold standard",
    },
    {
        "name": "BP4D",
        "license": "research request",
        "access": "http://www.cs.binghamton.edu/~lijun/Research/3DFE.html",
        "labels": ["AU + 3D"],
        "notes": "41 subjects spontaneous",
    },
    {
        "name": "EmotioNet",
        "license": "research",
        "access": "http://cbcsl.ece.ohio-state.edu/enc/",
        "labels": ["AU + expr 1M"],
        "notes": "large-scale auto+manual AU",
    },
    {
        "name": "Aff-Wild2",
        "license": "research request (ABAW)",
        "access": "https://ibug.doc.ic.ac.uk/resources/aff-wild2/",
        "labels": ["expr + VA + AU video"],
        "notes": "in-wild video; VA continuous",
    },
    {
        "name": "AFEW/DFEW/FERV39k",
        "license": "research (EmotiW)",
        "access": "https://dfew-dataset.github.io/",
        "labels": ["video expr"],
        "notes": "film-clip dynamic; cross-clip eval",
    },
    {
        "name": "RAVDESS",
        "license": "CC-BY-NC-SA",
        "access": "https://zenodo.org/record/1188976",
        "labels": ["8 sung/spoken expr"],
        "notes": "24 actors multimodal",
    },
    {
        "name": "CREMA-D",
        "license": "research",
        "access": "https://github.com/CheyneyComputerScience/CREMA-D",
        "labels": ["6 expr AV"],
        "notes": "7442 clips diverse actors",
    },
    {
        "name": "FABO/BIWI/UIEB",
        "license": "varies research",
        "access": "see docs/DATASETS.md",
        "labels": ["body/pose"],
        "notes": "body-expression candidates; verify per-use",
    },
]
