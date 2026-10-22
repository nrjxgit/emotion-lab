"""Archive public provenance evidence, not private data or credentials."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval_results/v0.5/sources"
MID = "dima806/facial_emotions_image_detection"
REV = "747cf16692eea925b54b7b543cf436848128b68d"


def get(url):
    req = urllib.request.Request(
        url, headers={"User-Agent": "expression-research-audit"}
    )
    return urllib.request.urlopen(req, timeout=60).read()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    documents = {}
    for filename in ("README.md", "config.json", "preprocessor_config.json"):
        documents["vit_" + filename] = (
            f"https://huggingface.co/{MID}/resolve/{REV}/{filename}"
        )
    github = json.loads(
        get("https://api.github.com/repos/oarriaga/face_classification/commits/master")
    )["sha"]
    for filename in (
        "src/utils/preprocessor.py",
        "src/utils/datasets.py",
        "src/video_emotion_color_demo.py",
        "src/train_emotion_classifier.py",
    ):
        documents["oarriaga_" + Path(filename).name] = (
            f"https://raw.githubusercontent.com/oarriaga/face_classification/{github}/{filename}"
        )
    sources = []
    for filename, url in documents.items():
        data = get(url)
        (OUT / filename).write_bytes(data)
        sources.append(
            {"file": filename, "url": url, "sha256": hashlib.sha256(data).hexdigest()}
        )
    url = "https://www.kaggle.com/api/v1/kernels/pull/dima806/facial-emotions-image-detection-vit"
    kernel = json.loads(get(url))
    notebook = json.loads(kernel["blob"]["sourceNullable"])
    source = "\n\n".join(
        "".join(c.get("source", []))
        for c in notebook["cells"]
        if c["cell_type"] == "code"
    )
    (OUT / "author_notebook.py.txt").write_text(source, encoding="utf-8")
    (OUT / "author_notebook_metadata.json").write_text(
        json.dumps(kernel["metadata"], indent=2)
    )
    sources.append(
        {
            "file": "author_notebook.py.txt",
            "url": url,
            "sha256": hashlib.sha256(source.encode()).hexdigest(),
            "version": kernel["metadata"].get("currentVersionNumber"),
        }
    )
    cache = (
        Path.home()
        / ".cache/huggingface/hub/models--dima806--facial_emotions_image_detection/snapshots"
        / REV
    )
    weights = {}
    for name in ("model.safetensors", "config.json", "preprocessor_config.json"):
        p = cache / name
        weights[name] = {
            "bytes": p.stat().st_size,
            "sha256": hashlib.file_digest(open(p, "rb"), "sha256").hexdigest(),
        }
    versions = {}
    for name in (
        "numpy",
        "opencv-python",
        "tensorflow-cpu",
        "torch",
        "transformers",
        "scikit-learn",
        "h5py",
        "huggingface-hub",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    evidence = {
        "vit_id": MID,
        "vit_revision": REV,
        "weights": weights,
        "oarriaga_source_revision": github,
        "sources": sources,
        "versions": versions,
        "overlap_evidence": [
            "notebook glob */*/*.* pools folders",
            "RandomOverSampler before train_test_split",
            "official source FER loader ignores Usage column",
        ],
        "warning": "Sources show risk; exact historical weight-training membership not released.",
    }
    (OUT / "provenance.json").write_text(json.dumps(evidence, indent=2))
    print("Sources archived", versions, flush=True)


if __name__ == "__main__":
    main()
