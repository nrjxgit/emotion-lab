"""ViT checkpoint forensics: card, config, training data, overlap risk."""

import json
import urllib.request

MID = "dima806/facial_emotions_image_detection"
api = f"https://huggingface.co/api/models/{MID}"
with urllib.request.urlopen(
    urllib.request.Request(api, headers={"User-Agent": "research-audit"}), timeout=30
) as r:
    meta = json.load(r)
print("id:", meta.get("id"))
print("license:", (meta.get("cardData") or {}).get("license"))
print("tags:", meta.get("tags"))
print("siblings:", [s.get("rfilename") for s in meta.get("siblings", [])][:20])
print("card excerpt:")
print(((meta.get("cardData") or {}).get("model-index") or "no model-index"))
# README
try:
    with urllib.request.urlopen(
        f"https://huggingface.co/{MID}/raw/main/README.md", timeout=30
    ) as r:
        print("--- README (first 2000 chars) ---")
        print(r.read().decode("utf-8", "ignore")[:2000])
except Exception as e:
    print("README fetch failed:", e)
