import json
import urllib.request

for path in ["VideoFlash", "VideoDemographics.csv"]:
    url = f"https://api.github.com/repos/CheyneyComputerScience/CREMA-D/contents/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "research-audit"})
    with urllib.request.urlopen(req, timeout=30) as r:
        items = json.load(r)
    if isinstance(items, dict):
        items = [items]
    print("==", path, len(items), "entries")
    for i in items[:12]:
        print(
            " ",
            i.get("type"),
            i.get("name"),
            i.get("size", ""),
            (i.get("download_url") or "")[:80],
        )
