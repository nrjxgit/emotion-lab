"""Probe CREMA-D repo for direct visual-media links."""

import json
import urllib.request

req = urllib.request.Request(
    "https://api.github.com/repos/CheyneyComputerScience/CREMA-D/contents",
    headers={"User-Agent": "research-audit"},
)
with urllib.request.urlopen(req, timeout=30) as r:
    items = json.load(r)
for i in items:
    print(i["type"], i["name"], i.get("size", ""))
