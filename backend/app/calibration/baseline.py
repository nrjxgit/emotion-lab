"""Opt-in personal baseline (local JSON only)."""

import json, os


def _default_path():
    return os.environ.get("PERSONAL_BASELINE_PATH", "configs/personal_baseline.json")


class PersonalBaseline:
    def __init__(self, path=None):
        self.path = path or _default_path()
        self.data = {}
        if os.path.exists(self.path):
            try:
                self.data = json.load(open(self.path))
            except Exception:
                self.data = {}

    def update(self, aus: list):
        for a in aus:
            c = a.get("code")
            self.data[c] = (
                self.data.get(c, 0.0) * 0.9 + float(a.get("intensity", 0)) * 0.1
            )
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        json.dump(self.data, open(self.path, "w"))

    def apply(self, aus: list) -> list:
        out = []
        for a in aus:
            c = a.get("code")
            b = self.data.get(c, 0.0)
            out.append(
                {
                    **a,
                    "intensity": max(
                        0.0, min(1.0, float(a.get("intensity", 0)) - 0.5 * b)
                    ),
                }
            )
        return out
