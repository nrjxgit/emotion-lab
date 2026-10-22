"""SyntheticAdapter: original reviewed manufactured-data adapter (EL-006 S2; second adapter added in S50)."""
from __future__ import annotations

from dataclasses import dataclass

from .. import synthetic
from .rights import DatasetAdapter

_PROXY = {"calm": "proxy_negative", "nervous-like": "proxy_positive"}


@dataclass(frozen=True)
class SyntheticItem:
    item_id: str
    group: str
    scenario: str
    seed: int

    def records(self) -> list:
        return synthetic.generate(self.scenario, seed=self.seed)


class SyntheticAdapter(DatasetAdapter):
    name = "synthetic"
    licence_tag = "synthetic-manufactured"
    source_class = "synthetic"
    allowed_uses = ("development_test", "research_evaluation", "research_training",
                    "commercial_evaluation", "commercial_training")
    scope = "synthetic"
    blocked_reason = None

    def __init__(self, n_participants: int = 6, seed: int = 505, scenarios=synthetic.SCENARIOS):
        self._n = int(n_participants)
        self._seed = int(seed)
        self._scenarios = tuple(scenarios)

    def _participants(self) -> list:
        return [f"synthetic-actor-{i:02d}" for i in range(self._n)]

    def _items(self) -> list:
        out = []
        for i, actor in enumerate(self._participants()):
            for j, sc in enumerate(self._scenarios):
                out.append(SyntheticItem(f"{actor}/{sc}", actor, sc, self._seed * 1000 + i * 10 + j))
        return out

    def label_map(self) -> dict:
        return {sc: _PROXY.get(sc, "excluded") for sc in self._scenarios}

    def group_id(self, item) -> str:
        return item.group
