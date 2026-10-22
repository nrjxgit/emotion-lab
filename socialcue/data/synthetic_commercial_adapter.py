"""Second reviewed synthetic adapter; no real data or product training authority."""
from .. import synthetic
from .synthetic_adapter import SyntheticAdapter


class SyntheticCommercialCleanAdapter(SyntheticAdapter):
    """Manufactured streams through the same protocol, distinct provenance/seed."""

    name = "synthetic_commercial_clean"
    licence_tag = "synthetic-commercial-clean"
    source_class = "synthetic"
    allowed_uses = ("commercial_training", "research_evaluation")
    scope = "commercial_clean"
    blocked_reason = None

    def __init__(self, n_participants=6, seed=1505, scenarios=synthetic.SCENARIOS):
        super().__init__(n_participants=n_participants, seed=seed, scenarios=scenarios)
