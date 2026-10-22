"""RAVDESS and CREMA-D adapter DECLARATIONS (EL-006 S2): metadata only, no file reading.

Both always report blocked until an operator rights/scope decision is recorded
as a reviewed source change. Their data methods can never return anything.
"""
from __future__ import annotations

from .rights import DatasetAdapter, RightsRefusal

BLOCKED = "blocked: needs operator rights/scope decision"


class _Declaration(DatasetAdapter):
    blocked_reason = BLOCKED
    allowed_uses = ("research_evaluation", "research_training")
    source_class = "research_corpus"
    scope = "research_noncommercial"

    def _participants(self) -> list:
        raise RightsRefusal("BLOCKED_NEEDS_OPERATOR_DECISION", BLOCKED)

    def _items(self) -> list:
        raise RightsRefusal("BLOCKED_NEEDS_OPERATOR_DECISION", BLOCKED)

    def label_map(self) -> dict:
        raise RightsRefusal("BLOCKED_NEEDS_OPERATOR_DECISION", BLOCKED)

    def group_id(self, item) -> str:
        raise RightsRefusal("BLOCKED_NEEDS_OPERATOR_DECISION", BLOCKED)


class PilotDeclaration(DatasetAdapter):
    """The consented pilot (S9pilot protocol). Declaration only: no data exists, none can be read.

    ``ethics_approval_token`` is deliberately None. Setting it to a string satisfies only the gate's field check; it
    is NOT an approval and nothing here verifies one. The allowlist entry stays blocked until the operator decides.
    """

    name = "pilot_consented"
    licence_tag = "consented-pilot (terms set by the pilot consent form)"
    source_class = "consented_pilot"
    allowed_uses = ("research_evaluation",)
    scope = "research_noncommercial"
    blocked_reason = "blocked: needs operator rights/scope decision and an ethics approval"
    ethics_approval_token = None

    def _participants(self) -> list:
        raise RightsRefusal("BLOCKED_NEEDS_OPERATOR_DECISION", self.blocked_reason)

    def _items(self) -> list:
        raise RightsRefusal("BLOCKED_NEEDS_OPERATOR_DECISION", self.blocked_reason)

    def label_map(self) -> dict:
        raise RightsRefusal("BLOCKED_NEEDS_OPERATOR_DECISION", self.blocked_reason)

    def group_id(self, item) -> str:
        raise RightsRefusal("BLOCKED_NEEDS_OPERATOR_DECISION", self.blocked_reason)


class RavdessDeclaration(_Declaration):
    name = "ravdess"
    licence_tag = "CC BY-NC-SA 4.0"


class CremaDDeclaration(_Declaration):
    name = "crema_d"
    licence_tag = "ODbL 1.0 / DbCL 1.0"
