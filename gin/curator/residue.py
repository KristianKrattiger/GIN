"""Candidate source that surfaces the issue_frame residue for labeling.

Reuses cartographer.escalation.escalation_candidates (the already-measured
residue: not same-story, cosine >= floor, cosine-sorted) so what the curator
labels stays aligned with what the escalation bar tests. Implements
sub-project A's CandidateSource protocol.
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional

from gin.cartographer.combined import CombinedRelationProposer
from gin.cartographer.escalation import (
    DEFAULT_ESCALATION_COS_FLOOR,
    DEFAULT_MAX_CANDIDATES,
    escalation_candidates,
)
from gin.cartographer.models import LabeledChunk
from gin.cartographer.scan import wire_same_story


class EscalationResidueCandidateSource:
    """A.CandidateSource over the escalation residue of a corpus."""

    def __init__(
        self,
        chunks: list[LabeledChunk],
        *,
        proposer: Optional[CombinedRelationProposer] = None,
        cos_floor: float = DEFAULT_ESCALATION_COS_FLOOR,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
    ) -> None:
        self._chunks = list(chunks)
        if proposer is None:
            proposer = CombinedRelationProposer()
        # escalation_candidates needs the stage-1 same-story provider; wire it
        # from this corpus unless one was injected (tests inject a fake).
        if proposer.same_story is None:
            wire_same_story(proposer, self._chunks)
        self._proposer = proposer
        self._cos_floor = cos_floor
        self._max_candidates = max_candidates

    def chunks(self) -> list[LabeledChunk]:
        return self._chunks

    def pairs(self) -> list[tuple[LabeledChunk, LabeledChunk]]:
        return escalation_candidates(
            combinations(self._chunks, 2),
            self._proposer,
            cos_floor=self._cos_floor,
            max_candidates=self._max_candidates,
        )
