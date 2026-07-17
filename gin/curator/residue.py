"""Candidate source that surfaces the issue_frame residue for labeling.

Reuses cartographer.escalation.escalation_candidates for the residue FILTER
(not same-story, cosine >= floor) so what the curator labels stays aligned with
what the escalation bar tests — but re-ranks the result mid-band-first before
capping. escalation_candidates sorts cosine-descending, which buries the
moderate-cosine issue_frame divergent band under high-cosine same-topic AGREE
pairs; a cosine-desc cap would then truncate away the exact class B0 exists to
collect. Implements sub-project A's CandidateSource protocol.
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

from .candidates import informativeness


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
        # escalation_candidates needs the stage-1 same-story provider. This
        # no-ops when the proposer already has one (an injected fake), so it is
        # safe to call unconditionally.
        wire_same_story(proposer, self._chunks)
        self._proposer = proposer
        self._cos_floor = cos_floor
        self._max_candidates = max_candidates
        self._pairs: Optional[list[tuple[LabeledChunk, LabeledChunk]]] = None

    def chunks(self) -> list[LabeledChunk]:
        return self._chunks

    def pairs(self) -> list[tuple[LabeledChunk, LabeledChunk]]:
        # Static for a given corpus + proposer — compute once.
        if self._pairs is not None:
            return self._pairs
        all_pairs = list(combinations(self._chunks, 2))
        # Full residue (pass an unbounded cap so the cosine-desc slice inside
        # escalation_candidates doesn't drop the moderate-cosine band before we
        # re-rank).
        residue = escalation_candidates(
            all_pairs,
            self._proposer,
            cos_floor=self._cos_floor,
            max_candidates=len(all_pairs),
        )
        # Re-rank mid-band-first (A's informativeness, cosine-only — no NLI), so
        # the cap keeps the issue_frame divergent band rather than high-cosine
        # AGREE pairs. A's order_backlog re-ranks again downstream once NLI
        # signals exist; this only decides what survives the cap.
        def rank_key(pair: tuple[LabeledChunk, LabeledChunk]):
            cos = self._proposer.embedding_cosine(pair[0].text, pair[1].text)
            return (-informativeness({"cosine": cos, "nli_p_contra": None}), -cos)

        residue.sort(key=rank_key)
        self._pairs = residue[: self._max_candidates]
        return self._pairs
