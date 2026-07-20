"""Candidate source that surfaces the issue_frame residue for labeling.

Reuses cartographer.escalation.escalation_candidates for the residue FILTER
(not same-story, cosine >= floor) so what the curator labels stays aligned with
what the escalation bar tests — then re-ranks the result before capping.

Ranking (corrected 2026-07-20 from real corpus_node4 evidence): real opposed
pairs on one proposition share dense vocabulary, so issue_frame lives at HIGH
cosine, not the mid band the module originally assumed. A strong NLI
contradiction is the sharpest instance, but framing/efficacy divergences (which
no cheap signal reliably flags — the reason the LLM escalation judge exists) are
also high-cosine. So the residue floats NLI contradictions first (by strength),
then ranks the rest cosine-descending. High-cosine AGREE pairs surface here too,
which is wanted — the readiness gauge also needs agree labels. Implements
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

from .candidates import CONTRA_THRESHOLD


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
        # Contradictions first (by strength), then cosine-descending: within the
        # already-filtered residue, high cosine IS the issue_frame zone, so the
        # cap keeps the highest-cosine pairs rather than mid-band cross-topic
        # noise. NLI runs once per residue pair here (cached); the injected
        # nli_scores keep this model-free under test.
        def rank_key(pair: tuple[LabeledChunk, LabeledChunk]):
            cos = self._proposer.embedding_cosine(pair[0].text, pair[1].text)
            p_contra = self._proposer.nli_p_contra(pair[0].text, pair[1].text)
            is_contra = p_contra >= CONTRA_THRESHOLD
            return (0 if is_contra else 1, -p_contra, -cos)

        residue.sort(key=rank_key)
        self._pairs = residue[: self._max_candidates]
        return self._pairs
