"""Candidate source over same-story pairs — node5's counterpart to the residue source.

EscalationResidueCandidateSource cannot serve this corpus for two independent
reasons. It filters TO the anchor-less residue (escalation_candidates returns
"pairs the cheap path cannot type: not same-story"), and it ranks mid-band cosine
first, so same-story pairs — high cosine from a shared lede — would rank last
even if they survived the filter.

Ranking is NLI-contradiction-descending so genuine conflicts reach the curator
before the negatives. Note it RANKS but never FILTERS on p_contra: the negatives
are the reason this corpus exists, and dropping low-p_contra pairs would discard
exactly the rows that can falsify combined.py's unconditional
"same_story => CONTRADICTS" branch.
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable, Optional

from gin.cartographer.models import LabeledChunk

DEFAULT_MAX_CANDIDATES = 2000


class SameStoryCandidateSource:
    """A.CandidateSource over pairs the stage-1 story predicate accepts."""

    # pairs() returns an evidence-based ranking; app.next_pairs must not re-sort.
    pre_ranked = True

    def __init__(
        self,
        chunks: list[LabeledChunk],
        *,
        same_story: Optional[Callable[[str, str], bool]] = None,
        p_contra: Optional[Callable[[str, str], float]] = None,
        proposer=None,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
    ) -> None:
        self._chunks = list(chunks)
        self._max_candidates = max_candidates
        # Only construct a proposer when the caller supplied no predicate. An
        # injected same_story must never drag a model load into a test.
        if same_story is None and proposer is None:
            from gin.cartographer.combined import CombinedRelationProposer

            proposer = CombinedRelationProposer()
        if same_story is None:
            if proposer.same_story is None:
                raise ValueError(
                    "SameStoryCandidateSource needs a same-story provider: pass "
                    "same_story=, or a proposer with scan.wire_same_story applied"
                )
            same_story = proposer.same_story
        if p_contra is None:
            # No scorer and no proposer means no ranking evidence: keep every
            # pair (the negatives are the point) in a stable, unranked order.
            p_contra = (
                (lambda a, b: proposer._p_contra(a, b))  # noqa: SLF001
                if proposer is not None
                else (lambda a, b: 0.0)
            )
        self._same_story = same_story
        self._p_contra = p_contra

    def chunks(self) -> list[LabeledChunk]:
        return self._chunks

    def pairs(self) -> list[tuple[LabeledChunk, LabeledChunk]]:
        scored: list[tuple[float, tuple[LabeledChunk, LabeledChunk]]] = []
        for a, b in combinations(self._chunks, 2):
            if not self._same_story(a.text, b.text):
                continue
            scored.append((self._p_contra(a.text, b.text), (a, b)))
        scored.sort(key=lambda row: -row[0])
        return [pair for _score, pair in scored[: self._max_candidates]]
