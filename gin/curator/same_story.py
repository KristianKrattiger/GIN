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
        self._cached_pairs: Optional[list[tuple[LabeledChunk, LabeledChunk]]] = None

    def chunks(self) -> list[LabeledChunk]:
        return list(self._chunks)

    def _score(self, a_text: str, b_text: str) -> float:
        # A pair with an unusable score must still reach the curator: retention
        # matters more than ranking. Treat None or a raising scorer as 0.0
        # (unranked, bottom of the descending sort) rather than losing the pair.
        try:
            score = self._p_contra(a_text, b_text)
        except Exception:
            return 0.0
        return 0.0 if score is None else score

    def pairs(self) -> list[tuple[LabeledChunk, LabeledChunk]]:
        # Memoized: a real model-backed proposer must not re-score every pair
        # on every page request from the curator app.
        if self._cached_pairs is not None:
            return self._cached_pairs

        scored: list[tuple[float, tuple[LabeledChunk, LabeledChunk]]] = []
        for a, b in combinations(self._chunks, 2):
            if not self._same_story(a.text, b.text):
                continue
            scored.append((self._score(a.text, b.text), (a, b)))
        scored.sort(key=lambda row: -row[0])

        total = len(scored)
        cap = self._max_candidates
        if total <= cap:
            selected = scored
        else:
            # Band-aware truncation: a plain descending slice always cuts the
            # tail, which is precisely where the same-story negatives (low
            # p_contra) live. Split the cap between the top (conflicts) and
            # bottom (negatives) bands so truncation can never eliminate the
            # negatives outright, then re-sort descending so conflicts still
            # reach the curator first.
            top_n = cap // 2
            bottom_n = cap - top_n
            top = scored[:top_n]
            bottom = scored[total - bottom_n :]
            selected = sorted(top + bottom, key=lambda row: -row[0])

        self._cached_pairs = [pair for _score, pair in selected]
        return self._cached_pairs
