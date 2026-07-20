"""Where pairs to label come from, and in what order to show them.

Pluggable source (offline DB-free default; a Postgres/live-residue adapter is a
later addition behind the same Protocol). Ordering is a static hard-cases-first
heuristic over the cheap pipeline's own signals: signal disagreements first,
then the ambiguous mid-band, then everything obvious — so curator time buys the
most boundary signal. No retraining loop (that needs the bi-encoder to exist).
"""
from __future__ import annotations

from itertools import combinations
from typing import Protocol

from gin.cartographer.models import LabeledChunk

from .models import pair_key

GATE_FLOOR = 0.13
CORROBORATE_CEILING = 0.45
CONTRA_THRESHOLD = 0.5


class CandidateSource(Protocol):
    # True iff pairs() is already returned in the order it wants shown to a
    # curator (e.g. an evidence-based ranking) — callers must not re-sort it
    # through order_backlog. Default False for any source that doesn't
    # declare it; check via getattr(source, "pre_ranked", False).
    pre_ranked: bool = False

    def chunks(self) -> list[LabeledChunk]: ...
    def pairs(self) -> list[tuple[LabeledChunk, LabeledChunk]]: ...


class OfflineCandidateSource:
    """DB-free source over an in-memory chunk set (the default)."""

    pre_ranked = False

    def __init__(self, chunks: list[LabeledChunk]) -> None:
        self._chunks = list(chunks)

    def chunks(self) -> list[LabeledChunk]:
        return self._chunks

    def pairs(self) -> list[tuple[LabeledChunk, LabeledChunk]]:
        return list(combinations(self._chunks, 2))


def informativeness(sig: dict) -> float:
    """Static tier score; higher is more worth a curator's attention."""
    cos = sig.get("cosine") or 0.0
    p = sig.get("nli_p_contra")
    if p is not None and p >= CONTRA_THRESHOLD and cos >= CORROBORATE_CEILING:
        return 2.0  # signal disagreement: NLI says contradict, cosine says corroborate
    if GATE_FLOOR <= cos < CORROBORATE_CEILING:
        return 1.0  # ambiguous mid-band (includes the not-same-story residue)
    return 0.0


def pre_ranked_unlabeled_pairs(
    source: CandidateSource,
    already_labeled: set[tuple[str, str]],
) -> list[tuple[LabeledChunk, LabeledChunk]]:
    """The next-pairs decision for a pre_ranked source: walk source.pairs() in
    its own order, dropping already-labeled pairs — no re-sort. Kept separate
    from order_backlog (and from any HTTP/signals concerns) so the ordering
    decision is testable without a running app or a model."""
    return [
        (a, b)
        for a, b in source.pairs()
        if pair_key(a.chunk_id, b.chunk_id) not in already_labeled
    ]


def order_backlog(
    scored: list[tuple[tuple[LabeledChunk, LabeledChunk], dict]],
    already_labeled: set[tuple[str, str]],
) -> list[tuple[tuple[LabeledChunk, LabeledChunk], dict]]:
    unlabeled = [
        (pair, sig)
        for pair, sig in scored
        if pair_key(pair[0].chunk_id, pair[1].chunk_id) not in already_labeled
    ]
    unlabeled.sort(
        key=lambda item: (
            -informativeness(item[1]),
            -(item[1].get("cosine") or 0.0),
            item[0][0].chunk_id,
            item[0][1].chunk_id,
        )
    )
    return unlabeled
