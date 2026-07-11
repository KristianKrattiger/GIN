"""Cheap relatedness gate — stage 1 of the Cartographer.

Cuts the O(n²) pair space to candidate pairs worth the expensive relation-type
stage, and emits ``unrelated`` assessments (stored negatives) for the rest. This
stage MAY use the relevance signals the rest of the system has (IDF-weighted token
overlap here); the *relation-type* detector may not (design §2). Lexical/entity
first for a deterministic, model-free harness; embeddings are the production
upgrade. See docs/nc_cartographer_design.plan.md.
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterable

from gin.corpus.relevance import _norm_tokens, corpus_idf

from .models import Assessment, LabeledChunk, Relation

# Overlap-coefficient floor: shared IDF mass as a fraction of the lighter chunk's
# mass. A distinctive shared entity ("wildfire") clears it; incidental generic
# overlap does not. Symmetric, so relatedness has no direction.
DEFAULT_RELATEDNESS_FLOOR = 0.20


def _idf_mass(text: str, idf: dict[str, float]) -> float:
    return sum(idf.get(t, 0.0) for t in _norm_tokens(text))


def idf_relatedness(a_text: str, b_text: str, idf: dict[str, float]) -> float:
    """Symmetric IDF-weighted overlap coefficient in [0, 1].

    shared_mass / min(mass_a, mass_b): rewards distinctive shared tokens, and is
    stable when the two chunks differ greatly in length (a short grassroots line
    vs. a long institutional paragraph).
    """
    a_tokens, b_tokens = _norm_tokens(a_text), _norm_tokens(b_text)
    shared = a_tokens & b_tokens
    shared_mass = sum(idf.get(t, 0.0) for t in shared)
    a_mass = sum(idf.get(t, 0.0) for t in a_tokens)
    b_mass = sum(idf.get(t, 0.0) for t in b_tokens)
    denom = min(a_mass, b_mass)
    if denom <= 0:
        return 0.0
    return min(1.0, shared_mass / denom)


class RelatednessGate:
    """Stage 1: partition chunk pairs into candidates vs. stored negatives."""

    def __init__(
        self,
        chunks: Iterable[LabeledChunk],
        *,
        floor: float = DEFAULT_RELATEDNESS_FLOOR,
        idf_corpus: Iterable[str] | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.floor = floor
        texts = list(idf_corpus) if idf_corpus is not None else [c.text for c in self.chunks]
        self.idf = corpus_idf(texts)

    def assess_pair(self, a: LabeledChunk, b: LabeledChunk) -> Assessment:
        score = idf_relatedness(a.text, b.text, self.idf)
        related = score >= self.floor
        return Assessment(
            src_chunk_id=a.chunk_id,
            dst_chunk_id=b.chunk_id,
            relation=Relation.RELATED_UNTYPED if related else Relation.UNRELATED,
            method="relatedness_gate:idf_overlap",
            confidence=score,
            rationale=(
                f"idf overlap {score:.3f} "
                f"{'>=' if related else '<'} floor {self.floor:.2f}"
            ),
        )

    def assess_all(self) -> list[Assessment]:
        """One assessment per unordered pair — negatives included (stored)."""
        return [
            self.assess_pair(a, b) for a, b in combinations(self.chunks, 2)
        ]

    def candidates(self) -> list[Assessment]:
        return [a for a in self.assess_all() if a.relation == Relation.RELATED_UNTYPED]

    def negatives(self) -> list[Assessment]:
        return [a for a in self.assess_all() if a.relation == Relation.UNRELATED]
