"""Cartographer data types — proposals and first-class negatives.

The Cartographer *proposes* typed epistemic edges and records what it assessed,
including pairs found unrelated. Nothing here writes canonical graph state; that
is the Bookkeeper's sole job. See docs/nc_cartographer_design.plan.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Relation(str, Enum):
    """Verdict a Cartographer assessment can carry.

    ``UNRELATED`` and ``RELATED_UNTYPED`` are stored graph content (negatives /
    relatedness-only results), not silence — they stop the same null being
    re-litigated every query and seed the federation cache.
    """

    CONTRADICTS = "contradicts"
    CORROBORATES = "corroborates"
    SUPERSEDES = "supersedes"
    RELATED_UNTYPED = "related_untyped"  # passed relatedness gate, not yet typed
    UNRELATED = "unrelated"              # assessed and found unrelated


# Relations that become admissible EdgeProposals (typed, non-negative).
TYPED_EDGE_RELATIONS = frozenset(
    {Relation.CONTRADICTS, Relation.CORROBORATES, Relation.SUPERSEDES}
)

# Relation types the graph layer carries as canonical edges today (models.EdgeType).
GRAPH_EDGE_RELATIONS = frozenset({Relation.CONTRADICTS, Relation.SUPERSEDES})


@dataclass(frozen=True)
class Assessment:
    """One pairwise assessment — the atomic unit the Cartographer emits.

    Every assessed pair produces exactly one Assessment, including negatives.
    ``method`` records which stage/signal produced the verdict, so relatedness-gate
    and relation-detector outputs remain distinguishable when the graph is audited.
    """

    src_chunk_id: str
    dst_chunk_id: str
    relation: Relation
    method: str
    confidence: float = 0.0
    rationale: str = ""

    @property
    def is_typed_edge(self) -> bool:
        return self.relation in TYPED_EDGE_RELATIONS


@dataclass(frozen=True)
class EdgeProposal:
    """A typed edge the Cartographer proposes for Bookkeeper adjudication.

    Carries optional token-offset sentence anchors now (design §5 / divergence
    plan §7.1 option b) so the schema need not be migrated after multi-sentence
    ingest. Anchors are proposed; the Bookkeeper verifies and stamps them.
    """

    src_chunk_id: str
    dst_chunk_id: str
    relation: Relation
    method: str
    confidence: float = 0.0
    rationale: str = ""
    src_anchor: Optional[tuple[int, int]] = None
    dst_anchor: Optional[tuple[int, int]] = None

    @classmethod
    def from_assessment(
        cls,
        assessment: Assessment,
        *,
        src_anchor: Optional[tuple[int, int]] = None,
        dst_anchor: Optional[tuple[int, int]] = None,
    ) -> "EdgeProposal":
        if not assessment.is_typed_edge:
            raise ValueError(
                f"cannot propose an edge from a {assessment.relation.value} assessment"
            )
        return cls(
            src_chunk_id=assessment.src_chunk_id,
            dst_chunk_id=assessment.dst_chunk_id,
            relation=assessment.relation,
            method=assessment.method,
            confidence=assessment.confidence,
            rationale=assessment.rationale,
            src_anchor=src_anchor,
            dst_anchor=dst_anchor,
        )


@dataclass(frozen=True)
class LabeledChunk:
    """A chunk the Cartographer reasons over (id + text is all the gate needs)."""

    chunk_id: str
    text: str
