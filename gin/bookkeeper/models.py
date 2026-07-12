"""Bookkeeper data types — admission outcomes and canonical stamped edges.

The Bookkeeper is the *sole writer* of canonical graph state. It does not propose
edges (the Cartographer does); it verifies anchor integrity, enforces DAG
invariants, stamps provenance, and admits or denies. See
docs/nc_cartographer_design.plan.md and GIN_Session_Synthesis_v1.md §1.2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from gin.cartographer.models import Relation

# Relations that carry a direction / ordering, so a cycle among them is a
# contradiction the Bookkeeper must refuse. ``contradicts`` / ``corroborates`` are
# symmetric and carry no ordering.
ORDERING_RELATIONS = frozenset({Relation.SUPERSEDES})
SYMMETRIC_RELATIONS = frozenset({Relation.CONTRADICTS, Relation.CORROBORATES})


class AdmissionCode(str, Enum):
    ADMITTED = "admitted"
    DENIED_UNKNOWN_CHUNK = "denied_unknown_chunk"
    DENIED_SELF_LOOP = "denied_self_loop"
    DENIED_INVALID_ANCHOR = "denied_invalid_anchor"
    DENIED_CYCLE = "denied_cycle"
    DENIED_DUPLICATE = "denied_duplicate"
    DENIED_LOW_CONFIDENCE = "denied_low_confidence"
    DENIED_RELATION_MISMATCH = "denied_relation_mismatch"


@dataclass(frozen=True)
class Provenance:
    """Stamped when an edge is admitted — who proposed it, how sure, when."""

    proposer: str
    confidence: float
    admitted_at: str  # ISO-8601 UTC
    content_hash: str


@dataclass(frozen=True)
class AdmittedEdge:
    """Canonical graph edge. Only the Bookkeeper constructs these."""

    src_chunk_id: str
    dst_chunk_id: str
    relation: Relation
    provenance: Provenance
    src_anchor: Optional[tuple[int, int]] = None
    dst_anchor: Optional[tuple[int, int]] = None

    @property
    def is_symmetric(self) -> bool:
        return self.relation in SYMMETRIC_RELATIONS


@dataclass(frozen=True)
class AdmissionResult:
    code: AdmissionCode
    reason: str = ""
    edge: Optional[AdmittedEdge] = None

    @property
    def admitted(self) -> bool:
        return self.code == AdmissionCode.ADMITTED


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
