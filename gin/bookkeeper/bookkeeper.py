"""The Bookkeeper — sole admission gate for canonical graph edges.

Takes Cartographer ``EdgeProposal``s and admits or denies each against a uniform
gate (no separate local/federated trust path): confidence floor, endpoint
existence, no self-loops, anchor integrity, deduplication, and DAG acyclicity for
ordering relations. On admission it stamps provenance and is the *only* thing that
writes to ``GraphState``. Its stored decisions double as the federation cache.

Falsifiable on its own terms (GIN_Session_Synthesis_v1.md §1.5): invariant
maintenance — no cycles, anchor integrity, correct admit/deny — independent of
Cartographer edge quality or reasoning-layer behaviour.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Mapping, Optional

from gin.cartographer.models import EdgeProposal, Relation

from .graph import GraphState
from .models import (
    AdmissionCode,
    AdmissionResult,
    AdmittedEdge,
    Provenance,
    now_iso,
)

# chunk_id -> token count, so anchor offsets can be range-checked.
ChunkRegistry = Mapping[str, int]


def _content_hash(proposal: EdgeProposal) -> str:
    payload = "|".join(
        str(x)
        for x in (
            proposal.relation.value,
            proposal.src_chunk_id,
            proposal.dst_chunk_id,
            proposal.src_anchor,
            proposal.dst_anchor,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _anchor_ok(anchor: Optional[tuple[int, int]], token_count: int) -> bool:
    if anchor is None:
        return True
    start, end = anchor
    return 0 <= start < end <= token_count


class Bookkeeper:
    def __init__(self, graph: Optional[GraphState] = None, *, min_confidence: float = 0.0):
        self.graph = graph or GraphState()
        self.min_confidence = min_confidence

    def _deny(self, code: AdmissionCode, reason: str) -> AdmissionResult:
        return AdmissionResult(code=code, reason=reason)

    def admit(
        self, proposal: EdgeProposal, *, registry: ChunkRegistry
    ) -> AdmissionResult:
        """Adjudicate one proposal. The only method that mutates graph state."""
        if proposal.confidence < self.min_confidence:
            return self._deny(
                AdmissionCode.DENIED_LOW_CONFIDENCE,
                f"confidence {proposal.confidence:.3f} < floor {self.min_confidence:.3f}",
            )

        for cid in (proposal.src_chunk_id, proposal.dst_chunk_id):
            if cid not in registry:
                return self._deny(
                    AdmissionCode.DENIED_UNKNOWN_CHUNK, f"unknown chunk {cid!r}"
                )

        if proposal.src_chunk_id == proposal.dst_chunk_id:
            return self._deny(
                AdmissionCode.DENIED_SELF_LOOP, f"self-loop on {proposal.src_chunk_id!r}"
            )

        if not _anchor_ok(proposal.src_anchor, registry[proposal.src_chunk_id]):
            return self._deny(
                AdmissionCode.DENIED_INVALID_ANCHOR,
                f"src anchor {proposal.src_anchor} out of range for "
                f"{proposal.src_chunk_id!r} ({registry[proposal.src_chunk_id]} tokens)",
            )
        if not _anchor_ok(proposal.dst_anchor, registry[proposal.dst_chunk_id]):
            return self._deny(
                AdmissionCode.DENIED_INVALID_ANCHOR,
                f"dst anchor {proposal.dst_anchor} out of range for "
                f"{proposal.dst_chunk_id!r} ({registry[proposal.dst_chunk_id]} tokens)",
            )

        if self.graph.contains(
            proposal.src_chunk_id, proposal.dst_chunk_id, proposal.relation
        ):
            return self._deny(
                AdmissionCode.DENIED_DUPLICATE,
                f"{proposal.relation.value} edge already admitted",
            )

        if self.graph.would_create_cycle(
            proposal.src_chunk_id, proposal.dst_chunk_id, proposal.relation
        ):
            return self._deny(
                AdmissionCode.DENIED_CYCLE,
                f"{proposal.relation.value} {proposal.src_chunk_id} -> "
                f"{proposal.dst_chunk_id} would create a cycle",
            )

        edge = AdmittedEdge(
            src_chunk_id=proposal.src_chunk_id,
            dst_chunk_id=proposal.dst_chunk_id,
            relation=proposal.relation,
            provenance=Provenance(
                proposer=proposal.method,
                confidence=proposal.confidence,
                admitted_at=now_iso(),
                content_hash=_content_hash(proposal),
            ),
            src_anchor=proposal.src_anchor,
            dst_anchor=proposal.dst_anchor,
        )
        self.graph.add(edge)
        return AdmissionResult(code=AdmissionCode.ADMITTED, edge=edge)

    def admit_all(
        self, proposals: Iterable[EdgeProposal], *, registry: ChunkRegistry
    ) -> list[AdmissionResult]:
        return [self.admit(p, registry=registry) for p in proposals]
