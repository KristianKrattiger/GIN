"""GIN evaluation harness: RAG vs SEAR designed experiment.

Runs a shared query set through pluggable generation arms (traditional RAG,
GIN No-Continuation, and — later — Flagged Generation), scores every emitted
claim for grounding, and aggregates per-arm / per-eval-layer metrics.
"""
from __future__ import annotations

from .claims import ClaimRecord, NodeScope, RawClaim, SpanType, Verdict
from .queryset import EvalQuery, load_query_set
from .verifier import ClaimVerdict, Verifier

__all__ = [
    "ClaimRecord",
    "ClaimVerdict",
    "EvalQuery",
    "NodeScope",
    "RawClaim",
    "SpanType",
    "Verdict",
    "Verifier",
    "load_query_set",
]
