"""Bookkeeper layer — sole admission gate and sole writer of canonical graph state.

Verifies anchor integrity, enforces DAG invariants, stamps provenance, and admits
or denies Cartographer proposals through one uniform gate (local == federated).
It makes nothing; it only adjudicates. See docs/nc_cartographer_design.plan.md.
"""
from .bookkeeper import Bookkeeper, ChunkRegistry
from .graph import GraphState, edge_key
from .models import (
    AdmissionCode,
    AdmissionResult,
    AdmittedEdge,
    Provenance,
    ORDERING_RELATIONS,
    SYMMETRIC_RELATIONS,
)

__all__ = [
    "AdmissionCode",
    "AdmissionResult",
    "AdmittedEdge",
    "Bookkeeper",
    "ChunkRegistry",
    "GraphState",
    "ORDERING_RELATIONS",
    "Provenance",
    "SYMMETRIC_RELATIONS",
    "edge_key",
]
