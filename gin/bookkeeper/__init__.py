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
from .persist import (
    ensure_edge_schema,
    load_bookkeeper,
    load_graph,
    sync_admissions,
    upsert_admitted_edge,
)

from .relation_verify import FRAMING_BAND_FLOOR, RelationVerifyResult, verify_contradicts

__all__ = [
    "AdmissionCode",
    "AdmissionResult",
    "AdmittedEdge",
    "Bookkeeper",
    "ChunkRegistry",
    "GraphState",
    "ORDERING_RELATIONS",
    "Provenance",
    "RelationVerifyResult",
    "SYMMETRIC_RELATIONS",
    "edge_key",
    "ensure_edge_schema",
    "load_bookkeeper",
    "load_graph",
    "sync_admissions",
    "upsert_admitted_edge",
    "verify_contradicts",
    "FRAMING_BAND_FLOOR",
]
