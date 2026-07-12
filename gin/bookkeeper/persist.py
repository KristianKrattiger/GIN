"""Persist Bookkeeper graph state to Postgres warm-tier edges."""
from __future__ import annotations

import uuid
from typing import Iterable, Optional

import psycopg

from gin.cartographer.models import Relation

from .bookkeeper import Bookkeeper
from .graph import GraphState
from .models import AdmissionCode, AdmissionResult, AdmittedEdge, Provenance

_RELATION_TO_EDGE_TYPE = {
    Relation.CONTRADICTS: "contradicts",
    Relation.SUPERSEDES: "supersedes",
    Relation.CORROBORATES: "corroborates",
}

_EDGE_TYPE_TO_RELATION = {v: k for k, v in _RELATION_TO_EDGE_TYPE.items()}


def ensure_edge_schema(conn: psycopg.Connection) -> None:
    """Add provenance columns to existing edges tables (idempotent)."""
    for col, typedef in (
        ("proposer", "TEXT"),
        ("confidence", "REAL"),
        ("content_hash", "TEXT"),
        ("src_anchor", "INT[]"),
        ("dst_anchor", "INT[]"),
    ):
        conn.execute(
            f"ALTER TABLE edges ADD COLUMN IF NOT EXISTS {col} {typedef}"
        )


def _anchor_to_pg(anchor: Optional[tuple[int, int]]) -> Optional[list[int]]:
    if anchor is None:
        return None
    return [anchor[0], anchor[1]]


def _anchor_from_pg(raw: Optional[list[int]]) -> Optional[tuple[int, int]]:
    if not raw or len(raw) != 2:
        return None
    return int(raw[0]), int(raw[1])


def _row_to_admitted(row: tuple) -> AdmittedEdge:
    """Hydrate an admitted edge from a warm-tier edges row."""
    (
        src_chunk_id,
        dst_chunk_id,
        edge_type,
        proposer,
        confidence,
        content_hash,
        src_anchor,
        dst_anchor,
        admitted_at,
    ) = row
    relation = _EDGE_TYPE_TO_RELATION.get(edge_type, Relation.CONTRADICTS)
    prov = Provenance(
        proposer=proposer or "unknown",
        confidence=float(confidence or 0.0),
        admitted_at=admitted_at.isoformat() if hasattr(admitted_at, "isoformat") else str(admitted_at),
        content_hash=content_hash or "",
    )
    return AdmittedEdge(
        src_chunk_id=src_chunk_id,
        dst_chunk_id=dst_chunk_id,
        relation=relation,
        provenance=prov,
        src_anchor=_anchor_from_pg(src_anchor),
        dst_anchor=_anchor_from_pg(dst_anchor),
    )


def load_graph(conn: psycopg.Connection, *, edge_types: Optional[list[str]] = None) -> GraphState:
    """Hydrate canonical graph state from Postgres edges with provenance."""
    ensure_edge_schema(conn)
    types = edge_types or ["contradicts", "supersedes"]
    rows = conn.execute(
        """
        SELECT src_chunk_id, dst_chunk_id, edge_type,
               proposer, confidence, content_hash,
               src_anchor, dst_anchor, admitted_at
        FROM edges
        WHERE edge_type = ANY(%s)
        ORDER BY admitted_at, src_chunk_id, dst_chunk_id
        """,
        (types,),
    ).fetchall()
    graph = GraphState()
    for row in rows:
        graph.add(_row_to_admitted(row))
    return graph


def upsert_admitted_edge(conn: psycopg.Connection, edge: AdmittedEdge, *, note: str = "") -> None:
    """Write one Bookkeeper-admitted edge to the warm tier."""
    ensure_edge_schema(conn)
    edge_type = _RELATION_TO_EDGE_TYPE.get(edge.relation, edge.relation.value)
    conn.execute(
        """
        INSERT INTO edges (
            edge_id, src_chunk_id, dst_chunk_id, edge_type, note,
            proposer, confidence, content_hash, src_anchor, dst_anchor
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (src_chunk_id, dst_chunk_id, edge_type) DO UPDATE SET
            note = EXCLUDED.note,
            proposer = EXCLUDED.proposer,
            confidence = EXCLUDED.confidence,
            content_hash = EXCLUDED.content_hash,
            src_anchor = EXCLUDED.src_anchor,
            dst_anchor = EXCLUDED.dst_anchor,
            admitted_at = NOW()
        """,
        (
            uuid.uuid4(),
            edge.src_chunk_id,
            edge.dst_chunk_id,
            edge_type,
            note or None,
            edge.provenance.proposer,
            edge.provenance.confidence,
            edge.provenance.content_hash,
            _anchor_to_pg(edge.src_anchor),
            _anchor_to_pg(edge.dst_anchor),
        ),
    )


def sync_admissions(
    conn: psycopg.Connection,
    results: Iterable[AdmissionResult],
    *,
    notes: Optional[dict[tuple[str, str, str], str]] = None,
) -> dict[AdmissionCode, int]:
    """Persist newly admitted edges; count outcomes by admission code."""
    ensure_edge_schema(conn)
    counts: dict[AdmissionCode, int] = {}
    notes = notes or {}
    for result in results:
        counts[result.code] = counts.get(result.code, 0) + 1
        if result.code != AdmissionCode.ADMITTED or result.edge is None:
            continue
        edge = result.edge
        edge_type = _RELATION_TO_EDGE_TYPE.get(edge.relation, edge.relation.value)
        note = notes.get((edge.src_chunk_id, edge.dst_chunk_id, edge_type), "")
        upsert_admitted_edge(conn, edge, note=note)
    return counts


def load_bookkeeper(
    conn: psycopg.Connection,
    *,
    min_confidence: float = 0.0,
    edge_types: Optional[list[str]] = None,
) -> Bookkeeper:
    """Construct a Bookkeeper whose graph is hydrated from Postgres."""
    return Bookkeeper(load_graph(conn, edge_types=edge_types), min_confidence=min_confidence)
