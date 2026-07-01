"""Warm tier — Postgres metadata, chunks, and edges."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

import psycopg

from .models import ChunkDraft, ChunkHit, DocumentRecord, EdgeDraft, EdgeRecord, EdgeType, EvalLayer


def _doc_uuid(doc_id: str) -> UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"gin:doc:{doc_id}")


def upsert_document(
    conn: psycopg.Connection,
    *,
    doc_id: str,
    content_hash: str,
    outlet: str,
    title: str,
    source_uri: str = "",
    source_type: str = "synthetic",
) -> UUID:
    uid = _doc_uuid(doc_id)
    conn.execute(
        """
        INSERT INTO documents (doc_id, content_hash, source_uri, source_type, outlet, title)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (content_hash) DO UPDATE SET
            outlet = EXCLUDED.outlet,
            title = EXCLUDED.title,
            source_uri = EXCLUDED.source_uri,
            source_type = EXCLUDED.source_type
        RETURNING doc_id
        """,
        (uid, content_hash, source_uri, source_type, outlet, title),
    )
    row = conn.execute(
        "SELECT doc_id FROM documents WHERE content_hash = %s",
        (content_hash,),
    ).fetchone()
    return row[0]


def upsert_chunk(conn: psycopg.Connection, chunk: ChunkDraft, doc_uuid: UUID) -> None:
    conn.execute(
        """
        INSERT INTO chunks (
            chunk_id, doc_id, chunk_index, text, head_sentence,
            eval_layer, eval_tag, content_hash
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (chunk_id) DO UPDATE SET
            text = EXCLUDED.text,
            head_sentence = EXCLUDED.head_sentence,
            eval_layer = EXCLUDED.eval_layer,
            eval_tag = EXCLUDED.eval_tag,
            content_hash = EXCLUDED.content_hash
        """,
        (
            chunk.chunk_id,
            doc_uuid,
            chunk.chunk_index,
            chunk.text,
            chunk.head_sentence,
            chunk.eval_layer.value,
            chunk.eval_tag,
            chunk.content_hash,
        ),
    )


def upsert_edge(conn: psycopg.Connection, edge: EdgeDraft) -> None:
    conn.execute(
        """
        INSERT INTO edges (edge_id, src_chunk_id, dst_chunk_id, edge_type, note)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (src_chunk_id, dst_chunk_id, edge_type) DO UPDATE SET
            note = EXCLUDED.note
        """,
        (
            uuid.uuid4(),
            edge.src_chunk_id,
            edge.dst_chunk_id,
            edge.edge_type.value,
            edge.note or None,
        ),
    )


def set_chunk_embedding(
    conn: psycopg.Connection, chunk_id: str, embedding: list[float]
) -> None:
    conn.execute(
        "UPDATE chunks SET embedding = %s WHERE chunk_id = %s",
        (embedding, chunk_id),
    )


def get_document_by_slug(conn: psycopg.Connection, doc_id: str) -> Optional[DocumentRecord]:
    uid = _doc_uuid(doc_id)
    row = conn.execute(
        """
        SELECT doc_id, content_hash, source_uri, source_type, outlet, title, ingested_at
        FROM documents WHERE doc_id = %s
        """,
        (uid,),
    ).fetchone()
    if row is None:
        return None
    return DocumentRecord(*row)


def count_chunks(conn: psycopg.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
    return int(row[0])


def list_chunks(
    conn: psycopg.Connection,
    *,
    eval_layer: Optional[EvalLayer] = None,
    limit: Optional[int] = None,
) -> list[ChunkHit]:
    clauses = ["1=1"]
    params: list[Any] = []
    if eval_layer is not None:
        clauses.append("c.eval_layer = %s")
        params.append(eval_layer.value)
    limit_clause = ""
    if limit is not None:
        limit_clause = "LIMIT %s"
        params.append(limit)
    rows = conn.execute(
        f"""
        SELECT c.chunk_id, c.doc_id, c.text, c.head_sentence, c.eval_layer, c.eval_tag,
               c.content_hash, d.outlet, d.title
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        WHERE {' AND '.join(clauses)}
        ORDER BY c.chunk_id
        {limit_clause}
        """,
        params,
    ).fetchall()
    return [
        ChunkHit(
            chunk_id=r[0],
            doc_id=r[1],
            text=r[2],
            head_sentence=r[3],
            eval_layer=r[4],
            eval_tag=r[5],
            content_hash=r[6],
            outlet=r[7],
            title=r[8],
        )
        for r in rows
    ]


def start_ingest_run(conn: psycopg.Connection) -> UUID:
    run_id = uuid.uuid4()
    conn.execute(
        """
        INSERT INTO ingest_runs (run_id, status, stats_json)
        VALUES (%s, 'running', '{}'::jsonb)
        """,
        (run_id,),
    )
    return run_id


def fetch_edges_among(
    conn: psycopg.Connection,
    chunk_ids: list[str],
    edge_types: list[str] | None = None,
) -> list[EdgeRecord]:
    if not chunk_ids:
        return []
    types = edge_types or [e.value for e in EdgeType]
    rows = conn.execute(
        """
        SELECT src_chunk_id, dst_chunk_id, edge_type, note
        FROM edges
        WHERE (src_chunk_id = ANY(%s) OR dst_chunk_id = ANY(%s))
          AND edge_type = ANY(%s)
        """,
        (chunk_ids, chunk_ids, types),
    ).fetchall()
    return [EdgeRecord(src_chunk_id=r[0], dst_chunk_id=r[1], edge_type=r[2], note=r[3]) for r in rows]


def _hits_from_rows(rows: list[tuple]) -> list[ChunkHit]:
    return [
        ChunkHit(
            chunk_id=r[0],
            doc_id=r[1],
            text=r[2],
            head_sentence=r[3],
            eval_layer=r[4],
            eval_tag=r[5],
            content_hash=r[6],
            outlet=r[7],
            title=r[8],
        )
        for r in rows
    ]


def fetch_chunks_by_ids(conn: psycopg.Connection, chunk_ids: list[str]) -> list[ChunkHit]:
    if not chunk_ids:
        return []
    rows = conn.execute(
        """
        SELECT c.chunk_id, c.doc_id, c.text, c.head_sentence, c.eval_layer, c.eval_tag,
               c.content_hash, d.outlet, d.title
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        WHERE c.chunk_id = ANY(%s)
        """,
        (chunk_ids,),
    ).fetchall()
    return _hits_from_rows(rows)


def fetch_neighbor_chunks(
    conn: psycopg.Connection,
    chunk_ids: list[str],
    edge_types: list[str],
) -> list[ChunkHit]:
    """Fetch chunk hits for edge endpoints not already in chunk_ids."""
    edges = fetch_edges_among(conn, chunk_ids, edge_types=edge_types)
    neighbor_ids: set[str] = set()
    seed = set(chunk_ids)
    for edge in edges:
        if edge.src_chunk_id not in seed:
            neighbor_ids.add(edge.src_chunk_id)
        if edge.dst_chunk_id not in seed:
            neighbor_ids.add(edge.dst_chunk_id)
    return fetch_chunks_by_ids(conn, sorted(neighbor_ids))


def finish_ingest_run(
    conn: psycopg.Connection, run_id: UUID, status: str, stats: dict[str, Any]
) -> None:
    conn.execute(
        """
        UPDATE ingest_runs
        SET finished_at = %s, status = %s, stats_json = %s::jsonb
        WHERE run_id = %s
        """,
        (datetime.now(timezone.utc), status, json.dumps(stats), run_id),
    )
