"""Corpus fingerprint for reproducible eval comparisons."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import psycopg

from .db import connect


def corpus_fingerprint(conn: psycopg.Connection | None = None) -> dict[str, Any]:
    """Hash of sorted (chunk_id, content_hash) pairs plus counts."""
    if conn is None:
        with connect() as conn:
            return corpus_fingerprint(conn)

    rows = conn.execute(
        "SELECT chunk_id, content_hash FROM chunks ORDER BY chunk_id"
    ).fetchall()
    edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    payload = "|".join(f"{cid}:{h}" for cid, h in rows)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return {
        "chunk_count": len(rows),
        "edge_count": int(edge_count),
        "content_hash": digest,
    }


def fingerprint_json(conn: psycopg.Connection | None = None) -> str:
    return json.dumps(corpus_fingerprint(conn), sort_keys=True)
