"""Where each node's cached copy of a peer's anchor set lives, and how a
node reads its OWN chunks as anchors to serve to a peer.

InMemoryPeerAnchorStore backs unit/integration tests; PostgresPeerAnchorStore
is the production cache, backed by the peer_anchors table. local_anchor_rows
reads THIS node's own chunks/documents — the set a peer's sync loop pulls
from. Never chunk text, in either direction.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import psycopg

from gin.corpus.db import connect, transaction

from .schema import AnchorLeaf


@runtime_checkable
class PeerAnchorStore(Protocol):
    def all_rows(self, peer_node_id: str) -> list[AnchorLeaf]: ...
    def bucket_rows(self, peer_node_id: str, bucket_index: int) -> list[AnchorLeaf]: ...
    def replace_bucket(
        self, peer_node_id: str, bucket_index: int, rows: list[AnchorLeaf]
    ) -> None: ...


class InMemoryPeerAnchorStore:
    def __init__(self) -> None:
        self._data: dict[str, dict[int, list[AnchorLeaf]]] = {}

    def all_rows(self, peer_node_id: str) -> list[AnchorLeaf]:
        buckets = self._data.get(peer_node_id, {})
        return [row for rows in buckets.values() for row in rows]

    def bucket_rows(self, peer_node_id: str, bucket_index: int) -> list[AnchorLeaf]:
        return list(self._data.get(peer_node_id, {}).get(bucket_index, []))

    def replace_bucket(
        self, peer_node_id: str, bucket_index: int, rows: list[AnchorLeaf]
    ) -> None:
        self._data.setdefault(peer_node_id, {})[bucket_index] = list(rows)


class PostgresPeerAnchorStore:
    """Opens a fresh connection per call — matches the corpus tier's
    connect()-per-call convention (gin/corpus/fingerprint.py)."""

    def all_rows(self, peer_node_id: str) -> list[AnchorLeaf]:
        with connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id, content_hash, outlet, title FROM peer_anchors "
                "WHERE peer_node_id = %s",
                (peer_node_id,),
            ).fetchall()
        return [AnchorLeaf(chunk_id=r[0], content_hash=r[1], outlet=r[2], title=r[3]) for r in rows]

    def bucket_rows(self, peer_node_id: str, bucket_index: int) -> list[AnchorLeaf]:
        with connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id, content_hash, outlet, title FROM peer_anchors "
                "WHERE peer_node_id = %s AND bucket_index = %s",
                (peer_node_id, bucket_index),
            ).fetchall()
        return [AnchorLeaf(chunk_id=r[0], content_hash=r[1], outlet=r[2], title=r[3]) for r in rows]

    def replace_bucket(
        self, peer_node_id: str, bucket_index: int, rows: list[AnchorLeaf]
    ) -> None:
        with transaction() as conn:
            conn.execute(
                "DELETE FROM peer_anchors WHERE peer_node_id = %s AND bucket_index = %s",
                (peer_node_id, bucket_index),
            )
            for row in rows:
                conn.execute(
                    "INSERT INTO peer_anchors "
                    "(peer_node_id, chunk_id, content_hash, outlet, title, bucket_index) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (peer_node_id, row.chunk_id, row.content_hash, row.outlet,
                     row.title, bucket_index),
                )


def local_anchor_rows(conn: Optional[psycopg.Connection] = None) -> list[AnchorLeaf]:
    """This node's own chunks as anchors — what a peer's sync loop may read."""
    if conn is None:
        with connect() as conn:
            return local_anchor_rows(conn)
    rows = conn.execute(
        "SELECT c.chunk_id, c.content_hash, d.outlet, d.title "
        "FROM chunks c JOIN documents d ON d.doc_id = c.doc_id"
    ).fetchall()
    return [AnchorLeaf(chunk_id=r[0], content_hash=r[1], outlet=r[2], title=r[3]) for r in rows]
