"""Where each node caches its peers' routing summaries, and how a node builds
its OWN summary to serve.

build_local_summary reads this node's chunk texts, computes a unit-normalized
mean embedding (the centroid) and the top-N distinctive IDF terms. Chunk text
is used only to derive these aggregates — it never leaves the node.
"""
from __future__ import annotations

import json
from typing import Optional, Protocol, runtime_checkable

import psycopg

from gin.corpus.db import connect, transaction
from gin.corpus.hot import EMBEDDING_DIM, embed_texts
from gin.corpus.relevance import corpus_idf

from .schema import PeerSummaryResponse


@runtime_checkable
class PeerSummaryStore(Protocol):
    def get(self, peer_node_id: str) -> Optional[PeerSummaryResponse]: ...
    def set(self, peer_node_id: str, summary: PeerSummaryResponse) -> None: ...


class InMemoryPeerSummaryStore:
    def __init__(self) -> None:
        self._data: dict[str, PeerSummaryResponse] = {}

    def get(self, peer_node_id: str) -> Optional[PeerSummaryResponse]:
        return self._data.get(peer_node_id)

    def set(self, peer_node_id: str, summary: PeerSummaryResponse) -> None:
        self._data[peer_node_id] = summary


class PostgresPeerSummaryStore:
    """Fresh connection per call, matching the corpus tier's convention."""

    def get(self, peer_node_id: str) -> Optional[PeerSummaryResponse]:
        with connect() as conn:
            row = conn.execute(
                "SELECT embedding_centroid, distinctive_terms, domains FROM peer_summaries "
                "WHERE peer_node_id = %s",
                (peer_node_id,),
            ).fetchone()
        if row is None:
            return None
        terms = row[1] if isinstance(row[1], dict) else json.loads(row[1])
        domains = row[2] if isinstance(row[2], list) else json.loads(row[2])
        return PeerSummaryResponse(
            node_id=peer_node_id,
            embedding_centroid=[float(x) for x in row[0]],
            distinctive_terms=terms,
            domains=domains,
        )

    def set(self, peer_node_id: str, summary: PeerSummaryResponse) -> None:
        with transaction() as conn:
            conn.execute(
                "INSERT INTO peer_summaries "
                "(peer_node_id, embedding_centroid, distinctive_terms, domains) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (peer_node_id) DO UPDATE SET "
                "embedding_centroid = EXCLUDED.embedding_centroid, "
                "distinctive_terms = EXCLUDED.distinctive_terms, "
                "domains = EXCLUDED.domains, "
                "synced_at = NOW()",
                (
                    peer_node_id,
                    list(summary.embedding_centroid),
                    json.dumps(summary.distinctive_terms),
                    json.dumps(summary.domains),
                ),
            )


def _unit_mean(vectors: list[list[float]]) -> list[float]:
    # Callers guard against an empty corpus before this point (they never
    # embed an empty list), so vectors is always non-empty here.
    dim = len(vectors[0])
    mean = [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]
    norm = sum(x * x for x in mean) ** 0.5
    if norm == 0.0:
        return mean
    return [x / norm for x in mean]


def build_local_summary(
    node_id: str, top_n: int = 40, conn: Optional[psycopg.Connection] = None
) -> PeerSummaryResponse:
    """This node's routing summary: unit-mean chunk embedding + top-N IDF
    terms + the distinct non-empty domains this node's corpus covers."""
    if conn is None:
        with connect() as conn:
            return build_local_summary(node_id, top_n, conn)
    texts = [r[0] for r in conn.execute("SELECT text FROM chunks").fetchall()]
    centroid = _unit_mean(embed_texts(texts)) if texts else [0.0] * EMBEDDING_DIM
    idf = corpus_idf(texts)
    top = dict(sorted(idf.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n])
    domains = sorted({
        r[0] for r in conn.execute(
            "SELECT DISTINCT domain FROM documents WHERE domain != ''"
        ).fetchall()
    })
    return PeerSummaryResponse(
        node_id=node_id, embedding_centroid=centroid, distinctive_terms=top,
        domains=domains,
    )
