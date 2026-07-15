"""Pure peer-ranking logic — no I/O, no network, no DB.

Mirrors the retrieval tier's hybrid fusion (gin/corpus/retrieve.py) one level
up: rank peers by dense similarity (query embedding vs. each peer's centroid)
and by sparse overlap (query keywords vs. each peer's distinctive IDF terms),
then RRF-fuse the two rankings with the same RRF_K. Peers without a cached
summary are appended last in config order, never dropped.
"""
from __future__ import annotations

import math

from gin.corpus.retrieve import RRF_K

from .schema import PeerSummaryResponse


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def dense_rank(
    query_embedding: list[float], centroids: dict[str, list[float]]
) -> list[str]:
    """Node ids by descending cosine similarity to the query (tiebreak: id)."""
    scored = [
        (cosine(query_embedding, centroid), node_id)
        for node_id, centroid in centroids.items()
    ]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [node_id for _, node_id in scored]


def sparse_rank(
    query_keywords: set[str], term_maps: dict[str, dict[str, float]]
) -> list[str]:
    """Node ids by descending matched-IDF mass (tiebreak: id)."""
    scored = []
    for node_id, terms in term_maps.items():
        mass = sum(terms.get(kw, 0.0) for kw in query_keywords)
        scored.append((mass, node_id))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [node_id for _, node_id in scored]


def rank_peers(
    query_embedding: list[float],
    query_keywords: set[str],
    summaries: dict[str, PeerSummaryResponse],
    peer_order: list[str],
) -> list[str]:
    """Full ranked peer order: RRF-fused for peers with a summary, then any
    remaining peers appended in config order."""
    centroids = {nid: s.embedding_centroid for nid, s in summaries.items()}
    term_maps = {nid: s.distinctive_terms for nid, s in summaries.items()}
    d_rank = {nid: i for i, nid in enumerate(dense_rank(query_embedding, centroids), start=1)}
    s_rank = {nid: i for i, nid in enumerate(sparse_rank(query_keywords, term_maps), start=1)}

    fused = []
    for nid in summaries:
        score = 1.0 / (RRF_K + d_rank[nid]) + 1.0 / (RRF_K + s_rank[nid])
        fused.append((score, nid))
    fused.sort(key=lambda t: (-t[0], t[1]))
    ranked = [nid for _, nid in fused]

    # Peers without a summary: appended in config order, never dropped.
    ranked += [nid for nid in peer_order if nid not in summaries]
    return ranked
