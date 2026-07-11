"""Hybrid retrieval — dense pgvector + sparse tsvector with RRF merge."""
from __future__ import annotations

from typing import Any, Optional

import psycopg
from pgvector.psycopg import register_vector

from .db import connect
from .hot import embed_query, register
from .models import ChunkHit, EdgeRecord, SynthesisBundle, SynthesisMode
from . import warm

RRF_K = 60
AMBIGUITY_SCORE_DELTA = 0.15
PAIR_SCORE_BOOST = 0.05
DEFAULT_MIN_RRF_DELTA = 0.25
RETRIEVAL_CONFIDENCE_FLOOR = 0.010
DIVERGENCE_RELEVANCE_FLOOR = 0.15
MIN_DIVERGENCE_KEYWORD_MATCHES = 2
# IDF-weighted divergence gate: min fraction of the query's IDF mass a chunk
# must match to count as one side of a real contradiction. Separates a single
# distinctive shared word (e.g. "wildfire") from a generic one (e.g. "district").
DIVERGENCE_IDF_FLOOR = 0.13

# Corpus IDF cache, keyed by chunk count (a wipe/re-ingest changes the count).
# Crude but fine for the eval workflow; only the latest corpus is retained.
_IDF_CACHE: dict[int, dict[str, float]] = {}


def _corpus_idf(conn) -> dict[str, float]:
    n = warm.count_chunks(conn)
    cached = _IDF_CACHE.get(n)
    if cached is not None:
        return cached
    from .relevance import corpus_idf

    idf = corpus_idf([c.text for c in warm.list_chunks(conn)])
    _IDF_CACHE.clear()
    _IDF_CACHE[n] = idf
    return idf


class RetrievalConfidenceError(Exception):
    """Raised when no chunk meets the absolute confidence floor."""

    def __init__(self, query: str, top_score: float, floor: float):
        self.query = query
        self.top_score = top_score
        self.floor = floor
        super().__init__(
            f"Retrieval confidence too low for query {query!r}: "
            f"top RRF score {top_score:.4f} < floor {floor:.4f}"
        )


def _apply_chunk_filters(clauses: list[str], params: list[Any], filters: dict[str, Any]) -> None:
    if "eval_layer" in filters:
        clauses.append("c.eval_layer = %s")
        params.append(filters["eval_layer"])
    if "eval_tag" in filters:
        clauses.append("c.eval_tag = %s")
        params.append(filters["eval_tag"])


def _row_to_hit(row: tuple, *, dense_rank: Optional[int], sparse_rank: Optional[int], score: float) -> ChunkHit:
    return ChunkHit(
        chunk_id=row[0],
        doc_id=row[1],
        text=row[2],
        head_sentence=row[3],
        eval_layer=row[4],
        eval_tag=row[5],
        content_hash=row[6],
        outlet=row[7],
        title=row[8],
        dense_rank=dense_rank,
        sparse_rank=sparse_rank,
        rrf_score=score,
    )


def _dense_search(conn: psycopg.Connection, query_vec: list[float], k: int, filters: dict[str, Any]) -> list[tuple]:
    clauses = ["c.embedding IS NOT NULL"]
    filter_params: list[Any] = []
    _apply_chunk_filters(clauses, filter_params, filters)
    where = " AND ".join(clauses)
    params: list[Any] = [query_vec, *filter_params, query_vec, k]
    return conn.execute(
        f"""
        SELECT c.chunk_id, c.doc_id, c.text, c.head_sentence, c.eval_layer, c.eval_tag,
               c.content_hash, d.outlet, d.title,
               c.embedding <=> %s::vector AS distance
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        WHERE {where}
        ORDER BY c.embedding <=> %s::vector, c.chunk_id
        LIMIT %s
        """,
        params,
    ).fetchall()


def _sparse_search(conn: psycopg.Connection, query: str, k: int, filters: dict[str, Any]) -> list[tuple]:
    clauses = ["c.tsv @@ plainto_tsquery('english', %s)"]
    params: list[Any] = [query]
    _apply_chunk_filters(clauses, params, filters)
    where = " AND ".join(clauses)
    params.extend([query, k])
    return conn.execute(
        f"""
        SELECT c.chunk_id, c.doc_id, c.text, c.head_sentence, c.eval_layer, c.eval_tag,
               c.content_hash, d.outlet, d.title,
               ts_rank(c.tsv, plainto_tsquery('english', %s)) AS rank
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        WHERE {where}
        ORDER BY rank DESC, c.chunk_id
        LIMIT %s
        """,
        params,
    ).fetchall()


def retrieve(query: str, k: int = 15, filters: Optional[dict[str, Any]] = None) -> list[ChunkHit]:
    filters = filters or {}
    query_vec = embed_query(query)
    with connect() as conn:
        register(conn)
        dense_rows = _dense_search(conn, query_vec, k, filters)
        sparse_rows = _sparse_search(conn, query, k, filters)

    scores: dict[str, float] = {}
    meta: dict[str, tuple] = {}
    dense_rank: dict[str, int] = {}
    sparse_rank: dict[str, int] = {}

    for rank, row in enumerate(dense_rows, start=1):
        cid = row[0]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        meta[cid] = row[:9]
        dense_rank[cid] = rank

    for rank, row in enumerate(sparse_rows, start=1):
        cid = row[0]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        meta[cid] = row[:9]
        sparse_rank[cid] = rank

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
    return [
        _row_to_hit(
            meta[cid],
            dense_rank=dense_rank.get(cid),
            sparse_rank=sparse_rank.get(cid),
            score=score,
        )
        for cid, score in ordered
    ]


def _apply_relevance_floor(hits: list[ChunkHit], min_rrf_delta: float) -> list[ChunkHit]:
    if not hits or min_rrf_delta <= 0:
        return hits
    top = max(h.rrf_score for h in hits)
    floor = top - min_rrf_delta
    return [h for h in hits if h.rrf_score >= floor]


def _divergence_relevant(
    text: str, query: str, idf: Optional[dict[str, float]] = None
) -> bool:
    """Query-relevance test for one side of a contradicts pair.

    With corpus ``idf`` (production path), a chunk is relevant when it matches
    enough of the query's IDF-weighted keyword mass — one DISTINCTIVE shared
    word suffices, a generic one does not. Without idf (unit tests), fall back
    to the lexical gate: a single shared keyword (e.g. "district" in both an
    election chunk and a school query) is not evidence the contradiction matters,
    so for queries with enough keywords a sentence must match at least
    MIN_DIVERGENCE_KEYWORD_MATCHES of them.
    """
    if idf is not None:
        from .relevance import idf_weighted_relevance

        return idf_weighted_relevance(text, query, idf) >= DIVERGENCE_IDF_FLOOR

    from .relevance import matched_keyword_count, max_sentence_score, query_keywords

    if max_sentence_score(text, query) < DIVERGENCE_RELEVANCE_FLOOR:
        return False
    required = (
        MIN_DIVERGENCE_KEYWORD_MATCHES
        if len(query_keywords(query)) >= 3
        else 1
    )
    return matched_keyword_count(text, query) >= required


def _pair_divergence_ok(
    left_text: str, right_text: str, query: str, idf: Optional[dict[str, float]] = None
) -> bool:
    """Both sides of a curated contradicts pair must be query-relevant.

    With IDF-weighted relevance (``idf``), a genuine reframing whose only shared
    word is DISTINCTIVE ("wildfire") passes while an unrelated edge partner
    sharing only a GENERIC word ("district" between an enrollment chunk and a
    turnout query) does not — the separation plain keyword-overlap can't make.
    The both-sides requirement is retained as defense against mis-curated edges.
    See docs/nc_phase3_divergence_correctness.plan.md.
    """
    return _divergence_relevant(left_text, query, idf) and _divergence_relevant(
        right_text, query, idf
    )


def _is_ambiguous(
    seed_hits: list[ChunkHit],
    edges: list[EdgeRecord],
    query: str = "",
    idf: Optional[dict[str, float]] = None,
) -> bool:
    contradicts_edges = [e for e in edges if e.edge_type == "contradicts"]
    if not query:
        # Legacy path: no query context, fall back to blunt heuristics.
        if contradicts_edges:
            return True
        if len(seed_hits) < 2:
            return False
        top_score = seed_hits[0].rrf_score
        close_competitors = [
            h for h in seed_hits[1:]
            if top_score - h.rrf_score <= AMBIGUITY_SCORE_DELTA
        ]
        outlets = {h.outlet for h in seed_hits}
        doc_ids = {h.doc_id for h in seed_hits}
        return len(close_competitors) >= 1 and (len(outlets) > 1 or len(doc_ids) > 1)

    # Divergent mode requires an actual, query-relevant contradiction. Close
    # RRF competitors from different outlets are corroboration (the convergent
    # success case), not divergence — see docs/nc_phase3_divergence_correctness.plan.md.
    hit_by_id = {h.chunk_id: h for h in seed_hits}
    for edge in contradicts_edges:
        left = hit_by_id.get(edge.src_chunk_id)
        right = hit_by_id.get(edge.dst_chunk_id)
        if left is None or right is None:
            continue
        if _pair_divergence_ok(left.text, right.text, query, idf):
            return True
    return False


def _build_pairs(
    hits_by_id: dict[str, ChunkHit],
    edges: list[EdgeRecord],
    query: str = "",
    idf: Optional[dict[str, float]] = None,
) -> list[tuple[ChunkHit, ChunkHit, EdgeRecord]]:
    pairs: list[tuple[ChunkHit, ChunkHit, EdgeRecord]] = []
    for edge in edges:
        if edge.edge_type not in ("contradicts", "cites"):
            continue
        left = hits_by_id.get(edge.src_chunk_id)
        right = hits_by_id.get(edge.dst_chunk_id)
        if left is None or right is None:
            continue
        if edge.edge_type == "contradicts" and query:
            # Same divergence-relevance test as _is_ambiguous, so a pair can
            # never be front-loaded by _prioritize_hits without also flipping
            # the mode.
            if not _pair_divergence_ok(left.text, right.text, query, idf):
                continue
        pairs.append((left, right, edge))
    return pairs


def _neighbor_ids_from_seed_edges(
    seed_ids: set[str],
    edges: list[EdgeRecord],
) -> set[str]:
    """One-hop neighbors: other endpoint of edges touching a seed chunk."""
    neighbor_ids: set[str] = set()
    for edge in edges:
        if edge.src_chunk_id in seed_ids and edge.dst_chunk_id not in seed_ids:
            neighbor_ids.add(edge.dst_chunk_id)
        if edge.dst_chunk_id in seed_ids and edge.src_chunk_id not in seed_ids:
            neighbor_ids.add(edge.src_chunk_id)
    return neighbor_ids


def _prioritize_hits(
    seed_hits: list[ChunkHit],
    neighbors: list[ChunkHit],
    pairs: list[tuple[ChunkHit, ChunkHit, EdgeRecord]],
    *,
    k_max: int,
    min_rrf_delta: float,
) -> list[ChunkHit]:
    seed_ids = {h.chunk_id for h in seed_hits}
    pair_ids: set[str] = set()
    for left, right, edge in pairs:
        if edge.edge_type == "contradicts":
            pair_ids.add(left.chunk_id)
            pair_ids.add(right.chunk_id)

    neighbor_by_id = {h.chunk_id: h for h in neighbors}
    ordered: list[ChunkHit] = []
    seen: set[str] = set()

    def _add(hit: ChunkHit) -> None:
        if hit.chunk_id not in seen:
            ordered.append(hit)
            seen.add(hit.chunk_id)

    for left, right, edge in pairs:
        if edge.edge_type != "contradicts":
            continue
        for hit in (left, right):
            if hit.chunk_id in seed_ids or hit.chunk_id in neighbor_by_id:
                _add(hit if hit.chunk_id in seed_ids else neighbor_by_id[hit.chunk_id])

    for hit in seed_hits:
        _add(hit)

    for hit in neighbors:
        _add(hit)

    # Preserve insertion order (query-ranked seeds first, then neighbors)
    # rather than re-sorting by RRF which undoes query-relevance ordering.
    merged = _apply_relevance_floor(ordered, min_rrf_delta)
    return merged[:k_max]


def _boost_paired_scores(
    hits: list[ChunkHit],
    pairs: list[tuple[ChunkHit, ChunkHit, EdgeRecord]],
) -> list[ChunkHit]:
    if not pairs:
        return hits
    boost_ids: set[str] = set()
    for left, right, _edge in pairs:
        boost_ids.add(left.chunk_id)
        boost_ids.add(right.chunk_id)
    boosted: list[ChunkHit] = []
    for hit in hits:
        if hit.chunk_id in boost_ids:
            boosted.append(
                ChunkHit(
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    text=hit.text,
                    head_sentence=hit.head_sentence,
                    eval_layer=hit.eval_layer,
                    eval_tag=hit.eval_tag,
                    content_hash=hit.content_hash,
                    outlet=hit.outlet,
                    title=hit.title,
                    dense_rank=hit.dense_rank,
                    sparse_rank=hit.sparse_rank,
                    rrf_score=hit.rrf_score + PAIR_SCORE_BOOST,
                )
            )
        else:
            boosted.append(hit)
    return boosted


def retrieve_for_synthesis(
    query: str,
    *,
    k_seed: int = 5,
    k_max: int = 6,
    filters: Optional[dict[str, Any]] = None,
    min_rrf_delta: float = DEFAULT_MIN_RRF_DELTA,
    confidence_floor: float = RETRIEVAL_CONFIDENCE_FLOOR,
) -> SynthesisBundle:
    """Retrieve seed hits, expand via edges, detect divergent mode."""
    filters = filters or {}
    seed_hits = retrieve(query, k=k_seed, filters=filters)
    if not seed_hits:
        raise RetrievalConfidenceError(query, 0.0, confidence_floor)
    if confidence_floor > 0 and seed_hits[0].rrf_score < confidence_floor:
        raise RetrievalConfidenceError(query, seed_hits[0].rrf_score, confidence_floor)
    seed_hits = _apply_relevance_floor(seed_hits, min_rrf_delta)

    # Phase B: re-rank seeds by query relevance before mode detection
    from .relevance import max_sentence_score, rerank_hits_by_query_score
    seed_hits = rerank_hits_by_query_score(seed_hits, query)

    # Phase C: drop zero-relevance seeds (keep all if none are relevant)
    relevant_seeds = [h for h in seed_hits if max_sentence_score(h.text, query) > 0]
    if relevant_seeds:
        seed_hits = relevant_seeds

    seed_ids = [h.chunk_id for h in seed_hits]
    seed_id_set = set(seed_ids)

    with connect() as conn:
        edges = warm.fetch_edges_among(
            conn, seed_ids, edge_types=["contradicts", "cites"]
        )
        neighbor_id_set = _neighbor_ids_from_seed_edges(seed_id_set, edges)
        neighbors = warm.fetch_chunks_by_ids(conn, sorted(neighbor_id_set))
        idf = _corpus_idf(conn)

    mode: SynthesisMode = (
        "divergent" if _is_ambiguous(seed_hits, edges, query, idf) else "convergent"
    )

    hits_by_id: dict[str, ChunkHit] = {h.chunk_id: h for h in seed_hits}
    for hit in neighbors:
        hits_by_id.setdefault(hit.chunk_id, hit)

    all_edges = [
        e for e in edges
        if e.src_chunk_id in hits_by_id and e.dst_chunk_id in hits_by_id
    ]
    pairs = _build_pairs(hits_by_id, all_edges, query, idf)

    merged = _prioritize_hits(
        seed_hits, neighbors, pairs, k_max=k_max, min_rrf_delta=min_rrf_delta
    )
    hits_by_id = {h.chunk_id: h for h in merged}
    pairs = _build_pairs(hits_by_id, all_edges, query, idf)
    hits = _boost_paired_scores(merged, pairs)

    # Mode follows the contradicts pairs that actually survived edge expansion
    # and the divergence-relevance gate. A contradiction reachable only via a
    # neighbor (one endpoint not in the seed set) is real divergence too, and
    # the seed-only _is_ambiguous would miss it — so for the query path derive
    # mode from the surviving pairs.
    if query:
        mode = (
            "divergent"
            if any(getattr(e, "edge_type", None) == "contradicts" for _, _, e in pairs)
            else "convergent"
        )

    return SynthesisBundle(hits=hits, edges=all_edges, mode=mode, pairs=pairs)
