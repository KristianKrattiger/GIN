"""Materialize warm-tier chunks into a SEAR Corpus."""
from __future__ import annotations

from typing import Callable, Optional

from sear.connectives import build_connective_inventory, phrases_for_edge_types
from sear.corpus import Corpus

from .corpus_manager import CorpusManager
from .db import connect
from .divergence import (
    compute_divergence_sentence_ranges,
    compute_divergence_zones,
    shared_sentence_starts,
)
from .models import ChunkHit, EvalLayer, SynthesisBundle, SynthesisContext
from .relevance import score_starts_by_sentence_match
from .retrieval_manifest import (
    RetrievalManifest,
    build_retrieval_manifest,
    write_retrieval_manifest,
)
from . import warm
from .retrieve import retrieve, retrieve_for_synthesis, RETRIEVAL_CONFIDENCE_FLOOR


def materialize_corpus(
    chunks: list[ChunkHit],
    tokenize: Callable[[bytes], list[int]],
) -> Corpus:
    ordered = [(hit.chunk_id, hit.text) for hit in chunks]
    doc_meta = [
        {
            "chunk_id": hit.chunk_id,
            "outlet": hit.outlet,
            "title": hit.title,
            "doc_id": str(hit.doc_id),
        }
        for hit in chunks
    ]
    return Corpus.from_chunks(ordered, tokenize=tokenize, doc_meta=doc_meta)


def materialize_all(
    tokenize: Callable[[bytes], list[int]],
    *,
    eval_layer: Optional[EvalLayer] = None,
    manifest_version: int | None = None,
) -> Corpus:
    if manifest_version is not None:
        return materialize_from_manifest(tokenize, manifest_version=manifest_version)
    with connect() as conn:
        hits = warm.list_chunks(conn, eval_layer=eval_layer)
    return materialize_corpus(hits, tokenize)


def materialize_from_manifest(
    tokenize: Callable[[bytes], list[int]],
    *,
    manifest_version: int | None = None,
    manager: CorpusManager | None = None,
) -> Corpus:
    mgr = manager or CorpusManager()
    docs = mgr.load_texts_from_manifest(version=manifest_version)
    return Corpus.from_chunks(docs, tokenize=tokenize)


def materialize_from_retrieval(
    query: str,
    tokenize: Callable[[bytes], list[int]],
    k: int = 15,
    filters: Optional[dict] = None,
) -> Corpus:
    hits = retrieve(query, k=k, filters=filters)
    return materialize_corpus(hits, tokenize)


def _order_hits_pair_adjacent(
    bundle: SynthesisBundle,
) -> list[ChunkHit]:
    """Place edge-linked pairs as consecutive hits for SEAR doc indices."""
    ordered: list[ChunkHit] = []
    seen: set[str] = set()

    for left, right, _edge in bundle.pairs:
        for hit in (left, right):
            if hit.chunk_id not in seen:
                ordered.append(hit)
                seen.add(hit.chunk_id)

    for hit in bundle.hits:
        if hit.chunk_id not in seen:
            ordered.append(hit)
            seen.add(hit.chunk_id)

    return ordered


def _required_doc_groups(
    ordered_hits: list[ChunkHit],
    pairs: list[tuple[ChunkHit, ChunkHit, object]],
) -> list[frozenset[int]]:
    chunk_to_doc_idx = {hit.chunk_id: i for i, hit in enumerate(ordered_hits)}
    groups: list[frozenset[int]] = []
    for left, right, edge in pairs:
        if getattr(edge, "edge_type", None) != "contradicts":
            continue
        li = chunk_to_doc_idx.get(left.chunk_id)
        ri = chunk_to_doc_idx.get(right.chunk_id)
        if li is not None and ri is not None:
            groups.append(frozenset({li, ri}))
    return groups


def materialize_synthesis_bundle(
    bundle: SynthesisBundle,
    tokenize: Callable[[bytes], list[int]],
    *,
    pair_adjacent: bool = True,
    query: Optional[str] = None,
) -> tuple[Corpus, SynthesisContext]:
    hits = _order_hits_pair_adjacent(bundle) if pair_adjacent else list(bundle.hits)
    corpus = materialize_corpus(hits, tokenize)
    doc_index_to_hit = {i: hit for i, hit in enumerate(hits)}
    cite_index_to_doc = {i + 1: i for i in range(len(hits))}
    groups = _required_doc_groups(hits, bundle.pairs)

    chunk_texts = [hit.text for hit in hits]
    preferred_starts: set[tuple[int, int]] = set()
    ranked: list[tuple[int, int, float]] = []
    if query:
        preferred_starts, ranked = score_starts_by_sentence_match(
            corpus, chunk_texts, query, tokenize
        )

    divergence_starts: dict[int, set[int]] = {}
    forbidden_starts: set[tuple[int, int]] = set()
    if bundle.mode == "divergent" and bundle.pairs:
        divergence_starts, forbidden_starts = compute_divergence_zones(
            hits, bundle.pairs, corpus, tokenize
        )
        all_divergence = {
            (d, p) for d, positions in divergence_starts.items() for p in positions
        }
        shared = shared_sentence_starts(hits, corpus, tokenize)
        forbidden_starts -= all_divergence
        first_group_docs: set[int] = set()
        if groups:
            first_group_docs = set(groups[0])
        if all_divergence:
            ranked_div = [
                (d, p, s) for d, p, s in ranked if (d, p) in all_divergence
            ]
            group_divergence = {
                (d, p)
                for d, positions in divergence_starts.items()
                for p in positions
                if d in first_group_docs
            }
            div_preferred = (preferred_starts & all_divergence) - shared
            if group_divergence:
                preferred_starts = group_divergence
            elif div_preferred:
                preferred_starts = {
                    (d, p) for d, p in div_preferred if d in first_group_docs
                } or div_preferred
            elif ranked_div:
                preferred_starts = {
                    (d, p) for d, p, _s in ranked_div if d in first_group_docs
                } or {(ranked_div[0][0], ranked_div[0][1])}
            else:
                preferred_starts = {
                    (d, p) for d, p in all_divergence if d in first_group_docs
                } or all_divergence
        forbidden_starts |= shared
        forbidden_starts -= all_divergence
        divergence_sentence_ends = compute_divergence_sentence_ranges(
            divergence_starts, corpus
        )
    else:
        divergence_sentence_ends = {}

    edge_type_set = {e.edge_type for e in bundle.edges}
    selected_phrases = phrases_for_edge_types(edge_type_set)
    conn_starts, conn_cont, conn_phrases, force_conn = build_connective_inventory(
        tokenize, corpus, phrases=selected_phrases
    )

    retrieval_manifest_hash = ""
    if query is not None:
        retrieval_manifest_hash = build_retrieval_manifest(query, bundle).manifest_hash

    ctx = SynthesisContext(
        doc_index_to_hit=doc_index_to_hit,
        cite_index_to_doc=cite_index_to_doc,
        mode=bundle.mode,
        edges=bundle.edges,
        required_doc_groups=groups,
        preferred_starts=preferred_starts,
        ranked_sentence_starts=ranked,
        divergence_starts=divergence_starts,
        forbidden_starts=forbidden_starts,
        divergence_sentence_ends=divergence_sentence_ends,
        connective_starts=conn_starts,
        connective_continuations=conn_cont,
        connective_phrases=conn_phrases,
        force_connective_ids=force_conn,
        active_edge_types=edge_type_set,
        retrieval_manifest_hash=retrieval_manifest_hash,
    )
    return corpus, ctx


def materialize_from_synthesis(
    query: str,
    tokenize: Callable[[bytes], list[int]],
    *,
    k_seed: int = 5,
    k_max: int = 6,
    filters: Optional[dict] = None,
    min_rrf_delta: float = 0.25,
    confidence_floor: float = RETRIEVAL_CONFIDENCE_FLOOR,
) -> tuple[Corpus, SynthesisContext, SynthesisBundle, RetrievalManifest | None]:
    bundle = retrieve_for_synthesis(
        query,
        k_seed=k_seed,
        k_max=k_max,
        filters=filters,
        min_rrf_delta=min_rrf_delta,
        confidence_floor=confidence_floor,
    )
    corpus, ctx = materialize_synthesis_bundle(bundle, tokenize, query=query)
    retrieval_manifest = build_retrieval_manifest(query, bundle)
    write_retrieval_manifest(retrieval_manifest)
    return corpus, ctx, bundle, retrieval_manifest
