"""Tests for edge-aware synthesis retrieval helpers."""
from unittest.mock import patch
from uuid import uuid4

import pytest

from gin.corpus.models import ChunkHit, EdgeRecord, SynthesisBundle
from gin.corpus.retrieve import (
    RETRIEVAL_CONFIDENCE_FLOOR,
    RetrievalConfidenceError,
    _apply_relevance_floor,
    _build_pairs,
    _is_ambiguous,
    _neighbor_ids_from_seed_edges,
    _prioritize_hits,
    retrieve_for_synthesis,
)


def _hit(chunk_id: str, outlet: str, score: float) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        doc_id=uuid4(),
        text="text",
        head_sentence="head",
        eval_layer="realism",
        eval_tag=None,
        content_hash="x",
        outlet=outlet,
        title="title",
        rrf_score=score,
    )


def test_is_ambiguous_on_contradicts_edge():
    hits = [_hit("a:0", "A", 0.5), _hit("b:0", "B", 0.4)]
    edges = [EdgeRecord("a:0", "b:0", "contradicts")]
    assert _is_ambiguous(hits, edges) is True


def test_is_ambiguous_on_close_competitors():
    hits = [_hit("a:0", "A", 0.5), _hit("b:0", "B", 0.45)]
    assert _is_ambiguous(hits, []) is True


def test_apply_relevance_floor():
    hits = [_hit("a:0", "A", 0.5), _hit("b:0", "B", 0.1)]
    filtered = _apply_relevance_floor(hits, 0.25)
    assert len(filtered) == 1
    assert filtered[0].chunk_id == "a:0"


def test_neighbor_ids_from_seed_edges():
    edges = [
        EdgeRecord("a:0", "b:0", "contradicts"),
        EdgeRecord("b:0", "c:0", "contradicts"),
    ]
    neighbors = _neighbor_ids_from_seed_edges({"a:0"}, edges)
    assert neighbors == {"b:0"}
    assert "c:0" not in neighbors


def test_prioritize_hits_orders_contradict_pair_first():
    seed = [_hit("a:0", "A", 0.5)]
    neighbors = [_hit("b:0", "B", 0.4)]
    left, right = seed[0], neighbors[0]
    edge = EdgeRecord("a:0", "b:0", "contradicts")
    pairs = [(left, right, edge)]
    ordered = _prioritize_hits(seed, neighbors, pairs, k_max=4, min_rrf_delta=0.0)
    assert ordered[0].chunk_id == "a:0"
    assert ordered[1].chunk_id == "b:0"


def test_build_pairs_from_edges():
    left = _hit("a:0", "A", 0.5)
    right = _hit("b:0", "B", 0.4)
    hits_by_id = {h.chunk_id: h for h in [left, right]}
    edges = [EdgeRecord("a:0", "b:0", "contradicts")]
    pairs = _build_pairs(hits_by_id, edges)
    assert len(pairs) == 1
    assert pairs[0][0].chunk_id == "a:0"
    assert pairs[0][1].chunk_id == "b:0"


def test_retrieval_confidence_floor_raises_on_low_score():
    low_hit = _hit("a:0", "A", 0.005)
    with patch("gin.corpus.retrieve.retrieve", return_value=[low_hit]):
        with patch("gin.corpus.retrieve.connect"):
            with patch("gin.corpus.retrieve.warm.fetch_edges_among", return_value=[]):
                with patch("gin.corpus.retrieve.warm.fetch_chunks_by_ids", return_value=[]):
                    with pytest.raises(RetrievalConfidenceError) as exc:
                        retrieve_for_synthesis("test query")
    assert exc.value.top_score == 0.005
    assert exc.value.floor == RETRIEVAL_CONFIDENCE_FLOOR


def test_retrieval_confidence_floor_passes_on_high_score():
    high_hit = _hit("a:0", "A", 0.020)
    with patch("gin.corpus.retrieve.retrieve", return_value=[high_hit]):
        with patch("gin.corpus.retrieve.connect"):
            with patch("gin.corpus.retrieve.warm.fetch_edges_among", return_value=[]):
                with patch("gin.corpus.retrieve.warm.fetch_chunks_by_ids", return_value=[]):
                    bundle = retrieve_for_synthesis("test query")
    assert isinstance(bundle, SynthesisBundle)
    assert bundle.hits[0].rrf_score == 0.020


def test_retrieval_confidence_floor_disabled_at_zero():
    low_hit = _hit("a:0", "A", 0.005)
    with patch("gin.corpus.retrieve.retrieve", return_value=[low_hit]):
        with patch("gin.corpus.retrieve.connect"):
            with patch("gin.corpus.retrieve.warm.fetch_edges_among", return_value=[]):
                with patch("gin.corpus.retrieve.warm.fetch_chunks_by_ids", return_value=[]):
                    bundle = retrieve_for_synthesis("test query", confidence_floor=0.0)
    assert bundle.hits[0].rrf_score == 0.005
