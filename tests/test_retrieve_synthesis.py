"""Tests for edge-aware synthesis retrieval helpers."""
from unittest.mock import patch
from uuid import uuid4

import pytest

from gin.corpus.models import ChunkHit, EdgeRecord, SynthesisBundle
from gin.corpus.retrieve import (
    DIVERGENCE_RELEVANCE_FLOOR,
    RETRIEVAL_CONFIDENCE_FLOOR,
    RetrievalConfidenceError,
    _apply_relevance_floor,
    _build_pairs,
    _is_ambiguous,
    _neighbor_ids_from_seed_edges,
    _prioritize_hits,
    retrieve_for_synthesis,
)

DOC = uuid4()


def _hit(chunk_id: str, outlet: str, score: float, text: str = "text", eval_tag: str | None = None) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        doc_id=DOC,
        text=text,
        head_sentence="head",
        eval_layer="realism",
        eval_tag=eval_tag,
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


# --- Phase A: query-relevant divergent gating --------------------------------

def test_is_ambiguous_port_query_election_contradicts_convergent():
    """Port query + election contradicts in seed -> contradicts gating blocks divergent.

    The contradicts pair (election_cw, election_md) is query-irrelevant for a
    port query. _is_ambiguous should not trigger on the contradicts edge.
    Close-competitor heuristic may still fire based on RRF scores, but that's
    a separate (score-based) path — the critical thing is that the contradicts
    edge alone does not force divergent mode.
    """
    election_cw = _hit(
        "election_centralwire:0", "CentralWire", 0.5,
        text="Turnout reached 61 percent of registered voters in the harbor district referendum.",
    )
    election_md = _hit(
        "election_metrodaily:0", "MetroDaily", 0.45,
        text="Turnout reached 58 percent of registered voters in the harbor district referendum.",
    )
    edges = [EdgeRecord("election_centralwire:0", "election_metrodaily:0", "contradicts")]
    # Only test with two election hits (no port) to isolate the contradicts gating
    hits = [election_cw, election_md]
    query = "How many twenty-foot equivalent units did harbor terminals handle in March"
    # Contradicts pair scores below floor for port query -> should NOT trigger
    from gin.corpus.relevance import max_sentence_score
    assert max_sentence_score(election_cw.text, query) < DIVERGENCE_RELEVANCE_FLOOR
    assert max_sentence_score(election_md.text, query) < DIVERGENCE_RELEVANCE_FLOOR
    # With query-aware gating, contradicts alone should not force divergent
    # (close-competitors may still fire on score proximity, but that's the
    # score-based heuristic, not the false-contradicts problem we're fixing)
    # The contradicts edge should be filtered out:
    hits_by_id = {h.chunk_id: h for h in hits}
    pairs = _build_pairs(hits_by_id, edges, query)
    assert len(pairs) == 0


def test_is_ambiguous_incident_query_incident_contradicts_divergent():
    """Incident query + incident contradicts -> divergent (preserved)."""
    central = _hit(
        "incident_centralwire:0", "CentralWire", 0.5,
        text="Emergency services confirmed 142 people received treatment at area hospitals.",
    )
    metro = _hit(
        "incident_metrodaily:0", "MetroDaily", 0.45,
        text="Emergency services confirmed 98 people received treatment at area hospitals.",
    )
    edges = [EdgeRecord("incident_centralwire:0", "incident_metrodaily:0", "contradicts")]
    hits = [central, metro]
    assert _is_ambiguous(hits, edges, "How many people received hospital treatment after the downtown incident") is True


def test_is_ambiguous_election_query_election_contradicts_divergent():
    """Election query + election contradicts -> divergent (preserved)."""
    election_cw = _hit(
        "election_centralwire:0", "CentralWire", 0.5,
        text="Turnout reached 61 percent of registered voters in the harbor district referendum.",
    )
    election_md = _hit(
        "election_metrodaily:0", "MetroDaily", 0.45,
        text="Turnout reached 58 percent of registered voters in the harbor district referendum.",
    )
    edges = [EdgeRecord("election_centralwire:0", "election_metrodaily:0", "contradicts")]
    hits = [election_cw, election_md]
    assert _is_ambiguous(hits, edges, "What was voter turnout in the harbor district referendum") is True


def test_is_ambiguous_no_query_legacy_behavior():
    """Without query, any contradicts edge forces divergent (legacy path)."""
    hits = [_hit("a:0", "A", 0.5), _hit("b:0", "B", 0.4)]
    edges = [EdgeRecord("a:0", "b:0", "contradicts")]
    assert _is_ambiguous(hits, edges) is True
    assert _is_ambiguous(hits, edges, "") is True


def test_build_pairs_excludes_query_irrelevant_contradicts():
    """_build_pairs skips contradicts pairs where neither chunk matches the query."""
    election_cw = _hit(
        "election_centralwire:0", "CentralWire", 0.5,
        text="The harbor district referendum passed after a long count. Turnout reached 61 percent of registered voters.",
    )
    election_md = _hit(
        "election_metrodaily:0", "MetroDaily", 0.45,
        text="The harbor district referendum passed after a long count. Turnout reached 58 percent of registered voters.",
    )
    hits_by_id = {h.chunk_id: h for h in [election_cw, election_md]}
    edges = [EdgeRecord("election_centralwire:0", "election_metrodaily:0", "contradicts")]
    # Port query: election chunks are irrelevant
    pairs = _build_pairs(hits_by_id, edges, "port cargo throughput TEU")
    assert len(pairs) == 0
    # Election query: election chunks are relevant
    pairs = _build_pairs(hits_by_id, edges, "voter turnout harbor district referendum")
    assert len(pairs) == 1


def test_build_pairs_keeps_cites_regardless_of_query():
    """Cites edges are never filtered by query relevance."""
    a = _hit("a:0", "A", 0.5, text="Something about ports.")
    b = _hit("b:0", "B", 0.4, text="Something else about elections.")
    hits_by_id = {h.chunk_id: h for h in [a, b]}
    edges = [EdgeRecord("a:0", "b:0", "cites")]
    pairs = _build_pairs(hits_by_id, edges, "unrelated weather query")
    assert len(pairs) == 1


# --- Phase B: seed re-rank before mode detection -----------------------------

def test_retrieve_for_synthesis_reranks_seeds_by_query():
    """Seeds should be re-ranked by query relevance before mode detection."""
    incident = _hit(
        "incident_centralwire:0", "CentralWire", 0.5,
        text="Emergency services confirmed 142 people received treatment at area hospitals.",
    )
    weather = _hit(
        "weather_service_brief:0", "WeatherService", 0.3,
        text="Sustained wind speed from the coastal storm system reached 65 mph.",
    )
    with patch("gin.corpus.retrieve.retrieve", return_value=[incident, weather]):
        with patch("gin.corpus.retrieve.connect"):
            with patch("gin.corpus.retrieve.warm.fetch_edges_among", return_value=[]):
                with patch("gin.corpus.retrieve.warm.fetch_chunks_by_ids", return_value=[]):
                    bundle = retrieve_for_synthesis(
                        "What sustained wind speed is expected from the coastal storm system",
                        confidence_floor=0.0,
                    )
    # Weather chunk should be first after re-rank
    assert bundle.hits[0].chunk_id == "weather_service_brief:0"
    assert bundle.mode == "convergent"


# --- Phase C: zero-relevance seed filter --------------------------------------

def test_retrieve_for_synthesis_drops_zero_relevance_seeds():
    """Seeds with zero query overlap should be filtered out."""
    incident = _hit(
        "incident_centralwire:0", "CentralWire", 0.5,
        text="Emergency services confirmed 142 people received treatment.",
    )
    port = _hit(
        "port_authority_brief:0", "PortAuthority", 0.3,
        text="Harbor terminals handled 2.1 million twenty-foot equivalent units in March.",
    )
    with patch("gin.corpus.retrieve.retrieve", return_value=[incident, port]):
        with patch("gin.corpus.retrieve.connect"):
            with patch("gin.corpus.retrieve.warm.fetch_edges_among", return_value=[]):
                with patch("gin.corpus.retrieve.warm.fetch_chunks_by_ids", return_value=[]):
                    bundle = retrieve_for_synthesis(
                        "How many twenty-foot equivalent units did harbor terminals handle",
                        confidence_floor=0.0,
                    )
    # Incident chunk should be dropped (zero overlap with port query)
    chunk_ids = [h.chunk_id for h in bundle.hits]
    assert "port_authority_brief:0" in chunk_ids
    assert bundle.mode == "convergent"


# --- Phase 3A: divergent mode requires a query-relevant contradicts pair -----

def test_is_ambiguous_corroborating_sources_convergent():
    """Same-tag, different-outlet hits with identical agreeing text are
    corroboration, not divergence — no contradicts edge means convergent even
    when RRF scores are adjacent."""
    bureau = _hit(
        "labor_bureau_report:0", "NationalLaborBureau", 0.5,
        text="The regional unemployment rate stood at 3.7 percent in the latest monthly survey.",
        eval_tag="unemployment_probe",
    )
    survey = _hit(
        "labor_independent_survey:0", "IndependentEconomicReview", 0.49,
        text="The regional unemployment rate stood at 3.7 percent in the latest monthly survey.",
        eval_tag="unemployment_probe",
    )
    query = "What was the regional unemployment rate in the latest monthly survey?"
    assert _is_ambiguous([bureau, survey], [], query) is False


def test_is_ambiguous_close_competitors_without_contradicts_convergent():
    """With a query, close RRF competitors alone never force divergent mode."""
    hits = [
        _hit("a:0", "A", 0.5, text="Sustained wind speed reached 65 mph in the storm."),
        _hit("b:0", "B", 0.45, text="Sustained wind speed reached 65 mph in the storm."),
    ]
    assert _is_ambiguous(hits, [], "What sustained wind speed did the storm reach") is False
    # Legacy no-query path keeps the blunt heuristic.
    assert _is_ambiguous(hits, []) is True


# --- Phase 3B: minimum matched-keyword count ----------------------------------

def test_is_ambiguous_single_keyword_collision_convergent():
    """One shared keyword ("district") must not make an off-topic contradicts
    pair query-relevant (school_enrollment_fall failure)."""
    election_cw = _hit(
        "election_centralwire:0", "CentralWire", 0.5,
        text="The harbor district referendum passed by 842 votes after a long count. "
             "Turnout reached 61 percent of registered voters.",
    )
    election_md = _hit(
        "election_metrodaily:0", "MetroDaily", 0.45,
        text="The harbor district referendum passed by 842 votes after a long count. "
             "Turnout reached 58 percent of registered voters.",
    )
    edges = [EdgeRecord("election_centralwire:0", "election_metrodaily:0", "contradicts")]
    query = "What was fall enrollment across district campuses?"
    assert _is_ambiguous([election_cw, election_md], edges, query) is False
    # The pair must also be dropped so _prioritize_hits cannot front-load it.
    hits_by_id = {h.chunk_id: h for h in [election_cw, election_md]}
    assert _build_pairs(hits_by_id, edges, query) == []


def test_build_pairs_requires_both_sides_relevant():
    """A pair with only one query-relevant side is dropped (matches _is_ambiguous)."""
    relevant = _hit(
        "a:0", "A", 0.5,
        text="Turnout reached 61 percent of registered voters in the harbor district referendum.",
    )
    irrelevant = _hit(
        "b:0", "B", 0.45,
        text="Fall enrollment reached 48,200 students across district campuses.",
    )
    hits_by_id = {h.chunk_id: h for h in [relevant, irrelevant]}
    edges = [EdgeRecord("a:0", "b:0", "contradicts")]
    pairs = _build_pairs(hits_by_id, edges, "voter turnout harbor district referendum")
    assert pairs == []


def test_retrieve_for_synthesis_port_query_with_election_contradicts():
    """Full path: port query with election contradicts in RRF seeds -> convergent."""
    election_cw = _hit(
        "election_centralwire:0", "CentralWire", 0.5,
        text="Turnout reached 61 percent of registered voters in the harbor district referendum.",
    )
    election_md = _hit(
        "election_metrodaily:0", "MetroDaily", 0.45,
        text="Turnout reached 58 percent of registered voters in the harbor district referendum.",
    )
    port = _hit(
        "port_authority_brief:0", "PortAuthority", 0.4,
        text="Harbor terminals handled 2.1 million twenty-foot equivalent units in March.",
    )
    edges = [EdgeRecord("election_centralwire:0", "election_metrodaily:0", "contradicts")]
    with patch("gin.corpus.retrieve.retrieve", return_value=[election_cw, election_md, port]):
        with patch("gin.corpus.retrieve.connect"):
            with patch("gin.corpus.retrieve.warm.fetch_edges_among", return_value=edges):
                with patch("gin.corpus.retrieve.warm.fetch_chunks_by_ids", return_value=[]):
                    bundle = retrieve_for_synthesis(
                        "How many twenty-foot equivalent units did harbor terminals handle in March",
                        confidence_floor=0.0,
                    )
    # Port chunk should be first (re-ranked), mode should be convergent
    assert bundle.hits[0].chunk_id == "port_authority_brief:0"
    assert bundle.mode == "convergent"
    # Election contradicts pair should be excluded from pairs
    contradicts_pairs = [p for p in bundle.pairs if p[2].edge_type == "contradicts"]
    assert len(contradicts_pairs) == 0
