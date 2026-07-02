"""Tests for claim segmentation and node-scope assignment."""
from gin.eval.claims import (
    NodeScope,
    SpanType,
    extract_citation_indices,
    node_scope_for,
    parse_citation_indices,
    rag_text_to_raw_claims,
    segments_to_raw_claims,
    strip_citation_markers,
)
from sear.processor import Segment

_VOCAB = {1: "Emergency", 2: "services", 3: "142", 4: "people", 5: "the", 6: "mayor"}


def _detok(ids):
    return " ".join(_VOCAB[i] for i in ids)


def test_segments_to_raw_claims_exact_and_ambiguous():
    segments = [
        Segment([1, 2, 3, 4], [(0, 0, 4)], "extract"),
        Segment([99], [], "connective"),
        Segment([5, 6], [(0, 0, 2), (1, 0, 2)], "extract"),
    ]
    doc_index_to_chunk_id = {0: "chunk_a", 1: "chunk_b"}
    claims = segments_to_raw_claims(segments, _detok, doc_index_to_chunk_id)

    assert len(claims) == 2  # connective skipped
    assert claims[0].span_type == SpanType.EXACT.value
    assert claims[0].cited_chunk_ids == ["chunk_a"]
    assert claims[0].text == "Emergency services 142 people"
    assert claims[1].span_type == SpanType.AMBIGUOUS.value
    assert claims[1].cited_chunk_ids == ["chunk_a", "chunk_b"]


def test_rag_text_to_raw_claims_parses_citations():
    text = "Emergency services confirmed 142 people [1]. The mayor scheduled a briefing [2]!"
    cite_index_to_chunk_id = {1: "chunk_a", 2: "chunk_b"}
    claims = rag_text_to_raw_claims(text, cite_index_to_chunk_id)

    assert len(claims) == 2
    assert all(c.span_type == SpanType.GENERATED.value for c in claims)
    assert claims[0].cited_chunk_ids == ["chunk_a"]
    assert "[1]" not in claims[0].text
    assert claims[1].cited_chunk_ids == ["chunk_b"]


def test_rag_text_uncited_sentence_has_no_sources():
    claims = rag_text_to_raw_claims("Something happened downtown.", {1: "chunk_a"})
    assert len(claims) == 1
    assert claims[0].cited_chunk_ids == []


def test_parse_citation_indices_chunk_id_suffix():
    assert parse_citation_indices("1: incident_regionalpost:0") == [1]
    assert parse_citation_indices("1: incident_regionalpost:0, 2: incident_centralwire:0") == [
        1,
        2,
    ]


def test_parse_citation_indices_comma_list():
    assert parse_citation_indices("4, 5") == [4, 5]
    assert parse_citation_indices("1:0, 2:0") == [1, 2]


def test_rag_text_parses_mistral_citation_variants():
    cite_index_to_chunk_id = {
        1: "incident_regionalpost:0",
        2: "incident_centralwire:0",
        3: "incident_metrodaily:0",
        4: "labor_bureau_report:0",
        5: "labor_independent_survey:0",
    }
    text = (
        "The sources confirm that 142 people received treatment "
        "[1: incident_regionalpost:0, 2: incident_centralwire:0, 3: incident_metrodaily:0]."
    )
    claims = rag_text_to_raw_claims(text, cite_index_to_chunk_id)
    assert len(claims) == 1
    assert claims[0].cited_chunk_ids == [
        "incident_regionalpost:0",
        "incident_centralwire:0",
        "incident_metrodaily:0",
    ]
    assert "[1:" not in claims[0].text

    text2 = "The regional unemployment rate stood at 3.7 percent [4, 5]."
    claims2 = rag_text_to_raw_claims(text2, cite_index_to_chunk_id)
    assert claims2[0].cited_chunk_ids == ["labor_bureau_report:0", "labor_independent_survey:0"]


def test_strip_citation_markers():
    sentence = "Passed by 842 votes [1:0, 2:0]."
    assert strip_citation_markers(sentence) == "Passed by 842 votes ."
    assert extract_citation_indices(sentence) == [1, 2]


def test_node_scope_classification():
    node_of = {"chunk_a": "CentralWire", "chunk_b": "MetroDaily"}.get
    assert node_scope_for(["chunk_a"], node_of) == NodeScope.WITHIN_NODE
    assert node_scope_for(["chunk_a", "chunk_b"], node_of) == NodeScope.CROSS_NODE
    assert node_scope_for([], node_of) == NodeScope.NONE
