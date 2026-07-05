"""Tests for eval generation arms."""
from gin.eval.arms import (
    NoContinuationArm,
    _claims_query_relevant,
    _refusal_output,
)
from gin.eval.claims import RawClaim, SpanType


def test_claims_query_relevant_by_overlap():
    claims = [RawClaim(text="cargo throughput rose 12 percent", span_type=SpanType.EXACT.value)]
    assert _claims_query_relevant(
        "port cargo throughput",
        claims,
        relevance_floor=0.20,
    )


def test_claims_query_relevant_by_diluted_keyword():
    # Direct overlap is diluted below the floor by extra numeric tokens, but the
    # claim itself still carries the distinctive query keywords -> relevant.
    claims = [
        RawClaim(
            text="container volume reached 2.1 million TEU",
            span_type=SpanType.EXACT.value,
            cited_chunk_ids=["port:0"],
        )
    ]
    assert _claims_query_relevant(
        "container volume TEU",
        claims,
        relevance_floor=0.90,
    )


def test_claims_query_relevant_rejects_off_topic():
    claims = [
        RawClaim(
            text="mayor race tightened ahead of Tuesday vote",
            span_type=SpanType.EXACT.value,
            cited_chunk_ids=["election:0"],
        )
    ]
    assert not _claims_query_relevant(
        "port cargo throughput",
        claims,
        relevance_floor=0.20,
    )


def test_claims_query_relevant_refuses_out_of_scope_with_contradicts_edge():
    # Out-of-scope regression (plan §6 #4): the query asks a vote MARGIN; the
    # retrieved harbor-referendum chunks carry a contradicts edge (turnout 61 vs
    # 58), so the query routes to divergent mode and decodes both turnout spans.
    # Each turnout claim is a substring of an on-topic (referendum) chunk, but
    # the claim itself shares no query keyword -> must refuse, not answer.
    claims = [
        RawClaim(
            text="Turnout reached 61 percent of registered voters.",
            span_type=SpanType.EXACT.value,
            cited_chunk_ids=["election_centralwire:0"],
        ),
        RawClaim(
            text="Turnout reached 58 percent of registered voters.",
            span_type=SpanType.EXACT.value,
            cited_chunk_ids=["election_metrodaily:0"],
        ),
    ]
    assert not _claims_query_relevant(
        "By how many votes did the harbor district referendum pass?",
        claims,
        relevance_floor=0.20,
    )


def test_no_continuation_refusal_output_shape():
    out = _refusal_output()
    assert out.refused
    assert out.claims == []
