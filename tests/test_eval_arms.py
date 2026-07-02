"""Tests for eval generation arms."""
from gin.eval.arms import (
    NoContinuationArm,
    _claims_query_relevant,
    _refusal_output,
)
from gin.eval.claims import RawClaim, SpanType


def test_claims_query_relevant_by_overlap():
    claims = [RawClaim(text="cargo throughput rose 12 percent", span_type=SpanType.EXACT.value)]
    chunks = [("port:0", "Harbor cargo throughput rose 12 percent in Q2.")]
    assert _claims_query_relevant(
        "port cargo throughput",
        claims,
        chunks,
        relevance_floor=0.20,
    )


def test_claims_query_relevant_by_cited_chunk_substring():
    claims = [
        RawClaim(
            text="container volume reached 2.1 million TEU",
            span_type=SpanType.EXACT.value,
            cited_chunk_ids=["port:0"],
        )
    ]
    chunks = [("port:0", "Container volume reached 2.1 million TEU in Q2.")]
    assert _claims_query_relevant(
        "container volume TEU",
        claims,
        chunks,
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
    chunks = [("election:0", "The harbor district mayor race tightened ahead of Tuesday's vote.")]
    assert not _claims_query_relevant(
        "port cargo throughput",
        claims,
        chunks,
        relevance_floor=0.20,
    )


def test_no_continuation_refusal_output_shape():
    out = _refusal_output()
    assert out.refused
    assert out.claims == []
