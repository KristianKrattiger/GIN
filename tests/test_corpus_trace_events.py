"""Corpus-tier trace primitives: dependency-free event types + ContextVar sink."""
from gin.corpus.models import ChunkHit, SynthesisContext, doc_index_to_chunk_id
from gin.corpus.trace_events import (
    ClaimClosedTrace,
    RetrievalSettledTrace,
    current_trace_sink,
)
from uuid import uuid4

DOC = uuid4()


def _hit(chunk_id: str) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id, doc_id=DOC, text="text", head_sentence="head",
        eval_layer="counterfactual", eval_tag=None, content_hash="x",
        outlet="o", title="t", rrf_score=0.5,
    )


def test_doc_index_to_chunk_id_maps_from_context():
    ctx = SynthesisContext(
        doc_index_to_hit={0: _hit("a:0"), 1: _hit("b:0")},
        cite_index_to_doc={1: 0, 2: 1},
        mode="convergent",
    )
    assert doc_index_to_chunk_id(ctx) == {0: "a:0", 1: "b:0"}


def test_current_trace_sink_defaults_to_none():
    assert current_trace_sink.get() is None


def test_current_trace_sink_set_and_reset():
    received = []
    token = current_trace_sink.set(received.append)
    try:
        sink = current_trace_sink.get()
        assert sink is not None
        sink(RetrievalSettledTrace(synthesis_mode="convergent", manifest_hash="h", chunk_count=3))
        sink(ClaimClosedTrace(text="claim text", span_type="EXACT", cited_chunk_ids=["a:0"]))
    finally:
        current_trace_sink.reset(token)
    assert current_trace_sink.get() is None
    assert len(received) == 2
    assert isinstance(received[0], RetrievalSettledTrace)
    assert isinstance(received[1], ClaimClosedTrace)
    assert received[1].cited_chunk_ids == ["a:0"]
