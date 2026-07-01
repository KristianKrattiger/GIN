"""Tests for synthesis materialization ordering."""
from uuid import uuid4

from gin.corpus.materialize import _order_hits_pair_adjacent, _required_doc_groups
from gin.corpus.models import ChunkHit, EdgeRecord, SynthesisBundle


def _hit(chunk_id: str) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        doc_id=uuid4(),
        text=f"text for {chunk_id}",
        head_sentence="head",
        eval_layer="realism",
        eval_tag=None,
        content_hash="x",
        outlet="O",
        title="T",
        rrf_score=0.5,
    )


def test_order_hits_pair_adjacent():
    left = _hit("a:0")
    right = _hit("b:0")
    other = _hit("c:0")
    edge = EdgeRecord("a:0", "b:0", "contradicts")
    bundle = SynthesisBundle(
        hits=[other, left, right],
        edges=[edge],
        mode="divergent",
        pairs=[(left, right, edge)],
    )
    ordered = _order_hits_pair_adjacent(bundle)
    ids = [h.chunk_id for h in ordered]
    assert ids.index("a:0") < ids.index("c:0")
    assert ids.index("b:0") == ids.index("a:0") + 1


def test_required_doc_groups_from_contradicts():
    hits = [_hit("a:0"), _hit("b:0")]
    edge = EdgeRecord("a:0", "b:0", "contradicts")
    groups = _required_doc_groups(hits, [(hits[0], hits[1], edge)])
    assert groups == [frozenset({0, 1})]
