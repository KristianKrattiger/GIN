"""LabelRecord (de)serialization and order-independent pair keys."""
from gin.cartographer.models import Relation
from gin.curator.models import LabelRecord, pair_key


def _rec(**kw):
    base = dict(
        id="r1", src_chunk_id="b:0", dst_chunk_id="a:0",
        relation=Relation.CONTRADICTS, relation_class="issue_frame",
        rationale="opposing frames", curator="kristian", ts="2026-07-17T00:00:00Z",
    )
    base.update(kw)
    return LabelRecord(**base)


def test_pair_key_is_order_independent():
    assert pair_key("a:0", "b:0") == pair_key("b:0", "a:0") == ("a:0", "b:0")


def test_to_json_round_trips():
    rec = _rec(src_anchor=(0, 3), dst_anchor=(0, 5))
    d = rec.to_json()
    assert d["relation"] == "contradicts"
    assert d["relation_class"] == "issue_frame"
    assert d["src_anchor"] == [0, 3]
    back = LabelRecord.from_json(d)
    assert back == rec


def test_from_json_handles_null_class_and_anchors():
    rec = _rec(relation=Relation.CORROBORATES, relation_class=None)
    back = LabelRecord.from_json(rec.to_json())
    assert back.relation is Relation.CORROBORATES
    assert back.relation_class is None
    assert back.src_anchor is None
