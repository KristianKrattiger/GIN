"""Append-only JSONL store: round-trip, latest-wins fold, loud on corruption."""
import pytest

from gin.cartographer.models import Relation
from gin.curator.models import LabelRecord, pair_key
from gin.curator.store import Store


def _rec(id, src, dst, relation, ts, relation_class=None):
    return LabelRecord(
        id=id, src_chunk_id=src, dst_chunk_id=dst, relation=relation,
        relation_class=relation_class, rationale="", curator="t", ts=ts,
    )


def test_append_then_read_round_trips(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    r = _rec("1", "a:0", "b:0", Relation.CONTRADICTS, "2026-07-17T00:00:00Z", "story")
    store.append(r)
    assert store.read_log() == [r]


def test_fold_is_latest_wins_per_pair(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    store.append(_rec("1", "a:0", "b:0", Relation.CONTRADICTS, "2026-07-17T00:00:00Z", "story"))
    store.append(_rec("2", "a:0", "b:0", Relation.CORROBORATES, "2026-07-17T01:00:00Z"))
    fold = store.fold_current()
    assert set(fold.keys()) == {pair_key("a:0", "b:0")}
    assert fold[pair_key("a:0", "b:0")].relation is Relation.CORROBORATES


def test_fold_collapses_reversed_pair(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    store.append(_rec("1", "a:0", "b:0", Relation.UNRELATED, "2026-07-17T00:00:00Z"))
    store.append(_rec("2", "b:0", "a:0", Relation.CORROBORATES, "2026-07-17T02:00:00Z"))
    fold = store.fold_current()
    assert len(fold) == 1
    assert fold[pair_key("a:0", "b:0")].relation is Relation.CORROBORATES


def test_gold_returns_reader_shape(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    store.append(_rec("1", "a:0", "b:0", Relation.CONTRADICTS, "2026-07-17T00:00:00Z", "issue_frame"))
    assert store.gold() == [("a:0", "b:0", Relation.CONTRADICTS, "issue_frame")]


def test_read_log_raises_loudly_on_malformed_line(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"id": "1", "src_chunk_id": "a", "dst_chunk_id": "b", "relation": "contradicts", "ts": "2026-07-17T00:00:00Z"}\nnot json at all\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        Store(path).read_log()


def test_empty_store_reads_empty(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    assert store.read_log() == []
    assert store.fold_current() == {}
    assert store.gold() == []
