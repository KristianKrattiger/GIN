"""Seeding imports the existing ~33 labels; the regression guard proves the
store reproduces today's gold before it grows it."""
import types

from gin.cartographer import gold_edges, labeled_set
from gin.cartographer.models import Relation
from gin.curator.models import pair_key
from gin.curator.seed import seed_store
from gin.curator.store import Store


def test_seed_reproduces_labeled_set_relations(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    seed_store(store)
    fold = store.fold_current()
    for src, dst, relation, _register in labeled_set.gold():
        key = pair_key(src, dst)
        assert key in fold, f"seeded pair missing: {key}"
        assert fold[key].relation is relation


def test_labeled_set_contradicts_seed_with_null_class(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    seed_store(store)
    fold = store.fold_current()
    # inst_em:0 <-> grass_em:0 is a labeled_set CONTRADICTS pair; labeled_set
    # carries no story/issue_frame tag, so it seeds as None (not a guess).
    rec = fold[pair_key("inst_em:0", "grass_em:0")]
    assert rec.relation is Relation.CONTRADICTS
    assert rec.relation_class is None


def test_gold_edges_class_is_preserved(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    seed_store(store)
    fold = store.fold_current()
    edges = gold_edges.load_all_gold_contradicts()
    assert edges  # fixtures exist in-repo
    # Full coverage: EVERY gold_edges pair must be present with a real class. A
    # silently dropped pair must FAIL here (the old `if key in fold` filter let
    # a partial-drop regression pass as long as one pair survived).
    for e in edges:
        key = pair_key(e.src_chunk_id, e.dst_chunk_id)
        assert key in fold, f"gold_edges pair dropped by seed: {key}"
        assert fold[key].relation_class in {"story", "issue_frame"}


def test_seed_is_idempotent(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    first = seed_store(store)
    assert first > 0
    assert seed_store(store) == 0


def test_collision_labeled_set_wins(monkeypatch, tmp_path):
    # Real gold has no cross-source pair collision, so nothing else exercises
    # the docstring's "labeled_set is emitted first, so it wins any collision"
    # claim. Force a synthetic collision: labeled_set says CORROBORATES (class
    # None), gold_edges says the same pair is CONTRADICTS/story. labeled_set is
    # looped first in seed_records, so it must win. This FAILS if the two loops
    # were ever swapped.
    monkeypatch.setattr(
        labeled_set, "gold",
        lambda: [("x:0", "y:0", Relation.CORROBORATES, "reg")],
    )
    monkeypatch.setattr(
        gold_edges, "load_all_gold_contradicts",
        lambda: [types.SimpleNamespace(
            src_chunk_id="x:0", dst_chunk_id="y:0",
            relation_class="story", note="",
        )],
    )
    store = Store(tmp_path / "labels.jsonl")
    seed_store(store)
    rec = store.fold_current()[pair_key("x:0", "y:0")]
    assert rec.relation is Relation.CORROBORATES  # labeled_set (first) won
    assert rec.relation_class is None
