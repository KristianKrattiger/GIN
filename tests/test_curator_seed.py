"""Seeding imports the existing ~33 labels; the regression guard proves the
store reproduces today's gold before it grows it."""
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
    classed = [
        fold[pair_key(e.src_chunk_id, e.dst_chunk_id)]
        for e in edges
        if pair_key(e.src_chunk_id, e.dst_chunk_id) in fold
    ]
    assert classed
    assert all(r.relation_class in {"story", "issue_frame"} for r in classed)


def test_seed_is_idempotent(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    first = seed_store(store)
    assert first > 0
    assert seed_store(store) == 0
