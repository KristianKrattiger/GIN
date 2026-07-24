"""Backfill of pre-relation_class seed contradicts, by register."""
from gin.cartographer.models import Relation
from gin.curator.models import LabelRecord, pair_key
from gin.curator.store import Store
from gin.frames.backfill import SEED_CLASS_BACKFILL, backfill_seed_classes


def _rec(src, dst, relation_class=None, rid="seed-1"):
    return LabelRecord(
        id=rid, src_chunk_id=src, dst_chunk_id=dst,
        relation=Relation.CONTRADICTS, relation_class=relation_class,
        rationale="", curator="seed", ts="2026-07-17T00:00:00Z",
    )


def test_backfill_map_covers_seven_pairs_five_issue_frame():
    assert len(SEED_CLASS_BACKFILL) == 7
    assert sum(1 for v in SEED_CLASS_BACKFILL.values() if v == "issue_frame") == 5
    assert sum(1 for v in SEED_CLASS_BACKFILL.values() if v == "story") == 2


def test_backfill_map_keys_are_sorted_pair_keys():
    for key in SEED_CLASS_BACKFILL:
        assert key == pair_key(*key)


def test_appends_superseding_record_with_class(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    store.append(_rec("inst_em:0", "grass_em:0"))
    assert backfill_seed_classes(store) == 1
    current = store.fold_current()[pair_key("inst_em:0", "grass_em:0")]
    assert current.relation_class == "issue_frame"
    assert current.supersedes == "seed-1"
    assert current.relation is Relation.CONTRADICTS


def test_securities_pairs_tagged_story(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    store.append(_rec("disc_nw_pr:0", "disc_nw_complaint:0", rid="seed-2"))
    backfill_seed_classes(store)
    assert store.fold_current()[pair_key("disc_nw_pr:0", "disc_nw_complaint:0")].relation_class == "story"


def test_is_idempotent(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    store.append(_rec("inst_em:0", "grass_em:0"))
    assert backfill_seed_classes(store) == 1
    assert backfill_seed_classes(store) == 0
    assert len(store.read_log()) == 2


def test_ignores_pairs_already_classified(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    store.append(_rec("inst_em:0", "grass_em:0", relation_class="story"))
    assert backfill_seed_classes(store) == 0
