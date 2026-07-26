"""hf_af_*/hf_kc_* are story, not issue_frame — restoring guide/seed consistency."""
from gin.cartographer.models import Relation
from gin.curator.models import LabelRecord, pair_key
from gin.curator.relabel_hf import HF_STORY_RELABEL, relabel_hf_to_story
from gin.curator.store import Store


def _rec(src, dst, relation_class, rid="orig-1"):
    return LabelRecord(
        id=rid, src_chunk_id=src, dst_chunk_id=dst, relation=Relation.CONTRADICTS,
        relation_class=relation_class, rationale="", curator="backfill",
        ts="2026-07-24T00:00:00Z",
    )


def test_map_covers_exactly_the_two_housing_pairs():
    assert len(HF_STORY_RELABEL) == 2
    assert set(HF_STORY_RELABEL.values()) == {"story"}
    assert pair_key("hf_af_staff:0", "hf_af_tenants:0") in HF_STORY_RELABEL
    assert pair_key("hf_kc_inspection:0", "hf_kc_tenants:0") in HF_STORY_RELABEL


def test_relabels_issue_frame_to_story(tmp_path):
    store = Store(tmp_path / "l.jsonl")
    store.append(_rec("hf_af_staff:0", "hf_af_tenants:0", "issue_frame"))
    assert relabel_hf_to_story(store) == 1
    current = store.fold_current()[pair_key("hf_af_staff:0", "hf_af_tenants:0")]
    assert current.relation_class == "story"
    assert current.relation is Relation.CONTRADICTS
    assert current.supersedes == "orig-1"


def test_is_idempotent(tmp_path):
    store = Store(tmp_path / "l.jsonl")
    store.append(_rec("hf_af_staff:0", "hf_af_tenants:0", "issue_frame"))
    assert relabel_hf_to_story(store) == 1
    assert relabel_hf_to_story(store) == 0
    assert len(store.read_log()) == 2


def test_leaves_already_story_alone(tmp_path):
    store = Store(tmp_path / "l.jsonl")
    store.append(_rec("hf_af_staff:0", "hf_af_tenants:0", "story"))
    assert relabel_hf_to_story(store) == 0


def test_does_not_touch_other_pairs(tmp_path):
    store = Store(tmp_path / "l.jsonl")
    store.append(_rec("inst_em:0", "grass_em:0", "issue_frame", rid="keep"))
    assert relabel_hf_to_story(store) == 0
    assert store.fold_current()[pair_key("inst_em:0", "grass_em:0")].relation_class == "issue_frame"
