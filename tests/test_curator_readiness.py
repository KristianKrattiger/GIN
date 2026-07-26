"""No-model readiness gauge: counts NEW (non-bar) labels per class vs a target."""
from gin.cartographer.escalation_eval import default_calibration_sets
from gin.cartographer.models import Relation
from gin.curator.models import LabelRecord
from gin.curator.readiness import ReadinessTarget, bar_pair_keys, readiness
from gin.curator.store import Store


def _rec(src, dst, relation, ts, relation_class=None):
    return LabelRecord(id=f"{src}-{dst}", src_chunk_id=src, dst_chunk_id=dst,
                       relation=relation, relation_class=relation_class,
                       rationale="", curator="t", ts=ts)


def test_bar_pairs_have_14_keys():
    assert len(bar_pair_keys()) == 14  # 4 issue_frame + 6 corroboration + 4 unrelated


def test_seeded_bar_issue_frame_counts_as_zero_new(tmp_path):
    # A store holding ONLY the 4 escalation-bar issue_frame pairs => 0 new.
    store = Store(tmp_path / "labels.jsonl")
    for i, (src, dst, _reg) in enumerate(default_calibration_sets()["issue_frame"]):
        store.append(_rec(src, dst, Relation.CONTRADICTS, f"2026-07-17T00:00:0{i}Z",
                          relation_class="issue_frame"))
    rep = readiness(store)
    assert rep.new_issue_frame == 0
    assert rep.ready is False


def test_counts_new_labels_and_verdict_flips(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    # story=0: this test predates the story class and exercises only the
    # issue_frame/agree/unrelated counting, so require none of it here.
    target = ReadinessTarget(issue_frame=2, agree=1, unrelated=1, story=0)
    # Two NEW issue_frame (non-bar), one agree, one unrelated.
    store.append(_rec("x:0", "y:0", Relation.CONTRADICTS, "2026-07-17T00:00:00Z", "issue_frame"))
    store.append(_rec("x:1", "y:1", Relation.CONTRADICTS, "2026-07-17T00:00:01Z", "issue_frame"))
    store.append(_rec("p:0", "q:0", Relation.CORROBORATES, "2026-07-17T00:00:02Z"))
    store.append(_rec("m:0", "n:0", Relation.UNRELATED, "2026-07-17T00:00:03Z"))
    rep = readiness(store, target)
    assert (rep.new_issue_frame, rep.new_agree, rep.new_unrelated, rep.new_story) == (2, 1, 1, 0)
    assert rep.ready is True
    # One short on issue_frame => not ready.
    assert readiness(store, ReadinessTarget(issue_frame=3, agree=1, unrelated=1, story=0)).ready is False


def test_none_class_contradicts_not_counted_as_issue_frame(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    store.append(_rec("x:0", "y:0", Relation.CONTRADICTS, "2026-07-17T00:00:00Z", None))
    assert readiness(store).new_issue_frame == 0


def test_text_aliased_bar_pair_does_not_count_as_new(tmp_path):
    # The fixture corpus stores bar chunks under different ids with byte-identical
    # text: inst_em:0 IS n1_doc_005:2 and grass_em:0 IS n2_doc_001:4, so this pair
    # IS the bar's first issue_frame pair. Counting it as progress overstates
    # readiness toward a detector the bar will measure.
    store = Store(tmp_path / "labels.jsonl")
    store.append(_rec("inst_em:0", "grass_em:0", Relation.CONTRADICTS,
                      "2026-07-25T00:00:00Z", relation_class="issue_frame"))
    assert readiness(store).new_issue_frame == 0


def test_story_class_is_counted(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    store.append(_rec("s1:0", "s2:0", Relation.CONTRADICTS, "2026-07-25T00:00:00Z",
                      relation_class="story"))
    rep = readiness(store)
    assert rep.new_story == 1
    assert rep.new_issue_frame == 0


def test_ready_requires_story_too(tmp_path):
    store = Store(tmp_path / "labels.jsonl")
    target = ReadinessTarget(issue_frame=1, agree=1, unrelated=1, story=1)
    store.append(_rec("a:0", "b:0", Relation.CONTRADICTS, "2026-07-25T00:00:01Z",
                      relation_class="issue_frame"))
    store.append(_rec("c:0", "d:0", Relation.CORROBORATES, "2026-07-25T00:00:02Z"))
    store.append(_rec("e:0", "f:0", Relation.UNRELATED, "2026-07-25T00:00:03Z"))
    assert readiness(store, target).ready is False   # story still 0
    store.append(_rec("g:0", "h:0", Relation.CONTRADICTS, "2026-07-25T00:00:04Z",
                      relation_class="story"))
    assert readiness(store, target).ready is True
