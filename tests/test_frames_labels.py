"""4-way frame schema: relation -> training class, and the bar blocklist."""
from gin.cartographer.models import Relation
from gin.frames.labels import (
    JUDGE_LABEL,
    TRAINING_CLASSES,
    FrameClass,
    bar_chunk_ids,
    frame_class_for,
)


def test_issue_frame_contradicts_is_divergent():
    assert frame_class_for(Relation.CONTRADICTS, "issue_frame") is FrameClass.DIVERGENT


def test_story_contradicts_is_excluded():
    # NLI owns propositional conflict upstream; escalation never sees these.
    assert frame_class_for(Relation.CONTRADICTS, "story") is None


def test_untyped_contradicts_is_excluded():
    assert frame_class_for(Relation.CONTRADICTS, None) is None


def test_plain_relations_map_directly():
    assert frame_class_for(Relation.CORROBORATES, None) is FrameClass.AGREE
    assert frame_class_for(Relation.RELATED_UNTYPED, None) is FrameClass.RELATED_UNTYPED
    assert frame_class_for(Relation.UNRELATED, None) is FrameClass.UNRELATED


def test_supersedes_is_not_a_training_class():
    assert frame_class_for(Relation.SUPERSEDES, None) is None


def test_judge_collapse_covers_every_training_class():
    assert set(JUDGE_LABEL) == set(TRAINING_CLASSES)
    assert set(JUDGE_LABEL.values()) == {"DIVERGENT", "AGREE", "UNRELATED"}


def test_related_untyped_collapses_to_unrelated():
    # The 4th class sharpens the DIVERGENT boundary in training; it is never emitted.
    assert JUDGE_LABEL[FrameClass.RELATED_UNTYPED] == "UNRELATED"
    assert JUDGE_LABEL[FrameClass.DIVERGENT] == "DIVERGENT"


def test_bar_has_21_distinct_chunks():
    ids = bar_chunk_ids()
    assert len(ids) == 21
    assert "n1_doc_005:1" in ids
    assert "inflation_bureau_report:0" in ids


def test_bar_chunk_ids_is_cached():
    assert bar_chunk_ids() is bar_chunk_ids()
