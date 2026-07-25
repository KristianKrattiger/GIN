"""Dataset assembly: fold -> schema -> bar exclusion -> text resolution."""
import pytest

from gin.cartographer.models import Relation
from gin.curator.models import LabelRecord
from gin.curator.store import Store
from gin.frames.dataset import (
    DEFAULT_LABELS,
    build_dataset,
    default_text_index,
    news_corpus_chunks,
)
from gin.frames.labels import FrameClass


def _rec(src, dst, relation, ts, relation_class=None):
    return LabelRecord(
        id=f"{src}|{dst}", src_chunk_id=src, dst_chunk_id=dst, relation=relation,
        relation_class=relation_class, rationale="", curator="t", ts=ts,
    )


def _text(*ids):
    return {i: f"text of {i}" for i in ids}


# build_dataset hard-errors on an empty class, so filter tests need a base of
# all four classes; the pair under test is added on top and its drop asserted.
_BASE = [
    ("base_a:0", "base_b:0", Relation.CONTRADICTS, "issue_frame"),
    ("base_c:0", "base_d:0", Relation.CORROBORATES, None),
    ("base_e:0", "base_f:0", Relation.RELATED_UNTYPED, None),
    ("base_g:0", "base_h:0", Relation.UNRELATED, None),
]


def _store_with_base(tmp_path, *extra):
    """Store holding one pair of every class, plus any extra rows."""
    store = Store(tmp_path / "l.jsonl")
    for i, (s, d, rel, cls) in enumerate(list(_BASE) + list(extra)):
        store.append(_rec(s, d, rel, f"2026-01-01T00:00:{i:02d}Z", relation_class=cls))
    return store


def _base_text():
    ids = [x for row in _BASE for x in row[:2]]
    return _text(*ids)


def test_news_corpus_supplies_21_chunks():
    assert len(news_corpus_chunks()) == 21


def test_default_index_resolves_every_bar_chunk():
    # Without the news YAML, 10 bar chunks resolve only via Postgres.
    from gin.frames.labels import bar_chunk_ids

    index = default_text_index()
    assert not (bar_chunk_ids() - set(index))


def test_real_label_log_yields_expected_counts():
    # Regression guard: if the label log drifts, this names the drift rather
    # than silently retraining on different data.
    report = build_dataset(Store(DEFAULT_LABELS))
    assert len(report.examples) == 49
    assert report.counts == {
        "DIVERGENT": 24, "AGREE": 9, "RELATED_UNTYPED": 10, "UNRELATED": 6,
    }
    assert report.drops == {"schema": 11, "bar_chunk": 11, "bar_text_alias": 31}


def test_bar_chunk_pair_is_dropped_and_counted(tmp_path):
    # n1_doc_005:1 is a real escalation-bar chunk reached via residue labeling.
    store = _store_with_base(
        tmp_path, ("n1_doc_005:1", "free_chunk:0", Relation.CORROBORATES, None)
    )
    text = _base_text() | _text("n1_doc_005:1", "free_chunk:0")
    report = build_dataset(store, text_index=text)
    assert report.drops["bar_chunk"] == 1
    assert len(report.examples) == 4
    assert all("n1_doc_005:1" not in (e.src_chunk_id, e.dst_chunk_id) for e in report.examples)


def test_unresolvable_text_is_dropped_and_counted(tmp_path):
    store = _store_with_base(tmp_path, ("solo:0", "ghost:0", Relation.UNRELATED, None))
    report = build_dataset(store, text_index=_base_text() | _text("solo:0"))
    assert report.drops["text_unresolved"] == 1
    assert len(report.examples) == 4


def test_story_contradicts_dropped_on_schema(tmp_path):
    store = _store_with_base(tmp_path, ("s1:0", "s2:0", Relation.CONTRADICTS, "story"))
    report = build_dataset(store, text_index=_base_text() | _text("s1:0", "s2:0"))
    assert report.drops["schema"] == 1
    assert report.counts["DIVERGENT"] == 1  # only the base issue_frame pair


def test_empty_class_is_a_hard_error(tmp_path):
    store = Store(tmp_path / "l.jsonl")
    store.append(_rec("a:0", "b:0", Relation.CORROBORATES, "2026-01-01T00:00:00Z"))
    with pytest.raises(ValueError, match="empty after filtering"):
        build_dataset(store, text_index=_text("a:0", "b:0"))


def test_examples_are_sorted_for_deterministic_folds(tmp_path):
    store = Store(tmp_path / "l.jsonl")
    pairs = [("z:0", "y:0", Relation.CORROBORATES), ("a:0", "b:0", Relation.UNRELATED),
             ("m:0", "n:0", Relation.RELATED_UNTYPED), ("c:0", "d:0", Relation.CONTRADICTS)]
    for i, (s, d, rel) in enumerate(pairs):
        store.append(_rec(s, d, rel, f"2026-01-01T00:00:0{i}Z",
                          relation_class="issue_frame" if rel is Relation.CONTRADICTS else None))
    ids = [e.src_chunk_id for e in
           build_dataset(store, text_index=_text(*[x for p in pairs for x in p[:2]])).examples]
    assert ids == sorted(ids)


def test_label_and_text_are_carried_through(tmp_path):
    store = Store(tmp_path / "l.jsonl")
    for s, d, rel, cls in [("a:0", "b:0", Relation.CONTRADICTS, "issue_frame"),
                           ("c:0", "d:0", Relation.CORROBORATES, None),
                           ("e:0", "f:0", Relation.RELATED_UNTYPED, None),
                           ("g:0", "h:0", Relation.UNRELATED, None)]:
        store.append(_rec(s, d, rel, "2026-01-01T00:00:00Z", relation_class=cls))
    report = build_dataset(store, text_index=_text(*[f"{c}:0" for c in "abcdefgh"]))
    first = report.examples[0]
    assert first.src_chunk_id == "a:0"
    assert first.src_text == "text of a:0"
    assert first.label is FrameClass.DIVERGENT
    assert report.counts["AGREE"] == 1


def test_bar_text_alias_is_dropped_even_when_chunk_id_differs(tmp_path):
    # The fixture corpus aliases bar chunks under different ids with
    # byte-identical text: inst_em:0 IS n1_doc_005:2. Excluding by chunk id
    # alone let 3 of the bar's 4 issue_frame pairs into training verbatim.
    from gin.frames.labels import bar_chunk_ids

    bar_chunk = sorted(bar_chunk_ids())[0]
    bar_text = default_text_index()[bar_chunk]
    store = _store_with_base(tmp_path, ("alias:0", "other:0", Relation.CORROBORATES, None))
    text = _base_text() | {"alias:0": bar_text, "other:0": "unrelated text"}
    report = build_dataset(store, text_index=text)
    assert report.drops["bar_text_alias"] == 1
    assert all(e.src_chunk_id != "alias:0" for e in report.examples)
