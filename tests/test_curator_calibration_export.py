"""Store -> calibration rows: eval pairs excluded, drops counted."""
import pytest

from gin.cartographer.models import Relation
from gin.curator.calibration_export import export_calibration_rows
from gin.curator.models import LabelRecord
from gin.curator.store import Store


def _rec(src, dst, relation, ts, relation_class=None):
    return LabelRecord(
        id=f"{src}|{dst}", src_chunk_id=src, dst_chunk_id=dst, relation=relation,
        relation_class=relation_class, rationale="", curator="t", ts=ts,
    )


def _signals(a_text, b_text):
    """Deterministic stand-in for embed + NLI; no models."""
    return (0.5, 0.25, True)


def _text(*ids):
    return {i: f"text of {i}" for i in ids}


def test_excludes_eval_pairs(tmp_path):
    # inst_em:0 <-> clim_pledges:0 is a labeled_set gold member, so it is an
    # eval pair and must never reach calibration.
    store = Store(tmp_path / "l.jsonl")
    store.append(_rec("inst_em:0", "clim_pledges:0", Relation.CORROBORATES, "2026-01-01T00:00:00Z"))
    store.append(_rec("free_a:0", "free_b:0", Relation.UNRELATED, "2026-01-01T00:00:01Z"))
    report = export_calibration_rows(
        store, _signals, text_index=_text("inst_em:0", "clim_pledges:0", "free_a:0", "free_b:0")
    )
    assert report.drops["eval_pair"] == 1
    assert len(report.rows) == 1
    assert report.rows[0]["relation"] == "unrelated"
    # Measured, but held out — never a calibration row.
    assert len(report.eval_rows) == 1
    assert report.eval_rows[0]["src"] == "inst_em:0"


def test_supersedes_rows_are_dropped(tmp_path):
    store = Store(tmp_path / "l.jsonl")
    store.append(_rec("a:0", "b:0", Relation.SUPERSEDES, "2026-01-01T00:00:00Z"))
    store.append(_rec("c:0", "d:0", Relation.CORROBORATES, "2026-01-01T00:00:01Z"))
    report = export_calibration_rows(store, _signals, text_index=_text("a:0", "b:0", "c:0", "d:0"))
    assert report.drops["not_a_classifier_output"] == 1
    assert len(report.rows) == 1


def test_unresolvable_text_is_dropped_and_counted(tmp_path):
    store = Store(tmp_path / "l.jsonl")
    store.append(_rec("a:0", "ghost:0", Relation.UNRELATED, "2026-01-01T00:00:00Z"))
    store.append(_rec("c:0", "d:0", Relation.CORROBORATES, "2026-01-01T00:00:01Z"))
    report = export_calibration_rows(store, _signals, text_index=_text("a:0", "c:0", "d:0"))
    assert report.drops["text_unresolved"] == 1
    assert len(report.rows) == 1


def test_rows_carry_the_injected_signals(tmp_path):
    store = Store(tmp_path / "l.jsonl")
    store.append(_rec("a:0", "b:0", Relation.CONTRADICTS, "2026-01-01T00:00:00Z"))
    report = export_calibration_rows(store, _signals, text_index=_text("a:0", "b:0"))
    assert report.rows[0] == {
        "cos": 0.5, "p_contra": 0.25, "same_story": True, "relation": "contradicts",
    }


def test_empty_result_is_a_hard_error(tmp_path):
    store = Store(tmp_path / "l.jsonl")
    store.append(_rec("inst_em:0", "clim_pledges:0", Relation.CORROBORATES, "2026-01-01T00:00:00Z"))
    with pytest.raises(ValueError, match="no calibration rows"):
        export_calibration_rows(store, _signals, text_index=_text("inst_em:0", "clim_pledges:0"))


def test_real_store_yields_expected_counts():
    # Regression guard on the split. gold_edges and labeled_set name the same
    # 5 pairs under different chunk-id schemes and the store holds both copies,
    # so eval_pair_keys() has 45 id-keys but only 40 are offline-measurable.
    # The long-form copies drop as text_unresolved; their short-form twins drop
    # as eval_pair. Neither reaches calibration.
    from pathlib import Path

    from gin.frames.dataset import DEFAULT_LABELS

    report = export_calibration_rows(Store(Path(DEFAULT_LABELS)), _signals)
    assert report.drops == {
        "eval_pair": 40, "text_unresolved": 5, "not_a_classifier_output": 2,
    }
    assert len(report.rows) == 131
    assert len(report.eval_rows) == 40
