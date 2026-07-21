"""Merging two curators' label logs and measuring agreement on shared pairs."""
from pathlib import Path

from gin.curator.models import LabelRecord
from gin.curator.store import Store
from scripts.curator_merge_check import AgreementResult, check_agreement, merge_logs


def _rec(id_, src, dst, relation, curator, ts, relation_class=None):
    return LabelRecord(
        id=id_, src_chunk_id=src, dst_chunk_id=dst, relation=relation,
        relation_class=relation_class, rationale="", curator=curator, ts=ts,
    )


def test_merge_logs_concatenates_both_files(tmp_path: Path):
    from gin.cartographer.models import Relation

    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    store_a = Store(a)
    store_b = Store(b)
    store_a.append(_rec("1", "x:0", "y:0", Relation.CONTRADICTS, "kristian", "2026-07-20T00:00:00Z"))
    store_b.append(_rec("2", "p:0", "q:0", Relation.UNRELATED, "alex", "2026-07-20T00:00:01Z"))

    out = tmp_path / "merged.jsonl"
    merge_logs([a, b], out)

    merged = Store(out).fold_current()
    assert len(merged) == 2
    assert merged[("x:0", "y:0")].curator == "kristian"
    assert merged[("p:0", "q:0")].curator == "alex"


def test_check_agreement_counts_matches_and_mismatches():
    from gin.cartographer.models import Relation

    fold_a = {
        ("x:0", "y:0"): _rec("1", "x:0", "y:0", Relation.CONTRADICTS, "kristian", "t1", "issue_frame"),
        ("p:0", "q:0"): _rec("2", "p:0", "q:0", Relation.UNRELATED, "kristian", "t2"),
    }
    fold_b = {
        ("x:0", "y:0"): _rec("3", "x:0", "y:0", Relation.CONTRADICTS, "alex", "t3", "issue_frame"),
        ("p:0", "q:0"): _rec("4", "p:0", "q:0", Relation.RELATED_UNTYPED, "alex", "t4"),
    }
    result = check_agreement(fold_a, fold_b, {("x:0", "y:0"), ("p:0", "q:0")})

    assert isinstance(result, AgreementResult)
    assert result.agree == 1
    assert result.disagree == 1
    assert len(result.disagreements) == 1
    assert result.disagreements[0][0] == ("p:0", "q:0")


def test_check_agreement_skips_keys_missing_from_either_fold():
    from gin.cartographer.models import Relation

    fold_a = {("x:0", "y:0"): _rec("1", "x:0", "y:0", Relation.CONTRADICTS, "kristian", "t1")}
    fold_b: dict = {}
    result = check_agreement(fold_a, fold_b, {("x:0", "y:0")})
    assert result.agree == 0
    assert result.disagree == 0
