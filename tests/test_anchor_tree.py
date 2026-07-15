"""Merkle bucket-tree logic: deterministic, order-independent, and a single
mutation perturbs exactly one of 16 buckets — the property that motivates
bucketing over a plain sorted-array tree."""
import pytest

from gin.federation.anchor_tree import (
    NUM_BUCKETS,
    all_bucket_hashes,
    bucket_index,
    build_buckets,
    diff_leaves,
    root_hash,
)
from gin.federation.schema import AnchorLeaf


def _leaf(chunk_id: str, content_hash: str = "h", outlet: str = "o", title: str = "t") -> AnchorLeaf:
    return AnchorLeaf(chunk_id=chunk_id, content_hash=content_hash, outlet=outlet, title=title)


def _corpus(n: int) -> list[AnchorLeaf]:
    return [_leaf(f"doc_{i}:{i % 3}", content_hash=f"h{i}") for i in range(n)]


def test_bucket_assignment_in_range():
    for i in range(50):
        assert 0 <= bucket_index(f"doc_{i}:0") < NUM_BUCKETS


def test_all_bucket_hashes_deterministic_and_order_independent():
    rows = _corpus(30)
    h1 = all_bucket_hashes(rows)
    h2 = all_bucket_hashes(list(reversed(rows)))
    assert h1 == h2
    assert len(h1) == NUM_BUCKETS


def test_root_hash_requires_exactly_num_buckets():
    with pytest.raises(ValueError):
        root_hash(["a", "b"])


def test_single_content_change_perturbs_exactly_one_bucket():
    rows = _corpus(40)
    before = all_bucket_hashes(rows)
    mutated = list(rows)
    mutated[5] = _leaf(mutated[5].chunk_id, content_hash="CHANGED")
    after = all_bucket_hashes(mutated)
    changed_indices = [i for i in range(NUM_BUCKETS) if before[i] != after[i]]
    assert len(changed_indices) == 1
    assert root_hash(before) != root_hash(after)


def test_single_insert_perturbs_exactly_one_bucket():
    rows = _corpus(40)
    before = all_bucket_hashes(rows)
    inserted = rows + [_leaf("brand_new_doc:0", content_hash="new")]
    after = all_bucket_hashes(inserted)
    changed_indices = [i for i in range(NUM_BUCKETS) if before[i] != after[i]]
    assert len(changed_indices) == 1


def test_single_delete_perturbs_exactly_one_bucket():
    rows = _corpus(40)
    before = all_bucket_hashes(rows)
    removed = rows[:10] + rows[11:]
    after = all_bucket_hashes(removed)
    changed_indices = [i for i in range(NUM_BUCKETS) if before[i] != after[i]]
    assert len(changed_indices) == 1


def test_unrelated_insert_leaves_other_buckets_contents_identical():
    rows = _corpus(40)
    before = build_buckets(rows)
    inserted = rows + [_leaf("brand_new_doc:0", content_hash="new")]
    after = build_buckets(inserted)
    new_bucket = bucket_index("brand_new_doc:0")
    for i in range(NUM_BUCKETS):
        if i == new_bucket:
            continue
        assert before[i] == after[i]


def test_diff_leaves_added_changed_removed():
    local = [_leaf("a", content_hash="1"), _leaf("b", content_hash="1")]
    remote = [_leaf("b", content_hash="2"), _leaf("c", content_hash="1")]
    diff = diff_leaves(local, remote)
    assert [r.chunk_id for r in diff.added] == ["c"]
    assert [r.chunk_id for r in diff.changed] == ["b"]
    assert diff.removed_chunk_ids == ["a"]


def test_build_buckets_sorts_within_bucket():
    rows = _corpus(10)
    buckets = build_buckets(rows)
    for bucket_rows in buckets.values():
        ids = [r.chunk_id for r in bucket_rows]
        assert ids == sorted(ids)
