"""Pure Merkle-tree logic over anchor metadata — no I/O, no network, no DB.

16 fixed buckets keyed by the first hex digit of sha256(chunk_id) rather than
a plain sorted-array tree: a chunk's bucket membership is stable regardless
of what else is inserted or removed elsewhere in the set, so a single change
perturbs exactly one bucket hash (and the root), not everything after it in
sort order. See docs/superpowers/specs/2026-07-14-merkle-anchor-sync-design.md
for the full rationale.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .schema import NUM_BUCKETS, AnchorLeaf

_EMPTY_BUCKET_SENTINEL = hashlib.sha256(b"empty").hexdigest()


def bucket_index(chunk_id: str) -> int:
    return int(hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()[0], 16)


def build_buckets(rows: list[AnchorLeaf]) -> dict[int, list[AnchorLeaf]]:
    buckets: dict[int, list[AnchorLeaf]] = {i: [] for i in range(NUM_BUCKETS)}
    for row in rows:
        buckets[bucket_index(row.chunk_id)].append(row)
    for bucket_rows in buckets.values():
        bucket_rows.sort(key=lambda r: r.chunk_id)
    return buckets


def bucket_hash(rows: list[AnchorLeaf]) -> str:
    if not rows:
        return _EMPTY_BUCKET_SENTINEL
    payload = "|".join(
        f"{r.chunk_id}:{r.content_hash}:{r.outlet}:{r.title}" for r in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def all_bucket_hashes(rows: list[AnchorLeaf]) -> list[str]:
    buckets = build_buckets(rows)
    return [bucket_hash(buckets[i]) for i in range(NUM_BUCKETS)]


def root_hash(bucket_hashes: list[str]) -> str:
    if len(bucket_hashes) != NUM_BUCKETS:
        raise ValueError(f"expected {NUM_BUCKETS} bucket hashes, got {len(bucket_hashes)}")
    return hashlib.sha256("|".join(bucket_hashes).encode("utf-8")).hexdigest()


@dataclass
class AnchorDiff:
    added: list[AnchorLeaf] = field(default_factory=list)
    changed: list[AnchorLeaf] = field(default_factory=list)
    removed_chunk_ids: list[str] = field(default_factory=list)


def diff_leaves(local: list[AnchorLeaf], remote: list[AnchorLeaf]) -> AnchorDiff:
    """What changes if ``local`` becomes ``remote`` — added/changed/removed by chunk_id."""
    local_by_id = {r.chunk_id: r for r in local}
    remote_by_id = {r.chunk_id: r for r in remote}
    diff = AnchorDiff()
    for chunk_id, remote_row in remote_by_id.items():
        local_row = local_by_id.get(chunk_id)
        if local_row is None:
            diff.added.append(remote_row)
        elif local_row != remote_row:
            diff.changed.append(remote_row)
    for chunk_id in local_by_id:
        if chunk_id not in remote_by_id:
            diff.removed_chunk_ids.append(chunk_id)
    return diff
