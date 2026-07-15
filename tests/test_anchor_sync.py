"""sync_once(): root-match short-circuit, bucket-level drill-down, and the
bandwidth property — matched buckets are never fetched."""
from gin.federation.anchor_store import InMemoryPeerAnchorStore
from gin.federation.anchor_sync import sync_once
from gin.federation.anchor_tree import all_bucket_hashes, bucket_index, root_hash
from gin.federation.config import PeerConfig
from gin.federation.schema import (
    AnchorBucketsResponse,
    AnchorLeaf,
    AnchorLeavesResponse,
    AnchorRootResponse,
)

PEER = PeerConfig(node_id="node_b", url="http://peer-b")


def _leaf(chunk_id: str, content_hash: str = "h") -> AnchorLeaf:
    return AnchorLeaf(chunk_id=chunk_id, content_hash=content_hash, outlet="o", title="t")


def _corpus(n: int) -> list[AnchorLeaf]:
    return [_leaf(f"doc_{i}:0", content_hash=f"h{i}") for i in range(n)]


class FakePeerClient:
    """Serves a fixed peer corpus over the anchor endpoints; counts bucket fetches."""

    def __init__(self, rows: list[AnchorLeaf]) -> None:
        self.rows = rows
        self.bucket_fetch_calls: list[int] = []

    def get_anchor_root(self, peer):
        return AnchorRootResponse(
            node_id=peer.node_id, root_hash=root_hash(all_bucket_hashes(self.rows)),
            leaf_count=len(self.rows),
        )

    def get_anchor_buckets(self, peer):
        return AnchorBucketsResponse(node_id=peer.node_id, bucket_hashes=all_bucket_hashes(self.rows))

    def get_anchor_bucket(self, peer, index):
        self.bucket_fetch_calls.append(index)
        buckets: dict[int, list[AnchorLeaf]] = {}
        for row in self.rows:
            buckets.setdefault(bucket_index(row.chunk_id), []).append(row)
        return AnchorLeavesResponse(node_id=peer.node_id, bucket_index=index, leaves=buckets.get(index, []))


def test_first_sync_bootstraps_full_cache():
    rows = _corpus(40)
    client = FakePeerClient(rows)
    store = InMemoryPeerAnchorStore()
    stats = sync_once(PEER, client, store)
    assert stats.root_matched is False
    assert {r.chunk_id for r in store.all_rows("node_b")} == {r.chunk_id for r in rows}
    assert stats.buckets_synced == len(set(client.bucket_fetch_calls))


def test_no_op_sync_after_convergence_fetches_no_buckets():
    rows = _corpus(40)
    client = FakePeerClient(rows)
    store = InMemoryPeerAnchorStore()
    sync_once(PEER, client, store)  # bootstrap
    client.bucket_fetch_calls.clear()
    stats = sync_once(PEER, client, store)
    assert stats.root_matched is True
    assert stats.buckets_synced == 0
    assert client.bucket_fetch_calls == []


def test_single_chunk_change_syncs_exactly_one_bucket():
    rows = _corpus(40)
    client = FakePeerClient(rows)
    store = InMemoryPeerAnchorStore()
    sync_once(PEER, client, store)  # bootstrap
    client.bucket_fetch_calls.clear()

    client.rows = list(rows)
    client.rows[5] = _leaf(client.rows[5].chunk_id, content_hash="CHANGED")
    stats = sync_once(PEER, client, store)

    assert stats.root_matched is False
    assert stats.buckets_synced == 1
    assert len(set(client.bucket_fetch_calls)) == 1
    changed_row = next(
        r for r in store.all_rows("node_b") if r.chunk_id == client.rows[5].chunk_id
    )
    assert changed_row.content_hash == "CHANGED"


def test_no_op_cycle_transfers_far_fewer_bytes_than_bootstrap():
    rows = _corpus(200)
    client = FakePeerClient(rows)
    store = InMemoryPeerAnchorStore()
    bootstrap_stats = sync_once(PEER, client, store)
    noop_stats = sync_once(PEER, client, store)
    assert noop_stats.bytes_transferred < bootstrap_stats.bytes_transferred / 10
