"""N>2 fix: create_app's lifespan must run one background sync loop PER
configured peer, not just peers[0]. Proven end-to-end through the app's own
lifespan (TestClient context-manager form runs startup/shutdown), with node A
configured with two peers (node_b, node_c) and a fake PeerClient serving a
distinct, non-empty corpus + routing summary per peer so every cycle mismatches
and triggers a summary fetch for both.

Regression guard: with the shipped single-peer (N=2) config, only one loop
should run and it must still update the endpoint's shared ``sync_stats``
object (unchanged /sync_stats contract).
"""
from __future__ import annotations

import time

from fastapi.testclient import TestClient

from gin.eval.arms import ArmOutput
from gin.federation.anchor_store import InMemoryPeerAnchorStore
from gin.federation.anchor_tree import all_bucket_hashes, bucket_index, root_hash
from gin.federation.config import NodeConfig, PeerConfig
from gin.federation.peer_summary_store import InMemoryPeerSummaryStore
from gin.federation.schema import (
    AnchorBucketsResponse,
    AnchorLeaf,
    AnchorLeavesResponse,
    AnchorRootResponse,
    PeerSummaryResponse,
)
from gin.federation.server import create_app

SECRET = "multi-peer-secret"


def _leaf(chunk_id: str, content_hash: str = "h") -> AnchorLeaf:
    return AnchorLeaf(chunk_id=chunk_id, content_hash=content_hash, outlet="o", title="t")


def _corpus(prefix: str, n: int) -> list[AnchorLeaf]:
    return [_leaf(f"{prefix}_doc_{i}:0", content_hash=f"{prefix}h{i}") for i in range(n)]


class FakeMultiPeerClient:
    """Serves a distinct fixed corpus + summary per peer, keyed by node_id —
    mirrors tests/test_anchor_sync.py's FakePeerClient, generalized to N peers
    so root/summary state never collides across peers."""

    def __init__(self, rows_by_peer: dict[str, list[AnchorLeaf]],
                 summary_by_peer: dict[str, PeerSummaryResponse]) -> None:
        self.rows_by_peer = rows_by_peer
        self.summary_by_peer = summary_by_peer

    def get_summary(self, peer):
        return self.summary_by_peer.get(peer.node_id)

    def get_anchor_root(self, peer):
        rows = self.rows_by_peer[peer.node_id]
        return AnchorRootResponse(
            node_id=peer.node_id, root_hash=root_hash(all_bucket_hashes(rows)),
            leaf_count=len(rows),
        )

    def get_anchor_buckets(self, peer):
        rows = self.rows_by_peer[peer.node_id]
        return AnchorBucketsResponse(node_id=peer.node_id, bucket_hashes=all_bucket_hashes(rows))

    def get_anchor_bucket(self, peer, index):
        rows = self.rows_by_peer[peer.node_id]
        buckets: dict[int, list[AnchorLeaf]] = {}
        for row in rows:
            buckets.setdefault(bucket_index(row.chunk_id), []).append(row)
        return AnchorLeavesResponse(
            node_id=peer.node_id, bucket_index=index, leaves=buckets.get(index, [])
        )


def _grounded(q: str) -> ArmOutput:
    return ArmOutput(raw_text="x", claims=[], retrieval_manifest_hash="")


def _config(peers: tuple[PeerConfig, ...]) -> NodeConfig:
    return NodeConfig(
        node_id="node_a", host="127.0.0.1", port=8471,
        database_url="postgresql://x/node_a", cold_path="data/cold_node_a",
        model_path="", n_gpu_layers=0, n_ctx=4096,
        shared_secret=SECRET, peer_timeout_s=5.0, peers=peers,
        anchor_sync_interval_s=0.02,
    )


def test_all_configured_peers_get_synced_not_just_the_first():
    peer_b = PeerConfig(node_id="node_b", url="http://peer-b")
    peer_c = PeerConfig(node_id="node_c", url="http://peer-c")
    client = FakeMultiPeerClient(
        rows_by_peer={
            "node_b": _corpus("b", 20),
            "node_c": _corpus("c", 20),
        },
        summary_by_peer={
            "node_b": PeerSummaryResponse(
                node_id="node_b", embedding_centroid=[1.0, 0.0],
                distinctive_terms={"b_term": 1.0},
            ),
            "node_c": PeerSummaryResponse(
                node_id="node_c", embedding_centroid=[0.0, 1.0],
                distinctive_terms={"c_term": 1.0},
            ),
        },
    )
    anchor_store = InMemoryPeerAnchorStore()
    summary_store = InMemoryPeerSummaryStore()
    config = _config((peer_b, peer_c))

    app = create_app(
        config, answer_fn=_grounded, peer_client=client,
        peer_anchor_store=anchor_store, peer_summary_store=summary_store,
    )

    with TestClient(app) as _:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if (
                summary_store.get("node_b") is not None
                and summary_store.get("node_c") is not None
            ):
                break
            time.sleep(0.05)

    got_b = summary_store.get("node_b")
    got_c = summary_store.get("node_c")
    assert got_b is not None, "node_b summary never cached"
    assert got_c is not None, "node_c summary never cached (peers[1:] not synced)"
    assert got_b.node_id == "node_b"
    assert got_c.node_id == "node_c"
    assert got_b.distinctive_terms == {"b_term": 1.0}
    assert got_c.distinctive_terms == {"c_term": 1.0}
