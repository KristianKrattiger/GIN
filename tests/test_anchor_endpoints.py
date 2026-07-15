"""Read-only anchor endpoints: root/buckets/bucket/sync_stats, auth-gated
like the query endpoint, backed by an injected local_anchor_rows callable."""
from fastapi.testclient import TestClient

from gin.eval.arms import ArmOutput
from gin.federation.anchor_tree import NUM_BUCKETS, all_bucket_hashes, bucket_index, root_hash
from gin.federation.config import NodeConfig, PeerConfig
from gin.federation.schema import (
    AnchorBucketsResponse,
    AnchorLeaf,
    AnchorLeavesResponse,
    AnchorRootResponse,
    AnchorSyncStats,
)
from gin.federation.server import create_app

CFG = NodeConfig(
    node_id="node_a", host="127.0.0.1", port=8471,
    database_url="postgresql://x/gin_node_a", cold_path="data/cold_node_a",
    model_path="", n_gpu_layers=0, n_ctx=4096,
    shared_secret="s3cret", peer_timeout_s=5.0,
    peers=(PeerConfig(node_id="node_b", url="http://peer-b"),),
)
AUTH = {"Authorization": "Bearer s3cret"}


def _grounded(q: str) -> ArmOutput:
    return ArmOutput(raw_text="x", claims=[], retrieval_manifest_hash="")


def _rows() -> list[AnchorLeaf]:
    return [
        AnchorLeaf(chunk_id="n1_doc_001:0", content_hash="h1", outlet="node_1", title="t1"),
        AnchorLeaf(chunk_id="n1_doc_002:0", content_hash="h2", outlet="node_1", title="t2"),
    ]


def test_anchors_root_matches_pure_computation():
    app = create_app(CFG, answer_fn=_grounded, local_anchor_rows=_rows)
    client = TestClient(app)
    r = client.get("/v1/federated/anchors/root", headers=AUTH)
    resp = AnchorRootResponse.model_validate(r.json())
    assert resp.root_hash == root_hash(all_bucket_hashes(_rows()))
    assert resp.leaf_count == 2


def test_anchors_root_requires_auth():
    app = create_app(CFG, answer_fn=_grounded, local_anchor_rows=_rows)
    client = TestClient(app)
    r = client.get("/v1/federated/anchors/root")
    assert r.status_code == 401


def test_anchors_buckets_has_16_entries():
    app = create_app(CFG, answer_fn=_grounded, local_anchor_rows=_rows)
    client = TestClient(app)
    r = client.get("/v1/federated/anchors/buckets", headers=AUTH)
    resp = AnchorBucketsResponse.model_validate(r.json())
    assert len(resp.bucket_hashes) == NUM_BUCKETS


def test_anchors_bucket_returns_only_that_bucket():
    rows = _rows()
    app = create_app(CFG, answer_fn=_grounded, local_anchor_rows=lambda: rows)
    client = TestClient(app)
    idx = bucket_index(rows[0].chunk_id)
    r = client.get(f"/v1/federated/anchors/bucket/{idx}", headers=AUTH)
    resp = AnchorLeavesResponse.model_validate(r.json())
    assert rows[0].chunk_id in {leaf.chunk_id for leaf in resp.leaves}
    assert resp.bucket_index == idx


def test_anchors_default_empty_when_not_configured():
    app = create_app(CFG, answer_fn=_grounded)  # no local_anchor_rows injected
    client = TestClient(app)
    r = client.get("/v1/federated/anchors/root", headers=AUTH)
    resp = AnchorRootResponse.model_validate(r.json())
    assert resp.leaf_count == 0


def test_sync_stats_defaults_before_any_cycle():
    app = create_app(CFG, answer_fn=_grounded, local_anchor_rows=_rows)
    client = TestClient(app)
    r = client.get("/v1/federated/anchors/sync_stats", headers=AUTH)
    resp = AnchorSyncStats.model_validate(r.json())
    assert resp.node_id == "node_a"
    assert resp.peer_node_id == "node_b"
    assert resp.cycles_run == 0


import asyncio

from gin.federation.anchor_store import InMemoryPeerAnchorStore
from gin.federation.anchor_tree import all_bucket_hashes, root_hash
from gin.federation.schema import AnchorBucketsResponse


class StubPeerClient:
    """Serves one fixed peer row set over the anchor GET methods only."""

    def __init__(self, rows):
        self.rows = rows

    def get_anchor_root(self, peer):
        return AnchorRootResponse(
            node_id=peer.node_id, root_hash=root_hash(all_bucket_hashes(self.rows)),
            leaf_count=len(self.rows),
        )

    def get_anchor_buckets(self, peer):
        return AnchorBucketsResponse(node_id=peer.node_id, bucket_hashes=all_bucket_hashes(self.rows))

    def get_anchor_bucket(self, peer, index):
        from gin.federation.anchor_tree import bucket_index as _bi
        matches = [r for r in self.rows if _bi(r.chunk_id) == index]
        return AnchorLeavesResponse(node_id=peer.node_id, bucket_index=index, leaves=matches)


def test_lifespan_starts_and_stops_background_sync():
    peer_rows = [AnchorLeaf(chunk_id="p:0", content_hash="h", outlet="o", title="t")]
    store = InMemoryPeerAnchorStore()
    app = create_app(
        CFG, answer_fn=_grounded, peer_client=StubPeerClient(peer_rows),
        peer_anchor_store=store,
    )

    async def _run():
        async with app.router.lifespan_context(app):
            await asyncio.sleep(0.2)

    asyncio.run(_run())
    assert {r.chunk_id for r in store.all_rows("node_b")} == {"p:0"}


def test_no_background_task_without_peer_anchor_store():
    # Existing callers (no peer_anchor_store) must see no background activity.
    app = create_app(CFG, answer_fn=_grounded, local_anchor_rows=_rows)

    async def _run():
        async with app.router.lifespan_context(app):
            await asyncio.sleep(0.1)

    asyncio.run(_run())  # must not raise, must not hang
