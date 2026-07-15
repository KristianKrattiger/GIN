"""Real-socket anchor sync loop: two uvicorn nodes, in-memory stores, no DB,
no model — the background task actually runs and converges the cache."""
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from gin.eval.arms import ArmOutput
from gin.federation.anchor_store import InMemoryPeerAnchorStore
from gin.federation.client import HttpPeerClient
from gin.federation.config import NodeConfig, PeerConfig
from gin.federation.schema import AnchorLeaf
from gin.federation.server import create_app

SECRET = "anchor-loop-secret"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _leaf(chunk_id: str, content_hash: str = "h") -> AnchorLeaf:
    return AnchorLeaf(chunk_id=chunk_id, content_hash=content_hash, outlet="o", title="t")


def _config(node_id: str, port: int, peer: PeerConfig, interval_s: float = 0.05) -> NodeConfig:
    return NodeConfig(
        node_id=node_id, host="127.0.0.1", port=port,
        database_url=f"postgresql://x/{node_id}", cold_path=f"data/cold_{node_id}",
        model_path="", n_gpu_layers=0, n_ctx=4096,
        shared_secret=SECRET, peer_timeout_s=10.0, peers=(peer,),
        anchor_sync_interval_s=interval_s,
    )


def _serve(app, port: int) -> uvicorn.Server:
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.05)
    return server


def _grounded(q: str) -> ArmOutput:
    return ArmOutput(raw_text="x", claims=[], retrieval_manifest_hash="")


@pytest.fixture
def two_nodes():
    port_a, port_b = _free_port(), _free_port()
    peer_client = HttpPeerClient(SECRET, timeout_s=10.0)
    b_rows = [_leaf(f"doc_{i}:0", content_hash=f"h{i}") for i in range(20)]
    cfg_a = _config("node_a", port_a, PeerConfig("node_b", f"http://127.0.0.1:{port_b}"))
    cfg_b = _config("node_b", port_b, PeerConfig("node_a", f"http://127.0.0.1:{port_a}"))
    store_a = InMemoryPeerAnchorStore()  # A's cache of B
    app_a = create_app(
        cfg_a, answer_fn=_grounded, peer_client=peer_client,
        local_anchor_rows=lambda: [], peer_anchor_store=store_a,
    )
    app_b = create_app(
        cfg_b, answer_fn=_grounded, peer_client=peer_client,
        local_anchor_rows=lambda: b_rows,
    )
    server_a = _serve(app_a, port_a)
    server_b = _serve(app_b, port_b)
    yield store_a, b_rows, f"http://127.0.0.1:{port_a}"
    server_a.should_exit = True
    server_b.should_exit = True
    time.sleep(0.2)


def test_background_loop_converges_cache_to_peer_ground_truth(two_nodes):
    store_a, b_rows, _ = two_nodes
    # HttpPeerClient opens a fresh httpx.Client (fresh TCP connection) per
    # request; on this machine each localhost round trip runs ~300-350ms, so
    # a first sync cycle across up to 16 mismatched buckets can take several
    # seconds. Generous deadline to absorb that without touching production
    # timing.
    deadline = time.monotonic() + 20
    expected = {r.chunk_id for r in b_rows}
    while time.monotonic() < deadline:
        if {r.chunk_id for r in store_a.all_rows("node_b")} == expected:
            break
        time.sleep(0.05)
    assert {r.chunk_id for r in store_a.all_rows("node_b")} == expected


def test_sync_stats_endpoint_reflects_cycles(two_nodes):
    _, _, url_a = two_nodes
    # Poll rather than a fixed sleep: the first cycle alone can take several
    # seconds on this machine (see note above), so a short fixed sleep is
    # unreliable — poll for the real signal (cycles_run >= 1) instead.
    deadline = time.monotonic() + 20
    cycles_run = 0
    while time.monotonic() < deadline:
        r = httpx.get(
            f"{url_a}/v1/federated/anchors/sync_stats",
            headers={"Authorization": f"Bearer {SECRET}"}, timeout=5.0,
        )
        cycles_run = r.json()["cycles_run"]
        if cycles_run >= 1:
            break
        time.sleep(0.1)
    assert cycles_run >= 1
