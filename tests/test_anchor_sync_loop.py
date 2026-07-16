"""Real-socket anchor sync loop, mutual TLS: two uvicorn nodes, in-memory
stores, no DB, no model — the background task actually runs and converges
the cache."""
import socket
import ssl
import threading
import time

import httpx
import pytest
import uvicorn

from gin.eval.arms import ArmOutput
from gin.federation.anchor_store import InMemoryPeerAnchorStore
from gin.federation.certs import build_ca_bundle, generate_self_signed_cert
from gin.federation.client import HttpPeerClient
from gin.federation.config import NodeConfig, PeerConfig
from gin.federation.schema import AnchorLeaf
from gin.federation.server import create_app


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _leaf(chunk_id: str, content_hash: str = "h") -> AnchorLeaf:
    return AnchorLeaf(chunk_id=chunk_id, content_hash=content_hash, outlet="o", title="t")


def _config(node_id, port, peer, cert_path, key_path, interval_s=0.05):
    return NodeConfig(
        node_id=node_id, host="127.0.0.1", port=port,
        database_url=f"postgresql://x/{node_id}", cold_path=f"data/cold_{node_id}",
        model_path="", n_gpu_layers=0, n_ctx=4096,
        cert_path=str(cert_path), key_path=str(key_path),
        peer_timeout_s=10.0, peers=(peer,),
        anchor_sync_interval_s=interval_s,
    )


def _serve(app, port, cert_path, key_path, ca_bundle_path) -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="error",
        ssl_certfile=str(cert_path), ssl_keyfile=str(key_path),
        ssl_ca_certs=str(ca_bundle_path), ssl_cert_reqs=ssl.CERT_REQUIRED,
    ))
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
def two_nodes(tmp_path):
    """Yields (store_a, b_rows, url_a, a_cert, b_cert, b_key): the driver in
    test_sync_stats_endpoint_reflects_cycles authenticates as node_b (the
    one identity node A's CA bundle trusts) and trusts a_cert to validate
    node A as the server."""
    port_a, port_b = _free_port(), _free_port()
    a_cert, a_key = generate_self_signed_cert("node_a", tmp_path)
    b_cert, b_key = generate_self_signed_cert("node_b", tmp_path)
    peer_client_a = HttpPeerClient(str(a_cert), str(a_key), timeout_s=10.0)
    peer_client_b = HttpPeerClient(str(b_cert), str(b_key), timeout_s=10.0)

    b_rows = [_leaf(f"doc_{i}:0", content_hash=f"h{i}") for i in range(20)]
    cfg_a = _config("node_a", port_a, PeerConfig("node_b", f"https://127.0.0.1:{port_b}", str(b_cert)), a_cert, a_key)
    cfg_b = _config("node_b", port_b, PeerConfig("node_a", f"https://127.0.0.1:{port_a}", str(a_cert)), b_cert, b_key)
    store_a = InMemoryPeerAnchorStore()
    app_a = create_app(
        cfg_a, answer_fn=_grounded, peer_client=peer_client_a,
        local_anchor_rows=lambda: [], peer_anchor_store=store_a,
    )
    app_b = create_app(
        cfg_b, answer_fn=_grounded, peer_client=peer_client_b,
        local_anchor_rows=lambda: b_rows,
    )
    a_bundle = build_ca_bundle([b_cert], tmp_path / "a_ca_bundle.pem")
    b_bundle = build_ca_bundle([a_cert], tmp_path / "b_ca_bundle.pem")
    server_a = _serve(app_a, port_a, a_cert, a_key, a_bundle)
    server_b = _serve(app_b, port_b, b_cert, b_key, b_bundle)
    yield store_a, b_rows, f"https://127.0.0.1:{port_a}", a_cert, b_cert, b_key
    server_a.should_exit = True
    server_b.should_exit = True
    time.sleep(0.2)


def test_background_loop_converges_cache_to_peer_ground_truth(two_nodes):
    store_a, b_rows, _, _, _, _ = two_nodes
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
    _, _, url_a, a_cert, driver_cert, driver_key = two_nodes
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.load_verify_locations(cafile=str(a_cert))
    ctx.load_cert_chain(certfile=str(driver_cert), keyfile=str(driver_key))

    # Poll rather than a fixed sleep: the first cycle alone can take several
    # seconds on this machine (see note above), so a short fixed sleep is
    # unreliable — poll for the real signal (cycles_run >= 1) instead.
    deadline = time.monotonic() + 20
    cycles_run = 0
    while time.monotonic() < deadline:
        with httpx.Client(verify=ctx) as client:
            r = client.get(f"{url_a}/v1/federated/anchors/sync_stats", timeout=5.0)
        cycles_run = r.json()["cycles_run"]
        if cycles_run >= 1:
            break
        time.sleep(0.1)
    assert cycles_run >= 1
