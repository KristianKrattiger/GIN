"""End-to-end sovereign delegation over real localhost sockets.

Two uvicorn servers (node A and node B) with stubbed answer paths — no model,
no database — exercising the full wire: driver -> A (hop 0) -> B (hop 1).
"""
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from gin.eval.arms import ArmOutput
from gin.eval.claims import RawClaim
from gin.federation.client import HttpPeerClient
from gin.federation.config import NodeConfig, PeerConfig
from gin.federation.schema import FederatedQuery, FederatedResponse
from gin.federation.server import create_app

SECRET = "loop-test-secret"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _grounded_b(q: str) -> ArmOutput:
    return ArmOutput(
        raw_text="Indigenous-led resistance efforts",
        claims=[RawClaim(text="Indigenous-led resistance efforts",
                         span_type="EXACT", cited_chunk_ids=["n2_doc_001:4"])],
        retrieval_manifest_hash="stub-b",
        synthesis_mode="convergent",
    )


def _grounded_a(q: str) -> ArmOutput:
    return ArmOutput(
        raw_text="2023 anomaly answer",
        claims=[RawClaim(text="2023 anomaly answer", span_type="EXACT",
                         cited_chunk_ids=["n1_doc_002:1"])],
        retrieval_manifest_hash="stub-a",
        synthesis_mode="convergent",
    )


def _refusing(reason: str):
    def fn(q: str) -> ArmOutput:
        return ArmOutput(raw_text="[REFUSAL]", claims=[],
                         retrieval_manifest_hash="", refused=True,
                         refusal_reason=reason)
    return fn


def _config(node_id: str, port: int, peer: PeerConfig) -> NodeConfig:
    return NodeConfig(
        node_id=node_id, host="127.0.0.1", port=port,
        database_url=f"postgresql://x/{node_id}", cold_path=f"data/cold_{node_id}",
        model_path="", n_gpu_layers=0, n_ctx=4096,
        cert_path=f"{node_id}_cert.pem", key_path=f"{node_id}_key.pem",
        peer_timeout_s=10.0, peers=(peer,),
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


@pytest.fixture
def two_nodes(request):
    """Start node B (grounded) and node A (answer_fn per-test via param)."""
    a_fn, b_fn = request.param
    port_a, port_b = _free_port(), _free_port()
    cfg_a = _config("node_a", port_a, PeerConfig("node_b", f"http://127.0.0.1:{port_b}"))
    cfg_b = _config("node_b", port_b, PeerConfig("node_a", f"http://127.0.0.1:{port_a}"))
    peer_client = HttpPeerClient(SECRET, timeout_s=10.0)
    app_a = create_app(cfg_a, answer_fn=a_fn, peer_client=peer_client)
    app_b = create_app(cfg_b, answer_fn=b_fn, peer_client=peer_client)
    server_a = _serve(app_a, port_a)
    server_b = _serve(app_b, port_b)
    yield f"http://127.0.0.1:{port_a}", f"http://127.0.0.1:{port_b}"
    server_a.should_exit = True
    server_b.should_exit = True
    time.sleep(0.2)


def _ask(url: str, hop: int = 0, secret: str = SECRET) -> httpx.Response:
    fq = FederatedQuery(query="q", origin_node="driver", hop_count=hop)
    return httpx.post(
        f"{url}/v1/federated/query",
        headers={"Authorization": f"Bearer {secret}"},
        json=fq.model_dump(),
        timeout=15.0,
    )


@pytest.mark.parametrize(
    "two_nodes", [(_refusing("retrieval_floor"), _grounded_b)], indirect=True
)
def test_delegation_crosses_the_wire(two_nodes):
    url_a, _ = two_nodes
    r = _ask(url_a)
    assert r.status_code == 200
    resp = FederatedResponse.model_validate(r.json())
    assert resp.answer.node_id == "node_b"
    assert resp.answer.claims[0].cited_chunk_ids == ["n2_doc_001:4"]
    assert resp.federation.answered_by == "node_b"
    assert resp.federation.hop_count == 1
    assert resp.refusal is None


@pytest.mark.parametrize(
    "two_nodes", [(_grounded_a, _grounded_b)], indirect=True
)
def test_local_answer_does_not_route(two_nodes):
    url_a, _ = two_nodes
    resp = FederatedResponse.model_validate(_ask(url_a).json())
    assert resp.answer.node_id == "node_a"
    assert resp.federation is None
    assert resp.refusal is None


@pytest.mark.parametrize(
    "two_nodes",
    [(_refusing("retrieval_floor"), _refusing("zero_cursors"))],
    indirect=True,
)
def test_both_refuse_aggregated_over_wire(two_nodes):
    url_a, _ = two_nodes
    resp = FederatedResponse.model_validate(_ask(url_a).json())
    assert resp.refusal.node_id == "node_a"
    assert resp.refusal.reason == "retrieval_floor"
    assert resp.refusal.peer_reasons == {"node_b": "zero_cursors"}
    assert resp.answer is None


@pytest.mark.parametrize(
    "two_nodes", [(_refusing("retrieval_floor"), _grounded_b)], indirect=True
)
def test_hop_one_at_a_never_reaches_b(two_nodes):
    """Loop prevention over the real wire: hop-1 into refusing A must refuse,
    not bounce to grounded B."""
    url_a, _ = two_nodes
    resp = FederatedResponse.model_validate(_ask(url_a, hop=1).json())
    assert resp.refusal is not None
    assert resp.refusal.reason == "retrieval_floor"
    assert resp.answer is None


@pytest.mark.parametrize(
    "two_nodes", [(_grounded_a, _grounded_b)], indirect=True
)
def test_wrong_secret_rejected(two_nodes):
    url_a, _ = two_nodes
    r = _ask(url_a, secret="wrong")
    assert r.status_code == 401
