"""End-to-end sovereign delegation over real localhost sockets, mutual TLS.

Two uvicorn servers (node A and node B) with stubbed answer paths — no model,
no database — exercising the full wire: driver -> A (hop 0) -> B (hop 1).
The external "driver" caller authenticates as node_b (reusing its already-
pinned identity) since that's the one cert node A's CA bundle trusts.
"""
import socket
import ssl
import threading
import time

import httpx
import pytest
import uvicorn

from gin.eval.arms import ArmOutput
from gin.eval.claims import RawClaim
from gin.federation.certs import build_ca_bundle, generate_self_signed_cert
from gin.federation.client import HttpPeerClient
from gin.federation.config import NodeConfig, PeerConfig
from gin.federation.schema import FederatedQuery, FederatedResponse
from gin.federation.server import create_app


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


def _config(node_id: str, port: int, peer: PeerConfig, cert_path, key_path) -> NodeConfig:
    return NodeConfig(
        node_id=node_id, host="127.0.0.1", port=port,
        database_url=f"postgresql://x/{node_id}", cold_path=f"data/cold_{node_id}",
        model_path="", n_gpu_layers=0, n_ctx=4096,
        cert_path=str(cert_path), key_path=str(key_path),
        peer_timeout_s=10.0, peers=(peer,),
    )


def _serve(app, port: int, cert_path, key_path, ca_bundle_path) -> uvicorn.Server:
    server = uvicorn.Server(
        uvicorn.Config(
            app, host="127.0.0.1", port=port, log_level="error",
            ssl_certfile=str(cert_path), ssl_keyfile=str(key_path),
            ssl_ca_certs=str(ca_bundle_path), ssl_cert_reqs=ssl.CERT_REQUIRED,
        )
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
def two_nodes(request, tmp_path):
    """Start node B (grounded) and node A (answer_fn per-test via param).
    Yields (url_a, a_cert, b_cert, b_key): the driver authenticates as
    node_b (the one identity node A's CA bundle trusts) and trusts a_cert
    to validate node A as the server."""
    a_fn, b_fn = request.param
    port_a, port_b = _free_port(), _free_port()
    a_cert, a_key = generate_self_signed_cert("node_a", tmp_path)
    b_cert, b_key = generate_self_signed_cert("node_b", tmp_path)

    cfg_a = _config("node_a", port_a, PeerConfig("node_b", f"https://127.0.0.1:{port_b}", str(b_cert)), a_cert, a_key)
    cfg_b = _config("node_b", port_b, PeerConfig("node_a", f"https://127.0.0.1:{port_a}", str(a_cert)), b_cert, b_key)
    peer_client = HttpPeerClient(str(a_cert), str(a_key), timeout_s=10.0)
    app_a = create_app(cfg_a, answer_fn=a_fn, peer_client=peer_client)
    app_b = create_app(cfg_b, answer_fn=b_fn, peer_client=peer_client)

    a_bundle = build_ca_bundle([b_cert], tmp_path / "a_ca_bundle.pem")
    b_bundle = build_ca_bundle([a_cert], tmp_path / "b_ca_bundle.pem")
    server_a = _serve(app_a, port_a, a_cert, a_key, a_bundle)
    server_b = _serve(app_b, port_b, b_cert, b_key, b_bundle)
    yield f"https://127.0.0.1:{port_a}", a_cert, b_cert, b_key
    server_a.should_exit = True
    server_b.should_exit = True
    time.sleep(0.2)


def _ask(url: str, trust_cert_path, cert_path, key_path, hop: int = 0) -> httpx.Response:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.load_verify_locations(cafile=str(trust_cert_path))
    ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
    fq = FederatedQuery(query="q", origin_node="driver", hop_count=hop)
    with httpx.Client(verify=ctx) as client:
        return client.post(f"{url}/v1/federated/query", json=fq.model_dump(), timeout=15.0)


@pytest.mark.parametrize(
    "two_nodes", [(_refusing("retrieval_floor"), _grounded_b)], indirect=True
)
def test_delegation_crosses_the_wire(two_nodes):
    url_a, a_cert, b_cert, b_key = two_nodes
    r = _ask(url_a, a_cert, b_cert, b_key)
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
    url_a, a_cert, b_cert, b_key = two_nodes
    resp = FederatedResponse.model_validate(_ask(url_a, a_cert, b_cert, b_key).json())
    assert resp.answer.node_id == "node_a"
    assert resp.federation is None
    assert resp.refusal is None


@pytest.mark.parametrize(
    "two_nodes",
    [(_refusing("retrieval_floor"), _refusing("zero_cursors"))],
    indirect=True,
)
def test_both_refuse_aggregated_over_wire(two_nodes):
    url_a, a_cert, b_cert, b_key = two_nodes
    resp = FederatedResponse.model_validate(_ask(url_a, a_cert, b_cert, b_key).json())
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
    url_a, a_cert, b_cert, b_key = two_nodes
    resp = FederatedResponse.model_validate(_ask(url_a, a_cert, b_cert, b_key, hop=1).json())
    assert resp.refusal is not None
    assert resp.refusal.reason == "retrieval_floor"
    assert resp.answer is None


@pytest.mark.parametrize(
    "two_nodes", [(_grounded_a, _grounded_b)], indirect=True
)
def test_wrong_cert_rejected(two_nodes, tmp_path):
    """The mTLS replacement for the old wrong-secret-401 test: a caller
    presenting a cert node A never pinned never reaches routing at all —
    rejection happens at the TLS layer (httpx.RemoteProtocolError, a
    subclass of httpx.HTTPError), not as an HTTP status code."""
    url_a, a_cert, _b_cert, _b_key = two_nodes
    stranger_cert, stranger_key = generate_self_signed_cert("stranger", tmp_path)
    with pytest.raises(httpx.HTTPError):
        _ask(url_a, a_cert, stranger_cert, stranger_key)
