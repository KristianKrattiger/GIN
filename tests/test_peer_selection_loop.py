"""Three uvicorn nodes over real sockets, mutual TLS, no model/DB: node A
ranks B vs. C from injected summaries and delegates to the right one on the
first try. The external "driver" caller authenticates as node_b (one of the
two identities node A's CA bundle trusts) and trusts a_cert to validate
node A as the server."""
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
from gin.federation.peer_summary_store import InMemoryPeerSummaryStore
from gin.federation.schema import FederatedQuery, FederatedResponse, PeerSummaryResponse
from gin.federation.server import create_app


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(app, port, cert_path, key_path, ca_bundle_path):
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="error",
        ssl_certfile=str(cert_path), ssl_keyfile=str(key_path),
        ssl_ca_certs=str(ca_bundle_path), ssl_cert_reqs=ssl.CERT_REQUIRED,
    ))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.05)
    return server


def _refuse(q):
    return ArmOutput(raw_text="[REFUSAL]", claims=[], retrieval_manifest_hash="",
                     refused=True, refusal_reason="retrieval_floor")


def _grounded(node_id):
    def fn(q):
        return ArmOutput(
            raw_text=f"{node_id} answer",
            claims=[RawClaim(text=f"{node_id} answer", span_type="EXACT",
                             cited_chunk_ids=[f"{node_id}:0"])],
            retrieval_manifest_hash="h", synthesis_mode="convergent",
        )
    return fn


def _cfg(node_id, port, peers, cert_path, key_path):
    return NodeConfig(
        node_id=node_id, host="127.0.0.1", port=port,
        database_url=f"postgresql://x/{node_id}", cold_path=f"data/cold_{node_id}",
        model_path="", n_gpu_layers=0, n_ctx=4096,
        cert_path=str(cert_path), key_path=str(key_path),
        peer_timeout_s=10.0, peers=peers,
    )


@pytest.fixture
def three_nodes(tmp_path):
    pa, pb, pc = _free_port(), _free_port(), _free_port()
    a_cert, a_key = generate_self_signed_cert("node_a", tmp_path)
    b_cert, b_key = generate_self_signed_cert("node_b", tmp_path)
    c_cert, c_key = generate_self_signed_cert("node_c", tmp_path)

    peer_client_a = HttpPeerClient(str(a_cert), str(a_key), timeout_s=10.0)
    peer_client_b = HttpPeerClient(str(b_cert), str(b_key), timeout_s=10.0)
    peer_client_c = HttpPeerClient(str(c_cert), str(c_key), timeout_s=10.0)

    cfg_a = _cfg("node_a", pa, (
        PeerConfig("node_b", f"https://127.0.0.1:{pb}", str(b_cert)),
        PeerConfig("node_c", f"https://127.0.0.1:{pc}", str(c_cert)),
    ), a_cert, a_key)
    cfg_b = _cfg("node_b", pb, (PeerConfig("node_a", f"https://127.0.0.1:{pa}", str(a_cert)),), b_cert, b_key)
    cfg_c = _cfg("node_c", pc, (PeerConfig("node_a", f"https://127.0.0.1:{pa}", str(a_cert)),), c_cert, c_key)

    summary_store = InMemoryPeerSummaryStore()
    summary_store.set("node_b", PeerSummaryResponse(
        node_id="node_b", embedding_centroid=[0.0, 1.0], distinctive_terms={"justice": 3.0}))
    summary_store.set("node_c", PeerSummaryResponse(
        node_id="node_c", embedding_centroid=[1.0, 0.0], distinctive_terms={"inflation": 3.0}))

    def embed(q):
        return [1.0, 0.0] if "inflation" in q else [0.0, 1.0]

    app_a = create_app(cfg_a, answer_fn=_refuse, peer_client=peer_client_a,
                       peer_summary_store=summary_store, embed_query_fn=embed)
    app_b = create_app(cfg_b, answer_fn=_grounded("node_b"), peer_client=peer_client_b)
    app_c = create_app(cfg_c, answer_fn=_grounded("node_c"), peer_client=peer_client_c)

    a_bundle = build_ca_bundle([b_cert, c_cert], tmp_path / "a_ca_bundle.pem")
    b_bundle = build_ca_bundle([a_cert], tmp_path / "b_ca_bundle.pem")
    c_bundle = build_ca_bundle([a_cert], tmp_path / "c_ca_bundle.pem")

    sa = _serve(app_a, pa, a_cert, a_key, a_bundle)
    sb = _serve(app_b, pb, b_cert, b_key, b_bundle)
    sc = _serve(app_c, pc, c_cert, c_key, c_bundle)
    yield f"https://127.0.0.1:{pa}", a_cert, b_cert, b_key
    sa.should_exit = sb.should_exit = sc.should_exit = True
    time.sleep(0.2)


def _ask(url, query, trust_cert_path, cert_path, key_path):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.load_verify_locations(cafile=str(trust_cert_path))
    ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
    fq = FederatedQuery(query=query, origin_node="driver", hop_count=0)
    with httpx.Client(verify=ctx) as client:
        r = client.post(f"{url}/v1/federated/query", json=fq.model_dump(), timeout=15.0)
    return FederatedResponse.model_validate(r.json())


def test_selects_c_for_inflation_query_first_try(three_nodes):
    url_a, a_cert, driver_cert, driver_key = three_nodes
    resp = _ask(url_a, "what drives inflation", a_cert, driver_cert, driver_key)
    assert resp.answer.node_id == "node_c"
    assert resp.federation.peers_attempted == ["node_c"]


def test_selects_b_for_justice_query_first_try(three_nodes):
    url_a, a_cert, driver_cert, driver_key = three_nodes
    resp = _ask(url_a, "environmental justice movements", a_cert, driver_cert, driver_key)
    assert resp.answer.node_id == "node_b"
    assert resp.federation.peers_attempted == ["node_b"]
