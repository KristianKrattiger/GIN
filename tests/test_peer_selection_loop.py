"""Three uvicorn nodes over real sockets, no model/DB: node A ranks B vs. C
from injected summaries and delegates to the right one on the first try."""
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
from gin.federation.peer_summary_store import InMemoryPeerSummaryStore
from gin.federation.schema import FederatedQuery, FederatedResponse, PeerSummaryResponse
from gin.federation.server import create_app

SECRET = "sel-secret"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(app, port):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
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


def _cfg(node_id, port, peers):
    return NodeConfig(
        node_id=node_id, host="127.0.0.1", port=port,
        database_url=f"postgresql://x/{node_id}", cold_path=f"data/cold_{node_id}",
        model_path="", n_gpu_layers=0, n_ctx=4096,
        shared_secret=SECRET, peer_timeout_s=10.0, peers=peers,
    )


@pytest.fixture
def three_nodes():
    pa, pb, pc = _free_port(), _free_port(), _free_port()
    peer_client = HttpPeerClient(SECRET, timeout_s=10.0)
    # B answers only "justice"-ish; C answers only "inflation"-ish — via stubs,
    # selection is driven purely by the injected summaries + query embedding.
    cfg_a = _cfg("node_a", pa, (PeerConfig("node_b", f"http://127.0.0.1:{pb}"),
                                PeerConfig("node_c", f"http://127.0.0.1:{pc}")))
    cfg_b = _cfg("node_b", pb, (PeerConfig("node_a", f"http://127.0.0.1:{pa}"),))
    cfg_c = _cfg("node_c", pc, (PeerConfig("node_a", f"http://127.0.0.1:{pa}"),))

    summary_store = InMemoryPeerSummaryStore()
    summary_store.set("node_b", PeerSummaryResponse(
        node_id="node_b", embedding_centroid=[0.0, 1.0], distinctive_terms={"justice": 3.0}))
    summary_store.set("node_c", PeerSummaryResponse(
        node_id="node_c", embedding_centroid=[1.0, 0.0], distinctive_terms={"inflation": 3.0}))

    # Query embedder: "inflation" -> near C's centroid; else near B's.
    def embed(q):
        return [1.0, 0.0] if "inflation" in q else [0.0, 1.0]

    app_a = create_app(cfg_a, answer_fn=_refuse, peer_client=peer_client,
                       peer_summary_store=summary_store, embed_query_fn=embed)
    app_b = create_app(cfg_b, answer_fn=_grounded("node_b"), peer_client=peer_client)
    app_c = create_app(cfg_c, answer_fn=_grounded("node_c"), peer_client=peer_client)
    sa, sb, sc = _serve(app_a, pa), _serve(app_b, pb), _serve(app_c, pc)
    yield f"http://127.0.0.1:{pa}"
    sa.should_exit = sb.should_exit = sc.should_exit = True
    time.sleep(0.2)


def _ask(url, query):
    fq = FederatedQuery(query=query, origin_node="driver", hop_count=0)
    r = httpx.post(f"{url}/v1/federated/query",
                   headers={"Authorization": f"Bearer {SECRET}"},
                   json=fq.model_dump(), timeout=15.0)
    return FederatedResponse.model_validate(r.json())


def test_selects_c_for_inflation_query_first_try(three_nodes):
    resp = _ask(three_nodes, "what drives inflation")
    assert resp.answer.node_id == "node_c"
    assert resp.federation.peers_attempted == ["node_c"]  # correct peer, first try


def test_selects_b_for_justice_query_first_try(three_nodes):
    resp = _ask(three_nodes, "environmental justice movements")
    assert resp.answer.node_id == "node_b"
    assert resp.federation.peers_attempted == ["node_b"]
