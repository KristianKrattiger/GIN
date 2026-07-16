# tests/test_trust_gate_wiring.py
"""Trust gate wired into create_app: a peer below the configured trust
threshold for a domain it serves is never contacted, even when it ranks
first on similarity. An unconfigured (empty) trust_weights config must
reproduce sub-project 3's ungated behavior exactly."""
from fastapi.testclient import TestClient

from gin.eval.arms import ArmOutput
from gin.federation.config import NodeConfig, PeerConfig
from gin.federation.peer_summary_store import InMemoryPeerSummaryStore
from gin.federation.schema import (
    FederatedAnswer,
    FederatedQuery,
    FederatedResponse,
    NodeRefusal,
    PeerSummaryResponse,
    WireClaim,
)
from gin.federation.server import create_app


def _cfg(trust_weights=None, trust_gate_threshold=0.5):
    return NodeConfig(
        node_id="node_a", host="127.0.0.1", port=8471,
        database_url="postgresql://x/a", cold_path="data/cold_a",
        model_path="", n_gpu_layers=0, n_ctx=4096,
        cert_path="a_cert.pem", key_path="a_key.pem", peer_timeout_s=5.0,
        peers=(PeerConfig("node_b", "http://b"), PeerConfig("node_c", "http://c")),
        trust_weights=trust_weights or {},
        trust_gate_threshold=trust_gate_threshold,
    )


def _refuse(q):
    return ArmOutput(raw_text="[REFUSAL]", claims=[], retrieval_manifest_hash="",
                     refused=True, refusal_reason="retrieval_floor")


class ScriptedPeer:
    """Answers for `answerer` node_id; refuses for everyone else. Records calls."""

    def __init__(self, answerer):
        self.answerer = answerer
        self.calls = []

    def query(self, peer, fq):
        self.calls.append(peer.node_id)
        if peer.node_id == self.answerer:
            return FederatedAnswer(
                request_id=fq.request_id, node_id=peer.node_id,
                answer_text="grounded", claims=[WireClaim(
                    text="grounded", span_type="EXACT", cited_chunk_ids=["c:0"])],
                corpus_fingerprint={"n": 1}, synthesis_mode="convergent",
            )
        return NodeRefusal(request_id=fq.request_id, node_id=peer.node_id, reason="zero_cursors")


def _summaries():
    store = InMemoryPeerSummaryStore()
    store.set("node_c", PeerSummaryResponse(
        node_id="node_c", embedding_centroid=[1.0, 0.0],
        distinctive_terms={"inflation": 3.0}, domains=["monetary_policy"],
    ))
    store.set("node_b", PeerSummaryResponse(
        node_id="node_b", embedding_centroid=[0.0, 1.0],
        distinctive_terms={"justice": 3.0}, domains=["environmental_impact"],
    ))
    return store


def _embed(q):
    return [1.0, 0.0] if "inflation" in q else [0.0, 1.0]


def test_gated_peer_never_contacted_falls_back_to_refusal():
    peer_client = ScriptedPeer(answerer="node_c")
    app = create_app(
        _cfg(trust_weights={"node_c": {"monetary_policy": 0.1}}),
        answer_fn=_refuse, peer_client=peer_client,
        peer_summary_store=_summaries(), embed_query_fn=_embed,
    )
    client = TestClient(app)
    fq = FederatedQuery(query="what drives inflation", origin_node="d", hop_count=0)
    r = client.post("/v1/federated/query", json=fq.model_dump())
    resp = FederatedResponse.model_validate(r.json())
    assert resp.refusal is not None
    assert peer_client.calls == ["node_b"]
    assert "node_c" not in (resp.refusal.peer_reasons or {})


def test_ungated_query_still_reaches_correct_peer():
    peer_client = ScriptedPeer(answerer="node_c")
    app = create_app(
        _cfg(),
        answer_fn=_refuse, peer_client=peer_client,
        peer_summary_store=_summaries(), embed_query_fn=_embed,
    )
    client = TestClient(app)
    fq = FederatedQuery(query="what drives inflation", origin_node="d", hop_count=0)
    r = client.post("/v1/federated/query", json=fq.model_dump())
    resp = FederatedResponse.model_validate(r.json())
    assert resp.answer.node_id == "node_c"
    assert peer_client.calls == ["node_c"]
