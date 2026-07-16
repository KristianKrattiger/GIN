# tests/test_federation_server.py
"""Server guards: version, hop limit; local-only for hop>=1."""
from fastapi.testclient import TestClient

from gin.eval.arms import ArmOutput
from gin.eval.claims import RawClaim
from gin.federation.config import NodeConfig, PeerConfig
from gin.federation.schema import FederatedQuery, FederatedResponse
from gin.federation.server import create_app

CFG = NodeConfig(
    node_id="node_b", host="127.0.0.1", port=8472,
    database_url="postgresql://x/gin_node_b", cold_path="data/cold_node_b",
    model_path="", n_gpu_layers=0, n_ctx=4096,
    cert_path="b_cert.pem", key_path="b_key.pem", peer_timeout_s=5.0,
    peers=(PeerConfig(node_id="node_a", url="http://peer-a"),),
)


def _grounded(q: str) -> ArmOutput:
    return ArmOutput(
        raw_text="grounded answer",
        claims=[RawClaim(text="grounded answer", span_type="EXACT",
                         cited_chunk_ids=["n2_doc_002:3"])],
        retrieval_manifest_hash="h",
        synthesis_mode="convergent",
    )


def _refusing(q: str) -> ArmOutput:
    return ArmOutput(
        raw_text="[REFUSAL]", claims=[], retrieval_manifest_hash="",
        refused=True, refusal_reason="zero_cursors",
    )


class ExplodingPeer:
    """Peer client that must never be consulted."""

    def query(self, peer, fq):  # pragma: no cover - failure is the assert
        raise AssertionError("peer consulted on a hop>=1 request")


def _post(client, payload):
    return client.post("/v1/federated/query", json=payload)


def _fq(hop: int) -> dict:
    return FederatedQuery(
        query="q", origin_node="node_a", hop_count=hop
    ).model_dump()


def test_version_mismatch_refused():
    app = create_app(CFG, answer_fn=_grounded, peer_client=ExplodingPeer())
    client = TestClient(app)
    payload = _fq(1)
    payload["protocol_version"] = 99
    r = _post(client, payload)
    assert r.status_code == 200
    resp = FederatedResponse.model_validate(r.json())
    assert resp.refusal.reason == "version_mismatch"


def test_hop_over_limit_refused():
    app = create_app(CFG, answer_fn=_grounded, peer_client=ExplodingPeer())
    client = TestClient(app)
    r = _post(client, _fq(2))
    resp = FederatedResponse.model_validate(r.json())
    assert resp.refusal.reason == "hop_limit"


def test_hop_one_answers_locally_with_fingerprint():
    app = create_app(
        CFG, answer_fn=_grounded, peer_client=ExplodingPeer(),
        corpus_fingerprint={"n_chunks": 46},
    )
    client = TestClient(app)
    r = _post(client, _fq(1))
    resp = FederatedResponse.model_validate(r.json())
    assert resp.answer.node_id == "node_b"
    assert resp.answer.corpus_fingerprint == {"n_chunks": 46}
    assert resp.answer.claims[0].cited_chunk_ids == ["n2_doc_002:3"]
    assert resp.federation is None


def test_hop_one_refusal_never_redelegates():
    app = create_app(CFG, answer_fn=_refusing, peer_client=ExplodingPeer())
    client = TestClient(app)
    r = _post(client, _fq(1))
    resp = FederatedResponse.model_validate(r.json())
    assert resp.refusal.reason == "zero_cursors"
    assert resp.refusal.node_id == "node_b"


def test_hop_zero_local_success_no_federation_layer():
    app = create_app(CFG, answer_fn=_grounded, peer_client=ExplodingPeer())
    client = TestClient(app)
    r = _post(client, _fq(0))
    resp = FederatedResponse.model_validate(r.json())
    assert resp.answer.node_id == "node_b"
    assert resp.federation is None


def test_relayed_answer_keeps_peer_empty_fingerprint():
    from gin.federation.schema import FederatedAnswer, WireClaim

    class FingerprintlessPeer:
        def query(self, peer, fq):
            return FederatedAnswer(
                request_id=fq.request_id, node_id="node_a_peer",
                answer_text="peer answer",
                claims=[WireClaim(text="peer answer", span_type="EXACT",
                                  cited_chunk_ids=["n2_doc_002:3"])],
                corpus_fingerprint={},
                synthesis_mode="convergent",
            )

    app = create_app(
        CFG, answer_fn=_refusing, peer_client=FingerprintlessPeer(),
        corpus_fingerprint={"chunk_count": 999},
    )
    client = TestClient(app)
    r = _post(client, _fq(0))
    resp = FederatedResponse.model_validate(r.json())
    assert resp.answer is not None
    assert resp.federation is not None
    assert resp.answer.corpus_fingerprint == {}
