"""Server guards: auth, version, hop limit; local-only for hop>=1."""
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
    shared_secret="s3cret", peer_timeout_s=5.0,
    peers=(PeerConfig(node_id="node_a", url="http://peer-a"),),
)
AUTH = {"Authorization": "Bearer s3cret"}


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


def _post(client, payload, headers=AUTH):
    return client.post("/v1/federated/query", json=payload, headers=headers)


def _fq(hop: int) -> dict:
    return FederatedQuery(
        query="q", origin_node="node_a", hop_count=hop
    ).model_dump()


def test_missing_bearer_is_401():
    app = create_app(CFG, answer_fn=_grounded, peer_client=ExplodingPeer())
    client = TestClient(app)
    r = client.post("/v1/federated/query", json=_fq(1))
    assert r.status_code == 401


def test_wrong_bearer_is_401():
    app = create_app(CFG, answer_fn=_grounded, peer_client=ExplodingPeer())
    client = TestClient(app)
    r = _post(client, _fq(1), headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


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
    # ExplodingPeer proves the peer is not consulted on local success.
    resp = FederatedResponse.model_validate(r.json())
    assert resp.answer.node_id == "node_b"
    assert resp.federation is None
