# tests/test_summary_endpoint.py
"""The /v1/federated/summary endpoint: injected summary callable."""
from fastapi.testclient import TestClient

from gin.eval.arms import ArmOutput
from gin.federation.config import NodeConfig, PeerConfig
from gin.federation.schema import PeerSummaryResponse
from gin.federation.server import create_app

CFG = NodeConfig(
    node_id="node_c", host="127.0.0.1", port=8473,
    database_url="postgresql://x/gin_node_c", cold_path="data/cold_node_c",
    model_path="", n_gpu_layers=0, n_ctx=4096,
    cert_path="c_cert.pem", key_path="c_key.pem", peer_timeout_s=5.0,
    peers=(PeerConfig(node_id="node_a", url="http://peer-a"),),
)


def _grounded(q: str) -> ArmOutput:
    return ArmOutput(raw_text="x", claims=[], retrieval_manifest_hash="")


def _summary() -> PeerSummaryResponse:
    return PeerSummaryResponse(
        node_id="node_c", embedding_centroid=[0.1, 0.2, 0.3],
        distinctive_terms={"inflation": 2.0},
    )


def test_summary_endpoint_returns_injected_summary():
    app = create_app(CFG, answer_fn=_grounded, local_summary=_summary)
    client = TestClient(app)
    r = client.get("/v1/federated/summary")
    resp = PeerSummaryResponse.model_validate(r.json())
    assert resp.node_id == "node_c"
    assert resp.distinctive_terms == {"inflation": 2.0}


def test_summary_endpoint_default_is_empty():
    app = create_app(CFG, answer_fn=_grounded)
    client = TestClient(app)
    r = client.get("/v1/federated/summary")
    resp = PeerSummaryResponse.model_validate(r.json())
    assert resp.node_id == "node_c"
    assert resp.embedding_centroid == []
    assert resp.distinctive_terms == {}
