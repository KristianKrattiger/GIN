"""Tests for gin.assay_proposer.app — contract shape, model pinning, validation. No model needed."""
from fastapi.testclient import TestClient

from gin.assay_proposer.app import create_app

PROPOSAL = {
    "type": "unsupported", "topic": "t", "statement": "s",
    "from": {"docId": "vendor", "quote": "q"}, "to": None,
    "rationale": "r", "confidence": 0.5,
}
BODY = {
    "model": "sear/test",
    "system": "S",
    "messages": [{"role": "user", "content": "U"}],
    "excerpts": [{"docId": "vendor", "role": "claimant", "label": "V", "text": "q"}],
}


def test_health_names_the_model():
    client = TestClient(create_app(lambda req: [], "sear/test"))
    assert client.get("/v1/health").json() == {"model": "sear/test"}


def test_propose_returns_the_contract_shape_and_passes_the_request_through():
    seen = []

    def propose_fn(req):
        seen.append(req)
        return [PROPOSAL]

    client = TestClient(create_app(propose_fn, "sear/test"))
    r = client.post("/v1/propose", json=BODY)
    assert r.status_code == 200
    assert r.json() == {"model": "sear/test", "proposals": [PROPOSAL], "stop_reason": "end_turn"}
    assert seen[0].excerpts[0].docId == "vendor"
    assert seen[0].messages[0].content == "U"


def test_a_request_for_another_model_is_refused_before_decoding():
    called = []
    client = TestClient(create_app(lambda req: called.append(req) or [], "sear/test"))
    r = client.post("/v1/propose", json={**BODY, "model": "sear/other"})
    assert r.status_code == 409
    assert called == []


def test_a_request_without_excerpts_is_rejected():
    client = TestClient(create_app(lambda req: [], "sear/test"))
    body = {k: v for k, v in BODY.items() if k != "excerpts"}
    assert client.post("/v1/propose", json=body).status_code == 422
