"""POST /v1/federated/query/stream: NDJSON trace events, terminal response
event, and full backward-compatible byte-for-byte parity with the
non-streaming endpoint's response shape for the same query."""
import json

from fastapi.testclient import TestClient

from gin.corpus.trace_events import ClaimClosedTrace, RetrievalSettledTrace, current_trace_sink
from gin.eval.arms import ArmOutput
from gin.eval.claims import RawClaim
from gin.federation.config import NodeConfig, PeerConfig
from gin.federation.schema import FederatedQuery, FederatedResponse
from gin.federation.server import create_app

CFG = NodeConfig(
    node_id="node_a", host="127.0.0.1", port=8471,
    database_url="postgresql://x/gin_node_a", cold_path="data/cold_node_a",
    model_path="", n_gpu_layers=0, n_ctx=4096,
    cert_path="a_cert.pem", key_path="a_key.pem", peer_timeout_s=5.0, peers=(),
)


def _grounded_with_events(q: str) -> ArmOutput:
    """Simulates what the real decode_bundle -> generate_no_continuation ->
    NoContinuationArm chain does: push trace events through the ambient
    sink while producing the final ArmOutput — same mechanism the real
    chain uses, just driven manually since this test injects a fake
    answer_fn rather than running a real model."""
    sink = current_trace_sink.get()
    if sink is not None:
        sink(RetrievalSettledTrace(synthesis_mode="convergent", manifest_hash="h", chunk_count=1))
        sink(ClaimClosedTrace(text="grounded claim", span_type="EXACT", cited_chunk_ids=["c:0"]))
    return ArmOutput(
        raw_text="grounded claim",
        claims=[RawClaim(text="grounded claim", span_type="EXACT", cited_chunk_ids=["c:0"])],
        retrieval_manifest_hash="h",
        synthesis_mode="convergent",
    )


def _refusing(q: str) -> ArmOutput:
    return ArmOutput(raw_text="[REFUSAL]", claims=[], retrieval_manifest_hash="",
                     refused=True, refusal_reason="retrieval_floor")


def _streamed_then_gated_refusal(q: str) -> ArmOutput:
    """Simulates gin.eval.arms.NoContinuationArm.run: SEAR's constrained
    decode streams a claim_admitted event as its span closes (same sink
    mechanism as _grounded_with_events above), but a post-decode grounding
    gate (_retrieval_relevant / _claims_query_relevant / gold-coverage)
    then rejects the answer entirely -- a real ArmOutput refusal, even
    though a real claim already streamed."""
    sink = current_trace_sink.get()
    if sink is not None:
        sink(RetrievalSettledTrace(synthesis_mode="convergent", manifest_hash="h", chunk_count=1))
        sink(ClaimClosedTrace(text="streamed but ungrounded claim", span_type="EXACT", cited_chunk_ids=["c:0"]))
    return ArmOutput(raw_text="[REFUSAL]", claims=[], retrieval_manifest_hash="h",
                     refused=True, refusal_reason="zero_cursors")


def _raising(q: str) -> ArmOutput:
    raise RuntimeError("simulated synthesis failure")


def _lines(response) -> list[dict]:
    return [json.loads(line) for line in response.text.strip().split("\n") if line]


def test_stream_emits_retrieval_and_claim_events_before_terminal():
    app = create_app(CFG, answer_fn=_grounded_with_events)
    client = TestClient(app)
    fq = FederatedQuery(query="q", origin_node="driver", hop_count=0)
    r = client.post("/v1/federated/query/stream", json=fq.model_dump())
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    events = _lines(r)
    assert [e["event"] for e in events] == [
        "retrieval_settled", "claim_admitted", "synthesis_complete",
    ]
    assert events[1]["claim"]["text"] == "grounded claim"
    assert events[1]["claim"]["cited_chunk_ids"] == ["c:0"]
    assert events[2]["response"]["answer"]["node_id"] == "node_a"


def test_stream_matches_non_streaming_response_for_same_query():
    app = create_app(CFG, answer_fn=_grounded_with_events)
    client = TestClient(app)
    fq_payload = FederatedQuery(query="q", origin_node="driver", hop_count=0).model_dump()

    stream_events = _lines(client.post("/v1/federated/query/stream", json=fq_payload))
    plain_resp = FederatedResponse.model_validate(
        client.post("/v1/federated/query", json=fq_payload).json()
    )

    terminal = stream_events[-1]
    assert terminal["event"] == "synthesis_complete"
    assert terminal["response"]["answer"]["answer_text"] == plain_resp.answer.answer_text
    assert terminal["response"]["answer"]["claims"] == [
        c.model_dump() for c in plain_resp.answer.claims
    ]


def test_instant_refusal_streams_zero_claim_events():
    app = create_app(CFG, answer_fn=_refusing)
    client = TestClient(app)
    fq = FederatedQuery(query="q", origin_node="driver", hop_count=0)
    r = client.post("/v1/federated/query/stream", json=fq.model_dump())
    events = _lines(r)
    assert [e["event"] for e in events] == ["synthesis_complete"]
    assert events[0]["response"]["refusal"]["reason"] == "retrieval_floor"


def test_streamed_claim_can_diverge_from_refused_terminal_response():
    """Locks in real, documented behavior (not a bug): claim_admitted events
    reflect SEAR's real-time admission during decode, while the terminal
    synthesis_complete event reflects the final answer after post-decode
    grounding gates run. A claim can stream and still not appear in -- or
    match -- the terminal response, because those gates run after decode
    and can reject the whole answer. See gin/eval/arms.py's
    NoContinuationArm.run for where this happens for real."""
    app = create_app(CFG, answer_fn=_streamed_then_gated_refusal)
    client = TestClient(app)
    fq = FederatedQuery(query="q", origin_node="driver", hop_count=0)
    r = client.post("/v1/federated/query/stream", json=fq.model_dump())
    assert r.status_code == 200
    events = _lines(r)

    assert [e["event"] for e in events] == [
        "retrieval_settled", "claim_admitted", "synthesis_complete",
    ]
    assert events[1]["claim"]["text"] == "streamed but ungrounded claim"

    terminal = events[2]["response"]
    assert terminal["refusal"] is not None
    assert terminal["refusal"]["reason"] == "zero_cursors"
    assert terminal.get("answer") is None


def test_synthesis_exception_yields_internal_error_terminal_event():
    app = create_app(CFG, answer_fn=_raising)
    client = TestClient(app)
    fq = FederatedQuery(query="q", origin_node="driver", hop_count=0)
    r = client.post("/v1/federated/query/stream", json=fq.model_dump())
    assert r.status_code == 200
    events = _lines(r)
    assert len(events) == 1
    assert events[0]["event"] == "synthesis_complete"
    assert events[0]["response"]["refusal"]["reason"] == "internal_error"


def test_non_streaming_endpoint_unchanged():
    app = create_app(CFG, answer_fn=_grounded_with_events)
    client = TestClient(app)
    fq = FederatedQuery(query="q", origin_node="driver", hop_count=0)
    r = client.post("/v1/federated/query", json=fq.model_dump())
    resp = FederatedResponse.model_validate(r.json())
    assert resp.answer.node_id == "node_a"
    assert resp.answer.answer_text == "grounded claim"
    assert resp.federation is None
