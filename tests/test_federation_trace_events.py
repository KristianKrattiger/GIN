"""Wire-protocol event shapes for the streaming reasoning-trace endpoint."""
import json

from gin.federation.schema import FederatedAnswer, FederatedResponse, WireClaim
from gin.federation.trace_events import (
    ClaimAdmittedEvent,
    RetrievalSettledEvent,
    SynthesisCompleteEvent,
)


def test_retrieval_settled_event_shape():
    e = RetrievalSettledEvent(synthesis_mode="convergent", manifest_hash="h", chunk_count=3)
    assert e.event == "retrieval_settled"
    data = json.loads(e.model_dump_json())
    assert data == {
        "event": "retrieval_settled", "synthesis_mode": "convergent",
        "manifest_hash": "h", "chunk_count": 3,
    }


def test_claim_admitted_event_wraps_wire_claim():
    claim = WireClaim(text="grounded claim", span_type="EXACT", cited_chunk_ids=["a:0"])
    e = ClaimAdmittedEvent(claim=claim)
    assert e.event == "claim_admitted"
    assert e.claim == claim


def test_synthesis_complete_event_wraps_federated_response():
    resp = FederatedResponse(
        answer=FederatedAnswer(request_id="r", node_id="node_a", answer_text="text")
    )
    e = SynthesisCompleteEvent(response=resp)
    assert e.event == "synthesis_complete"
    assert e.response.answer.node_id == "node_a"
