"""Wire-protocol event shapes for the streaming reasoning-trace endpoint."""
import json

from pydantic import TypeAdapter

from gin.federation.schema import FederatedAnswer, FederatedResponse, WireClaim
from gin.federation.trace_events import (
    ClaimAdmittedEvent,
    RetrievalSettledEvent,
    SynthesisCompleteEvent,
    TraceEvent,
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


def test_trace_event_union_resolves_discriminator():
    """Validate that TraceEvent union discriminator correctly resolves all three event types."""
    adapter = TypeAdapter(TraceEvent)

    retrieval = adapter.validate_json(
        RetrievalSettledEvent(synthesis_mode="convergent", manifest_hash="h", chunk_count=3).model_dump_json()
    )
    assert isinstance(retrieval, RetrievalSettledEvent)

    claim = adapter.validate_json(
        ClaimAdmittedEvent(claim=WireClaim(text="t", span_type="EXACT", cited_chunk_ids=[])).model_dump_json()
    )
    assert isinstance(claim, ClaimAdmittedEvent)

    resp = FederatedResponse(answer=FederatedAnswer(request_id="r", node_id="node_a", answer_text="text"))
    terminal = adapter.validate_json(SynthesisCompleteEvent(response=resp).model_dump_json())
    assert isinstance(terminal, SynthesisCompleteEvent)
