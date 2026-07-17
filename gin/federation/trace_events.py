"""Wire-protocol events for the streaming reasoning-trace endpoint
(POST /v1/federated/query/stream). Translates gin.corpus.trace_events'
primitive, dependency-free trace types into this layer's wire vocabulary —
the same translation boundary gin.federation.service.claims_to_wire already
draws between gin.eval's RawClaim and this module's WireClaim.
"""
from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel

from .schema import FederatedResponse, WireClaim


class RetrievalSettledEvent(BaseModel):
    event: Literal["retrieval_settled"] = "retrieval_settled"
    synthesis_mode: str
    manifest_hash: str
    chunk_count: int


class ClaimAdmittedEvent(BaseModel):
    event: Literal["claim_admitted"] = "claim_admitted"
    claim: WireClaim


class SynthesisCompleteEvent(BaseModel):
    event: Literal["synthesis_complete"] = "synthesis_complete"
    response: FederatedResponse


TraceEvent = Union[RetrievalSettledEvent, ClaimAdmittedEvent, SynthesisCompleteEvent]
