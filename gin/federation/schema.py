"""Wire protocol for Federation v1.

These Pydantic models ARE the protocol contract; HTTP is incidental transport
behind the PeerClient seam (client.py). Version every change through
PROTOCOL_VERSION — a node that receives a different version refuses with
``version_mismatch`` rather than best-effort parsing.
"""
from __future__ import annotations

from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

PROTOCOL_VERSION = 1

# A node's own failure reason. peer_reasons values are free-form strings
# (they include transport outcomes like "unreachable").
RefusalReason = Literal[
    "retrieval_floor", "zero_cursors", "hop_limit", "version_mismatch"
]


def new_request_id() -> str:
    return str(uuid4())


class WireClaim(BaseModel):
    """One extracted claim, mirroring gin.eval.claims.RawClaim on the wire."""

    text: str
    span_type: str
    cited_chunk_ids: list[str] = Field(default_factory=list)


class FederationLayer(BaseModel):
    """Provenance extension: how a delegated answer reached the caller."""

    answered_by: str
    hop_count: int
    transport: str = "http"
    peer_url: str = ""
    request_id: str


class FederatedQuery(BaseModel):
    protocol_version: int = PROTOCOL_VERSION
    request_id: str = Field(default_factory=new_request_id)
    query: str
    origin_node: str
    hop_count: int = 0


class FederatedAnswer(BaseModel):
    protocol_version: int = PROTOCOL_VERSION
    request_id: str
    node_id: str
    answer_text: str
    claims: list[WireClaim] = Field(default_factory=list)
    corpus_fingerprint: dict = Field(default_factory=dict)
    synthesis_mode: str = "unknown"
    timing_s: float = 0.0


class NodeRefusal(BaseModel):
    protocol_version: int = PROTOCOL_VERSION
    request_id: str
    node_id: str
    reason: RefusalReason
    detail: str = ""
    # On an aggregated (hop-0) refusal: what each consulted peer said.
    peer_reasons: dict[str, str] = Field(default_factory=dict)


class FederatedResponse(BaseModel):
    """Endpoint envelope: exactly one of answer/refusal, plus optional
    federation provenance (present only on hop-0 delegated answers)."""

    answer: Optional[FederatedAnswer] = None
    refusal: Optional[NodeRefusal] = None
    federation: Optional[FederationLayer] = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "FederatedResponse":
        if (self.answer is None) == (self.refusal is None):
            raise ValueError("exactly one of answer/refusal must be set")
        return self
