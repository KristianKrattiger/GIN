"""Request and response shapes for the proposer sidecar, mirroring Receipts' ProposalBatchSchema."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Excerpt(BaseModel):
    docId: str
    role: Literal["claimant", "independent"]
    label: str
    text: str


class Message(BaseModel):
    role: str
    content: str


class ProposeRequest(BaseModel):
    model: str
    system: str
    messages: list[Message] = Field(min_length=1)
    excerpts: list[Excerpt]
    # The pass's task in Receipts (see slots.propose_pass); absent = claim-first over every type.
    mode: Optional[Literal["relational", "unsupported"]] = None


class Span(BaseModel):
    docId: str
    quote: str


class Proposal(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    type: Literal["contradicts", "corroborates", "updates", "unsupported"]
    topic: str
    statement: str
    from_: Span = Field(alias="from")
    to: Optional[Span]
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class ProposeResponse(BaseModel):
    model: str
    proposals: list[Proposal]
    stop_reason: Literal["end_turn"] = "end_turn"


class Health(BaseModel):
    model: str
