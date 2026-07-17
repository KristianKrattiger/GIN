"""Ambient, request-scoped hook for streaming synthesis progress.

A ContextVar rather than a parameter threaded through decode_bundle's
callers, so instrumenting it for one streaming caller doesn't touch every
function signature between the HTTP layer and here. Deliberately
dependency-free — no gin.eval, no gin.federation imports — this module
sits at the gin.corpus layer; gin.federation (which already depends on
gin.corpus) translates these primitives into its own wire event types.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable, Optional, Union


@dataclass(frozen=True)
class RetrievalSettledTrace:
    synthesis_mode: str
    manifest_hash: str
    chunk_count: int


@dataclass(frozen=True)
class ClaimClosedTrace:
    text: str
    span_type: str
    cited_chunk_ids: list[str] = field(default_factory=list)


TraceEvent = Union[RetrievalSettledTrace, ClaimClosedTrace]

current_trace_sink: ContextVar[Optional[Callable[[TraceEvent], None]]] = ContextVar(
    "current_trace_sink", default=None
)
