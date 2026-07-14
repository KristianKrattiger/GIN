"""Local answer seam for federation.

One function both the node server and the router call. It wraps
NoContinuationArm — the measured Phase 1/2 answer path — unchanged: retrieval
floor, materialize, constrained decode, claim extraction, and the refusal
semantics federation delegates on. The spec's ``answer_query`` service
function is this thin wrapper; the CLI keeps its existing direct path.
"""
from __future__ import annotations

from typing import Any, Optional

from gin.eval.arms import ArmConfig, ArmOutput, NoContinuationArm

from .schema import WireClaim


def answer_query(
    query: str, llm: Any, arm_config: Optional[ArmConfig] = None
) -> ArmOutput:
    """Run this node's full local answer path for one query."""
    return NoContinuationArm(arm_config).run(query, llm)


def claims_to_wire(output: ArmOutput) -> list[WireClaim]:
    """Serialize extracted claims for the wire, field-for-field."""
    return [
        WireClaim(
            text=c.text,
            span_type=c.span_type,
            cited_chunk_ids=list(c.cited_chunk_ids),
        )
        for c in output.claims
    ]
