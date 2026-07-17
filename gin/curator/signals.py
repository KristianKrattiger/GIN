"""Read-only view of the cheap pipeline's signals, for display + ordering.

Wraps CombinedRelationProposer.type_relation — no new model code. Whatever the
detector already computes for a pair (cosine always; NLI p_contra only when the
pair passes the gate and isn't story-blocked) is surfaced; unavailable signals
are reported as None rather than recomputed.
"""
from __future__ import annotations

from typing import Optional

from gin.cartographer.combined import CombinedRelationProposer


def pair_signals(a_text: str, b_text: str, proposer: CombinedRelationProposer) -> dict:
    relation, ev = proposer.type_relation(a_text, b_text)
    same_story: Optional[bool] = ev.get("same_story")
    return {
        "cosine": ev.get("cos"),
        "nli_p_contra": ev.get("p_contra"),
        "same_story": same_story,
        "cheap_verdict": relation.value,
        "channel": ev.get("channel"),
    }
