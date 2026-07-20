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
    p_contra = ev.get("p_contra")
    if p_contra is None and same_story is False:
        # type_relation story-blocks the NLI channel for not-same-story pairs
        # (combined.py), so its own dict never carries p_contra for them — but
        # every residue pair is, by construction, exactly this population.
        # Fall back to the same cross-encoder call directly so the signal
        # panel isn't blind to the contradiction strength that ranked the
        # pair to the top. Memoized on the proposer, so this is a cache hit
        # whenever the ranking already consulted NLI for this pair.
        p_contra = proposer.nli_p_contra(a_text, b_text)
    return {
        "cosine": ev.get("cos"),
        "nli_p_contra": p_contra,
        "same_story": same_story,
        "cheap_verdict": relation.value,
        "channel": ev.get("channel"),
    }
