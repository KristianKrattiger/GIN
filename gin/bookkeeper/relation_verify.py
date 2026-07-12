"""Bookkeeper semantic re-check for contradicts proposals.

Relation-type correctness is not recoverable at read time from relevance alone
(see docs/nc_reasoning_robustness_noisy_edges.plan.md class C). This module
denies spurious ``contradicts`` that pass structural gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from gin.cartographer.models import EdgeProposal, Relation

# (premise, hypothesis) -> (p_contradiction, p_entailment, p_neutral)
NliScorer = Callable[[str, str], tuple[float, float, float]]

DEFAULT_ENTAIL_FLOOR = 0.5
# Mid-band framing divergences (cos 0.13–0.42 on labeled set); below this is
# noise. Enforced at the Bookkeeper confidence gate for band-channel proposals.
FRAMING_BAND_FLOOR = 0.35


@dataclass(frozen=True)
class RelationVerifyResult:
    ok: bool
    reason: str = ""


def verify_contradicts(
    proposal: EdgeProposal,
    *,
    src_text: str,
    dst_text: str,
    nli_scores: NliScorer,
    entail_floor: float = DEFAULT_ENTAIL_FLOOR,
) -> RelationVerifyResult:
    """Deny a ``contradicts`` proposal whose texts entail each other both ways.

    This is the re-check's only live rule. The earlier band/nli branches were
    circular — the same NLI scores and thresholds the proposer had already
    applied — so they could not catch the proposer's systematic errors, and the
    band confidence floor duplicated the Bookkeeper's gate. A genuinely
    independent re-check needs a second signal source (a different model);
    until one exists, this check stays deliberately narrow.
    """
    if proposal.relation != Relation.CONTRADICTS:
        return RelationVerifyResult(ok=True)

    p_ent_ab = nli_scores(src_text, dst_text)[1]
    p_ent_ba = nli_scores(dst_text, src_text)[1]

    if p_ent_ab >= entail_floor and p_ent_ba >= entail_floor:
        return RelationVerifyResult(
            ok=False,
            reason=(
                f"bidirectional entailment ({p_ent_ab:.3f}, {p_ent_ba:.3f}) "
                f">= {entail_floor:.3f}"
            ),
        )
    return RelationVerifyResult(ok=True)
