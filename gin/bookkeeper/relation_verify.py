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
DEFAULT_BAND_ADMIT_FLOOR = 0.65
# Mid-band framing divergences (cos 0.13–0.42 on labeled set); below this is noise.
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
    band_admit_floor: float = DEFAULT_BAND_ADMIT_FLOOR,
    contra_threshold: float = 0.686,
    min_confidence: float = 0.5,
) -> RelationVerifyResult:
    """Conservative semantic checks for ``contradicts`` proposals."""
    if proposal.relation != Relation.CONTRADICTS:
        return RelationVerifyResult(ok=True)

    scores_ab = nli_scores(src_text, dst_text)
    scores_ba = nli_scores(dst_text, src_text)
    p_contra = max(scores_ab[0], scores_ba[0])
    p_ent_ab, p_ent_ba = scores_ab[1], scores_ba[1]

    if p_ent_ab >= entail_floor and p_ent_ba >= entail_floor:
        return RelationVerifyResult(
            ok=False,
            reason=(
                f"bidirectional entailment ({p_ent_ab:.3f}, {p_ent_ba:.3f}) "
                f">= {entail_floor:.3f}"
            ),
        )

    if proposal.method.endswith(":nli") and proposal.confidence >= min_confidence:
        return RelationVerifyResult(ok=True)

    if proposal.method.endswith(":band"):
        if p_contra >= contra_threshold or proposal.confidence >= band_admit_floor:
            return RelationVerifyResult(ok=True)
        if proposal.confidence < FRAMING_BAND_FLOOR:
            return RelationVerifyResult(
                ok=False,
                reason=(
                    f"band contradicts conf {proposal.confidence:.3f} < "
                    f"{FRAMING_BAND_FLOOR:.3f} framing floor"
                ),
            )

    return RelationVerifyResult(ok=True)
