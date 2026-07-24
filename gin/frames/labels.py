"""The 4-way training schema and the escalation-bar chunk blocklist.

DIVERGENT is issue_frame ONLY. story-class contradicts are excluded: NLI already
types propositional conflict upstream (combined.py, p_contra 0.899 on the legal
register), so the escalation judge never meets those pairs in production, and
mixing them dilutes the stance axis this detector exists to learn.

RELATED_UNTYPED is kept as a first-class training label because "topically
related, no typed relation" is exactly the hard negative every LLM judge
collapsed on. It is never emitted — inference folds it into UNRELATED.
"""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Optional

from gin.cartographer.escalation_eval import default_calibration_sets
from gin.cartographer.models import Relation


class FrameClass(str, Enum):
    DIVERGENT = "DIVERGENT"
    AGREE = "AGREE"
    RELATED_UNTYPED = "RELATED_UNTYPED"
    UNRELATED = "UNRELATED"


TRAINING_CLASSES: tuple[FrameClass, ...] = (
    FrameClass.DIVERGENT,
    FrameClass.AGREE,
    FrameClass.RELATED_UNTYPED,
    FrameClass.UNRELATED,
)

_LABEL_MAP: dict[tuple[Relation, Optional[str]], FrameClass] = {
    (Relation.CONTRADICTS, "issue_frame"): FrameClass.DIVERGENT,
    (Relation.CORROBORATES, None): FrameClass.AGREE,
    (Relation.RELATED_UNTYPED, None): FrameClass.RELATED_UNTYPED,
    (Relation.UNRELATED, None): FrameClass.UNRELATED,
}

# 4-way training class -> the 3-label FrameJudge contract.
JUDGE_LABEL: dict[FrameClass, str] = {
    FrameClass.DIVERGENT: "DIVERGENT",
    FrameClass.AGREE: "AGREE",
    FrameClass.RELATED_UNTYPED: "UNRELATED",
    FrameClass.UNRELATED: "UNRELATED",
}


def frame_class_for(relation: Relation, relation_class: Optional[str]) -> Optional[FrameClass]:
    """Training class for a labeled pair, or None if the pair is not trainable."""
    return _LABEL_MAP.get((relation, relation_class))


@lru_cache(maxsize=1)
def bar_chunk_ids() -> frozenset[str]:
    """Every chunk id appearing anywhere in the escalation bar.

    Chunk-level, not pair-level: labeling the residue drew from the same n1/n2
    corpus the bar was built from, so 9 bar chunks entered the label pool in
    different pairings. readiness.py filters exact bar PAIRS, so that reuse
    passed the gauge invisibly. Training on them would make a bar score
    partly unearned.
    """
    ids: set[str] = set()
    for group in default_calibration_sets().values():
        for src, dst, _register in group:
            ids.add(src)
            ids.add(dst)
    return frozenset(ids)
