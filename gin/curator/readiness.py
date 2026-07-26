"""No-model readiness gauge for sub-project B (bi-encoder).

Counts NEW labeled pairs per frame class, EXCLUDING the fixed escalation-bar
14 pairs (so the bar's own data never counts as progress toward training a
detector that will be measured on it). Pure counting — trains nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from gin.cartographer.escalation_eval import default_calibration_sets
from gin.cartographer.models import Relation

from .models import pair_key
from .store import Store
from .text_index import default_text_index, touches_bar_text


@dataclass(frozen=True)
class ReadinessTarget:
    issue_frame: int = 20
    agree: int = 20
    unrelated: int = 20
    story: int = 20


@dataclass(frozen=True)
class ReadinessReport:
    new_issue_frame: int
    new_agree: int
    new_unrelated: int
    new_story: int
    target: ReadinessTarget
    ready: bool


@lru_cache(maxsize=1)
def bar_pair_keys() -> frozenset[tuple[str, str]]:
    """The fixed escalation-bar pairs (issue_frame + corroboration + unrelated).

    Cached: the bar is constant, so the gold YAML fixtures behind
    default_calibration_sets() are read once, not on every readiness() call.
    """
    keys: set[tuple[str, str]] = set()
    for group in default_calibration_sets().values():
        for src, dst, _reg in group:
            keys.add(pair_key(src, dst))
    return frozenset(keys)


def readiness(store: Store, target: ReadinessTarget = ReadinessTarget()) -> ReadinessReport:
    bar = bar_pair_keys()
    index = default_text_index()
    n_if = n_ag = n_un = n_st = 0
    for src, dst, relation, relation_class in store.gold():
        if pair_key(src, dst) in bar:
            continue
        # Chunk-id exclusion alone overstates readiness: the fixture corpus
        # files bar chunks under alias ids with identical text, and the
        # consumer (gin.frames) drops any pair touching bar TEXT. Count what
        # that consumer can actually train on, not what the ids suggest.
        if touches_bar_text(src, dst, index):
            continue
        if relation is Relation.CONTRADICTS and relation_class == "issue_frame":
            n_if += 1
        elif relation is Relation.CONTRADICTS and relation_class == "story":
            n_st += 1
        elif relation is Relation.CORROBORATES:
            n_ag += 1
        elif relation is Relation.UNRELATED:
            n_un += 1
    ready = (
        n_if >= target.issue_frame
        and n_ag >= target.agree
        and n_un >= target.unrelated
        and n_st >= target.story
    )
    return ReadinessReport(n_if, n_ag, n_un, n_st, target, ready)
