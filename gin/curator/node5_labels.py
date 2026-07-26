"""The node5 curator labels and the stance channel's pre-registered metric.

Four consumers need the same two things -- the labels folded latest-wins with
their event membership, and the P/R/P_all arithmetic -- so both live here once.
Node5's precedent is the same shape: gin/curator/node5_verify.py holds the
logic and scripts/verify_node5_surfacing.py is a thin shell over it.

The fold is Store.gold()'s, not a second implementation: Store already folds
the append-only log latest-wins and already yields a Relation enum rather than
a raw string.

Lives in gin.curator because it reads the label store. gin.cartographer may
not import gin.curator, so nothing in the cartographer package imports this --
its consumers are scripts and tests.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from gin.cartographer.models import Relation

from .corpus_json import load_corpus_chunks
from .store import Store

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = REPO_ROOT / "data" / "curator" / "labels.jsonl"
NODE5_CORPUS = REPO_ROOT / "corpus_node5.json"

_NODE5_PREFIX = "n5_doc_"

# Pre-registered in the spec BEFORE any held-out number was measured. Named
# here rather than derived by a rule later, because a rule chosen after the
# fact can be chosen to flatter.
HELD_OUT_EVENTS = frozenset({
    "lakeshore_algae_bloom",
    "civic_bond_audit",
    "stadium_capacity_ruling",
})

# combined.py's unconditional `if same_story: return CONTRADICTS`, measured on
# these 24 labels at ebceb46. P is within-event precision, P_all counts the 5
# cross-event stage-1 false positives against stage 2 as well.
BASELINE_P = 12 / 19
BASELINE_R = 1.0
BASELINE_P_ALL = 12 / 24


@dataclass(frozen=True)
class Node5Pair:
    src: str
    dst: str
    relation: Relation
    event: str            # the src endpoint's event
    within_event: bool    # both endpoints report the same event
    held_out: bool        # event is in HELD_OUT_EVENTS

    @property
    def gold_contradicts(self) -> bool:
        return self.relation is Relation.CONTRADICTS


@dataclass(frozen=True)
class MetricScore:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        """NaN, not 0.0, when nothing was typed CONTRADICTS.

        "Emitted nothing" and "emitted only wrong answers" are different
        failures and the report must not conflate them.
        """
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else math.nan

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else math.nan


def node5_texts(corpus: Path = NODE5_CORPUS) -> dict[str, str]:
    """chunk_id -> text for the node5 corpus, under NORMALISED ids.

    load_corpus_chunks turns the JSON's "n5_doc_001_c000" into "n5_doc_001:0",
    which is the form the label store and the candidate source both use.
    """
    return {c.chunk_id: c.text for c in load_corpus_chunks([corpus])}


def _event_of_doc(corpus: Path = NODE5_CORPUS) -> dict[str, str]:
    payload = json.loads(Path(corpus).read_text(encoding="utf-8"))
    return {doc["doc_id"]: doc["metadata"]["event"] for doc in payload["documents"]}


def node5_pairs(
    labels: Path = DEFAULT_LABELS, corpus: Path = NODE5_CORPUS
) -> list[Node5Pair]:
    """The curator's node5 labels, latest-wins, sorted for reproducibility."""
    event_of = _event_of_doc(corpus)
    pairs: list[Node5Pair] = []
    for src, dst, relation, _relation_class in Store(Path(labels)).gold():
        if not (src.startswith(_NODE5_PREFIX) and dst.startswith(_NODE5_PREFIX)):
            continue
        src_event = event_of[src.split(":")[0]]
        dst_event = event_of[dst.split(":")[0]]
        pairs.append(Node5Pair(
            src=src,
            dst=dst,
            relation=relation,
            event=src_event,
            within_event=src_event == dst_event,
            held_out=src_event in HELD_OUT_EVENTS,
        ))
    return sorted(pairs, key=lambda p: (p.event, p.src, p.dst))


def score(rows: Iterable[tuple[Node5Pair, bool]]) -> MetricScore:
    """Confusion counts for the CONTRADICTS channel.

    ``rows`` pairs each label with whether the pipeline typed it CONTRADICTS.
    A pair that is neither typed nor gold contradicts is a true negative and is
    counted in none of the three -- precision and recall are both about the
    contradicts channel only.
    """
    tp = fp = fn = 0
    for pair, typed in rows:
        if typed and pair.gold_contradicts:
            tp += 1
        elif typed:
            fp += 1
        elif pair.gold_contradicts:
            fn += 1
    return MetricScore(tp=tp, fp=fp, fn=fn)
