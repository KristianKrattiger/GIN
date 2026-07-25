"""Training-set assembly from the curator label store.

Three ordered filters, each drop counted by reason and surfaced — never silent:

  1. schema          — relation/relation_class not in the 4-way map
  2. bar_chunk       — either endpoint appears anywhere in the escalation bar
  3. text_unresolved — no text available for an endpoint

Rows come from Store.gold(), the latest-wins FOLD of the append-only log, never
from raw JSONL lines: 104 lines currently fold to 102 unique pairs, so counting
lines double-counts relabeled pairs and trains on stale labels.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from gin.curator.models import pair_key
from gin.curator.store import Store
# Shared with gin.curator.readiness: one bar-text guard, one definition.
from gin.curator.text_index import (  # noqa: F401  (re-exported for callers)
    CORPUS_NODES,
    NEWS_CORPUS,
    bar_chunk_texts,
    default_text_index,
    news_corpus_chunks,
)

from .labels import TRAINING_CLASSES, FrameClass, bar_chunk_ids, frame_class_for

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = REPO_ROOT / "data" / "curator" / "labels.jsonl"


@dataclass(frozen=True)
class FrameExample:
    src_chunk_id: str
    dst_chunk_id: str
    src_text: str
    dst_text: str
    label: FrameClass


@dataclass(frozen=True)
class DatasetReport:
    examples: list[FrameExample]
    drops: dict[str, int]

    @property
    def counts(self) -> dict[str, int]:
        return dict(Counter(e.label.value for e in self.examples))


def build_dataset(store: Store, text_index: Optional[dict[str, str]] = None) -> DatasetReport:
    """Fold the label log into trainable examples, counting every drop."""
    text = default_text_index() if text_index is None else text_index
    bar = bar_chunk_ids()
    # Derived from the DEFAULT index, not the caller's: the bar is fixed and its
    # text is canonical, so a caller passing a partial index must not be able to
    # silently disable the leakage guard.
    bar_texts = bar_chunk_texts()
    drops: Counter[str] = Counter()
    examples: list[FrameExample] = []

    # Sorted so leave-one-out folds are reproducible run to run.
    for src, dst, relation, relation_class in sorted(
        store.gold(), key=lambda row: pair_key(row[0], row[1])
    ):
        label = frame_class_for(relation, relation_class)
        if label is None:
            drops["schema"] += 1
            continue
        if src in bar or dst in bar:
            drops["bar_chunk"] += 1
            continue
        if src not in text or dst not in text:
            drops["text_unresolved"] += 1
            continue
        if text[src] in bar_texts or text[dst] in bar_texts:
            drops["bar_text_alias"] += 1
            continue
        examples.append(FrameExample(src, dst, text[src], text[dst], label))

    report = DatasetReport(examples, dict(drops))
    if not examples:
        raise ValueError(f"no trainable examples after filtering (drops: {report.drops})")
    empty = [c.value for c in TRAINING_CLASSES if report.counts.get(c.value, 0) == 0]
    if empty:
        raise ValueError(f"class(es) empty after filtering: {', '.join(empty)}")
    return report
