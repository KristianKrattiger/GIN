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
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

from gin.cartographer.labeled_set import chunks as labeled_set_chunks
from gin.curator.corpus_json import load_corpus_chunks
from gin.curator.models import pair_key
from gin.curator.store import Store

from .labels import TRAINING_CLASSES, FrameClass, bar_chunk_ids, frame_class_for

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = REPO_ROOT / "data" / "curator" / "labels.jsonl"
NEWS_CORPUS = REPO_ROOT / "data" / "synthetic" / "news_corpus.yaml"
CORPUS_NODES = tuple(REPO_ROOT / f"corpus_node{i}.json" for i in (1, 2, 3, 4))


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


def news_corpus_chunks(path: Path = NEWS_CORPUS) -> dict[str, str]:
    """Chunk texts from the synthetic news corpus.

    Ten escalation-bar chunks (inflation_*, labor_*, wage_*, export_*, school_*,
    transit_*) live here and nowhere else offline. Reading the YAML directly is
    what lets the bar be scored without Postgres.
    """
    if not path.is_file():
        raise FileNotFoundError(f"news corpus not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    index: dict[str, str] = {}
    for doc in data.get("documents", []):
        doc_id = doc["id"]
        for position, text in enumerate(doc.get("chunks", [])):
            index[f"{doc_id}:{position}"] = text
    return index


def default_text_index() -> dict[str, str]:
    """Union of the three offline text sources (236 chunks)."""
    index = {c.chunk_id: c.text for c in labeled_set_chunks()}
    for chunk in load_corpus_chunks(CORPUS_NODES):
        index[chunk.chunk_id] = chunk.text
    index.update(news_corpus_chunks())
    return index


@lru_cache(maxsize=1)
def bar_text_set() -> frozenset[str]:
    """Canonical TEXT of every escalation-bar chunk.

    Chunk-id exclusion alone is not sufficient. The fixture corpus aliases bar
    chunks under different ids with byte-identical text — `inst_em:0` IS
    `n1_doc_005:2`, `grass_wf:0` IS `n2_doc_005:1`, and six more. Guarding only
    on ids let 3 of the bar's 4 issue_frame pairs into training verbatim, which
    would let a future retrain report a green bar built on memorization. The
    encoder sees text, so the guard must too.
    """
    index = default_text_index()
    return frozenset(index[chunk_id] for chunk_id in bar_chunk_ids() if chunk_id in index)


def build_dataset(store: Store, text_index: Optional[dict[str, str]] = None) -> DatasetReport:
    """Fold the label log into trainable examples, counting every drop."""
    text = default_text_index() if text_index is None else text_index
    bar = bar_chunk_ids()
    # Derived from the DEFAULT index, not the caller's: the bar is fixed and its
    # text is canonical, so a caller passing a partial index must not be able to
    # silently disable the leakage guard.
    bar_texts = bar_text_set()
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
