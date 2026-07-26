"""Offline chunk-text index and the escalation bar's canonical TEXT.

Lives in `gin.curator` rather than a consumer package because two independent
consumers need it — the readiness gauge here, and the frame-detector dataset in
`gin.frames` — and nothing may import `gin.frames`.

**Why a TEXT-level bar guard exists at all.** The fixture corpus stores the same
passage under more than one chunk id: `inst_em:0` IS `n1_doc_005:2`,
`grass_wf:0` IS `n2_doc_005:1`, and six more. Any exclusion keyed on chunk id
therefore leaks — 3 of the escalation bar's 4 issue_frame pairs are reachable
under fixture ids. Both the training filter and the readiness gauge must compare
the text an encoder would actually see, not the id it happens to be filed under.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from gin.cartographer.escalation_eval import default_calibration_sets
from gin.cartographer.labeled_set import chunks as labeled_set_chunks

from .corpus_json import load_corpus_chunks

REPO_ROOT = Path(__file__).resolve().parents[2]
NEWS_CORPUS = REPO_ROOT / "data" / "synthetic" / "news_corpus.yaml"
CORPUS_NODES = tuple(REPO_ROOT / f"corpus_node{i}.json" for i in (1, 2, 3, 4, 5))


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
    """Union of the three offline text sources."""
    index = {c.chunk_id: c.text for c in labeled_set_chunks()}
    for chunk in load_corpus_chunks(CORPUS_NODES):
        index[chunk.chunk_id] = chunk.text
    index.update(news_corpus_chunks())
    return index


@lru_cache(maxsize=1)
def bar_chunk_texts() -> frozenset[str]:
    """Canonical TEXT of every escalation-bar chunk.

    Derived from the default index rather than a caller-supplied one, so a
    partial index cannot silently disable the guard built on it.
    """
    index = default_text_index()
    texts: set[str] = set()
    for group in default_calibration_sets().values():
        for src, dst, _register in group:
            for chunk_id in (src, dst):
                if chunk_id in index:
                    texts.add(index[chunk_id])
    return frozenset(texts)


def touches_bar_text(src_chunk_id: str, dst_chunk_id: str,
                     index: dict[str, str] | None = None) -> bool:
    """True if either endpoint's text is escalation-bar text, under any id."""
    index = default_text_index() if index is None else index
    bar_texts = bar_chunk_texts()
    return any(
        index.get(chunk_id) in bar_texts
        for chunk_id in (src_chunk_id, dst_chunk_id)
        if chunk_id in index
    )
