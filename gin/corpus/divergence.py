"""Compute divergence zones and forbidden tail spans for divergent synthesis."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable

from sear.corpus import Corpus, sentence_token_spans

from .models import ChunkHit

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=—)\s+")


def _sentence_texts(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]


def _doc_sentence_records(
    hit: ChunkHit,
    tokenize: Callable[[bytes], list[int]],
) -> list[tuple[str, int, int]]:
    """Return (sentence_text, start_pos, end_pos) aligned with tokenizer spans."""
    sents = _sentence_texts(hit.text)
    spans = sentence_token_spans(hit.text, tokenize)
    records: list[tuple[str, int, int]] = []
    for i, (start, end) in enumerate(spans):
        sent = sents[i] if i < len(sents) else ""
        records.append((sent, start, end))
    return records


def _word_overlap(a: str, b: str) -> int:
    wa = set(re.findall(r"[a-z0-9]+", a.lower()))
    wb = set(re.findall(r"[a-z0-9]+", b.lower()))
    return len(wa & wb)


def compute_divergence_zones(
    hits: list[ChunkHit],
    pairs: list[tuple[ChunkHit, ChunkHit, object]],
    corpus: Corpus,
    tokenize: Callable[[bytes], list[int]],
) -> tuple[dict[int, set[int]], set[tuple[int, int]]]:
    """
    From contradict pairs, mark per-doc sentence-start positions that diverge.

    Also returns forbidden_starts: doc-unique tail sentences (mayor/council/union lines)
    that are not part of any contradict-pair divergence zone.
    """
    chunk_to_doc = {hit.chunk_id: i for i, hit in enumerate(hits)}
    divergence_starts: dict[int, set[int]] = defaultdict(set)

    doc_records: dict[int, list[tuple[str, int, int]]] = {
        i: _doc_sentence_records(hit, tokenize) for i, hit in enumerate(hits)
    }

    for left, right, edge in pairs:
        edge_type = getattr(edge, "edge_type", edge)
        if str(edge_type) != "contradicts":
            continue
        li = chunk_to_doc.get(left.chunk_id)
        ri = chunk_to_doc.get(right.chunk_id)
        if li is None or ri is None:
            continue
        left_sents = doc_records[li]
        right_sents = doc_records[ri]
        for i, (ls, lp, _le) in enumerate(left_sents):
            if i < len(right_sents):
                rs, rp, _re = right_sents[i]
                if ls != rs:
                    if _word_overlap(ls, rs) < 3:
                        continue
                    if (li, lp) in corpus.sentence_starts:
                        divergence_starts[li].add(lp)
                    if (ri, rp) in corpus.sentence_starts:
                        divergence_starts[ri].add(rp)

    sent_text_to_docs: dict[str, set[int]] = defaultdict(set)
    for doc_idx, records in doc_records.items():
        for sent, _start, _end in records:
            sent_text_to_docs[sent].add(doc_idx)

    all_divergence = {
        (d, p) for d, poses in divergence_starts.items() for p in poses
    }
    forbidden: set[tuple[int, int]] = set()
    for doc_idx, records in doc_records.items():
        for sent, start, _end in records:
            key = (doc_idx, start)
            if key in all_divergence:
                continue
            if len(sent_text_to_docs[sent]) == 1:
                forbidden.add(key)

    return dict(divergence_starts), forbidden


def compute_divergence_sentence_ranges(
    divergence_starts: dict[int, set[int]],
    corpus: Corpus,
) -> dict[int, dict[int, int]]:
    """Map doc -> sentence_start -> inclusive end token index for divergence spans."""
    ranges: dict[int, dict[int, int]] = {}
    for doc, starts in divergence_starts.items():
        for start in starts:
            end = corpus.sentence_end_by_start.get((doc, start))
            if end is not None:
                ranges.setdefault(doc, {})[start] = end
    return ranges


def shared_sentence_starts(
    hits: list[ChunkHit],
    corpus: Corpus,
    tokenize: Callable[[bytes], list[int]],
) -> set[tuple[int, int]]:
    """Sentence starts whose text is identical across two or more docs (shared lede)."""
    sent_text_to_docs: dict[str, set[int]] = defaultdict(set)
    start_by_doc_sent: dict[tuple[int, str], int] = {}
    for doc_idx, hit in enumerate(hits):
        for sent, start, _end in _doc_sentence_records(hit, tokenize):
            sent_text_to_docs[sent].add(doc_idx)
            start_by_doc_sent[(doc_idx, sent)] = start

    shared: set[tuple[int, int]] = set()
    for sent, docs in sent_text_to_docs.items():
        if len(docs) < 2:
            continue
        for doc_idx in docs:
            pos = start_by_doc_sent.get((doc_idx, sent))
            if pos is not None and (doc_idx, pos) in corpus.sentence_starts:
                shared.add((doc_idx, pos))
    return shared
