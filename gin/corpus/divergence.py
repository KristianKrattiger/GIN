"""Compute divergence zones and forbidden tail spans for divergent synthesis."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable, Optional

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
    *,
    sentence_scorer: Optional[Callable[[str], float]] = None,
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
        marked = False
        for i, (ls, lp, _le) in enumerate(left_sents):
            if i < len(right_sents):
                rs, rp, _re = right_sents[i]
                if ls != rs:
                    if _word_overlap(ls, rs) < 3:
                        continue
                    if (li, lp) in corpus.sentence_starts:
                        divergence_starts[li].add(lp)
                        marked = True
                    if (ri, rp) in corpus.sentence_starts:
                        divergence_starts[ri].add(rp)
                        marked = True
        if not marked:
            # Structurally-dissimilar contradicts pair: an institutional
            # statistic ("56,580 wildfires burned 2.7M acres") vs a grassroots
            # reframing ("populations face risk from wildfire smoke") share no
            # aligned lede, so the index-aligned >=3-word overlap test above
            # never fires and the pair is left with no divergence zone. That is
            # fatal downstream: every doc-unique sentence -- including the
            # pair's own anchors -- falls into the forbidden tail net below,
            # blocking all span starts so the divergent decode "refuses". In a
            # reframing pair the divergence IS the framing on each side, so mark
            # an anchor sentence start per doc as its own divergence zone.
            #
            # With a sentence_scorer (query relevance), mark only the single
            # most-relevant sentence per doc. Without one, mark every sentence
            # start. These coincide for single-sentence chunks, but on a real
            # multi-paragraph chunk "mark every sentence" would turn filler/tail
            # lines ("we thank our volunteers") into divergence-steered starts
            # -- reintroducing the forbidden-tail problem this fallback exists to
            # avoid, one level down at chunk granularity.
            for doc_idx, sents in ((li, left_sents), (ri, right_sents)):
                eligible = [
                    (text, start)
                    for text, start, _end in sents
                    if (doc_idx, start) in corpus.sentence_starts
                ]
                if not eligible:
                    continue
                if sentence_scorer is not None:
                    # Ties (e.g. all sentences score 0 on a side that shares no
                    # query vocabulary) fall back to the earliest sentence.
                    best_start = max(
                        eligible, key=lambda es: (sentence_scorer(es[0]), -es[1])
                    )[1]
                    divergence_starts[doc_idx].add(best_start)
                else:
                    for _text, start in eligible:
                        divergence_starts[doc_idx].add(start)

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


def sentence_start_for_whitespace_anchor(
    text: str,
    anchor: tuple[int, int],
    tokenize: Callable[[bytes], list[int]],
    corpus: Corpus,
    doc_idx: int,
) -> Optional[int]:
    """Map a Bookkeeper whitespace-token anchor to a tokenizer sentence start."""
    ws_start, _ws_end = anchor
    sents = _sentence_texts(text)
    spans = sentence_token_spans(text, tokenize)
    word_idx = 0
    for i, (start, _end) in enumerate(spans):
        sent_words = len(sents[i].split()) if i < len(sents) else 0
        if word_idx <= ws_start < word_idx + max(sent_words, 1):
            if (doc_idx, start) in corpus.sentence_starts:
                return start
        word_idx += max(sent_words, 1)
    if spans and (doc_idx, spans[0][0]) in corpus.sentence_starts:
        return spans[0][0]
    return None


def divergence_starts_from_edge_anchors(
    hits: list[ChunkHit],
    pairs: list[tuple[ChunkHit, ChunkHit, object]],
    corpus: Corpus,
    tokenize: Callable[[bytes], list[int]],
) -> dict[int, set[int]]:
    """Seed divergence zones from admitted graph anchors when present."""
    chunk_to_doc = {hit.chunk_id: i for i, hit in enumerate(hits)}
    divergence_starts: dict[int, set[int]] = defaultdict(set)

    for left, right, edge in pairs:
        edge_type = getattr(edge, "edge_type", edge)
        if str(edge_type) != "contradicts":
            continue
        li = chunk_to_doc.get(left.chunk_id)
        ri = chunk_to_doc.get(right.chunk_id)
        if li is None or ri is None:
            continue
        src_anchor = getattr(edge, "src_anchor", None)
        dst_anchor = getattr(edge, "dst_anchor", None)
        if src_anchor is not None:
            start = sentence_start_for_whitespace_anchor(
                left.text, src_anchor, tokenize, corpus, li
            )
            if start is not None:
                divergence_starts[li].add(start)
        if dst_anchor is not None:
            start = sentence_start_for_whitespace_anchor(
                right.text, dst_anchor, tokenize, corpus, ri
            )
            if start is not None:
                divergence_starts[ri].add(start)

    return dict(divergence_starts)


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
