"""Query relevance scoring for synthesis span steering."""
from __future__ import annotations

import re
from typing import Callable, Optional

from sear.corpus import Corpus, sentence_token_spans

from .models import ChunkHit

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "to", "in", "for", "on", "at", "by",
    "with", "from", "is", "was", "were", "be", "been", "being",
})

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=—)\s+")


def query_keywords(query: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", query.lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def _sentence_texts(chunk_text: str) -> list[str]:
    text = chunk_text.strip()
    if not text:
        return []
    return [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]


def score_starts_by_sentence_match(
    corpus: Corpus,
    chunk_texts: list[str],
    query: str,
    tokenize: Callable[[bytes], list[int]],
    *,
    min_score: float = 0.1,
) -> tuple[set[tuple[int, int]], list[tuple[int, int, float]]]:
    """Map query keywords to sentence_starts via shared tokenizer span logic."""
    keywords = query_keywords(query)
    if not keywords:
        return set(corpus.sentence_starts), []

    best: dict[tuple[int, int], float] = {}
    for doc, text in enumerate(chunk_texts):
        sents = _sentence_texts(text)
        spans = sentence_token_spans(text, tokenize)
        for i, (start, _end) in enumerate(spans):
            if i >= len(sents):
                break
            sent = sents[i]
            words = set(re.findall(r"[a-z0-9]+", sent.lower()))
            score = len(keywords & words) / max(len(keywords), 1)
            if score <= 0:
                continue
            sent_start = (doc, start)
            if sent_start in corpus.sentence_starts:
                best[sent_start] = max(best.get(sent_start, 0.0), score)

    ranked = sorted(
        [(d, p, s) for (d, p), s in best.items()],
        key=lambda x: x[2],
        reverse=True,
    )
    if not ranked:
        return set(corpus.sentence_starts), []

    preferred = {(d, p) for d, p, s in ranked if s >= min_score}
    if not preferred:
        preferred = {(ranked[0][0], ranked[0][1])}
    return preferred, ranked


def max_sentence_score(text: str, query: str) -> float:
    """Best keyword-overlap score across sentences in a chunk."""
    keywords = query_keywords(query)
    if not keywords:
        return 0.0
    best = 0.0
    for sent in _sentence_texts(text):
        words = set(re.findall(r"[a-z0-9]+", sent.lower()))
        score = len(keywords & words) / max(len(keywords), 1)
        best = max(best, score)
    return best


def matched_keyword_count(text: str, query: str) -> int:
    """Most query keywords matched by any single sentence in a chunk."""
    keywords = query_keywords(query)
    if not keywords:
        return 0
    best = 0
    for sent in _sentence_texts(text):
        words = set(re.findall(r"[a-z0-9]+", sent.lower()))
        best = max(best, len(keywords & words))
    return best


def rerank_hits_by_query_score(hits: list[ChunkHit], query: str) -> list[ChunkHit]:
    """Order hits by max per-sentence query relevance (descending)."""
    if not hits or not query_keywords(query):
        return hits
    return sorted(hits, key=lambda h: max_sentence_score(h.text, query), reverse=True)


def score_starts_for_convergent(
    corpus: Corpus,
    chunk_texts: list[str],
    query: str,
    tokenize: Callable[[bytes], list[int]],
    *,
    min_score: float = 0.25,
) -> tuple[set[tuple[int, int]], list[tuple[int, int, float]], Optional[int]]:
    """Steer convergent decode to the top query-relevant doc only."""
    _preferred, ranked = score_starts_by_sentence_match(
        corpus, chunk_texts, query, tokenize, min_score=0.0
    )
    if not ranked:
        return set(corpus.sentence_starts), [], None

    doc_max: dict[int, float] = {}
    for doc, _pos, score in ranked:
        doc_max[doc] = max(doc_max.get(doc, 0.0), score)

    top_doc = max(doc_max.items(), key=lambda item: (item[1], -item[0]))[0]
    if doc_max[top_doc] < min_score:
        top_doc = ranked[0][0]

    preferred = {(d, p) for d, p, s in ranked if d == top_doc and s >= min_score}
    if not preferred:
        preferred = {(d, p) for d, p, _s in ranked if d == top_doc}
    if not preferred:
        preferred = {(ranked[0][0], ranked[0][1])}

    return preferred, ranked, top_doc
