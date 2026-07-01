"""Query relevance scoring for synthesis span steering."""
from __future__ import annotations

import re
from typing import Callable

from sear.corpus import Corpus, sentence_token_spans

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
