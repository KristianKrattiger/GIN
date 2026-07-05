"""Query relevance scoring for synthesis span steering."""
from __future__ import annotations

import math
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


# --- IDF-weighted divergence relevance --------------------------------------
# Self-contained scorer used only by the divergent-mode gate. It normalizes
# singular/plural ("wildfires" -> "wildfire") and weights each shared query
# keyword by its corpus IDF, so a single DISTINCTIVE match ("wildfire") clears
# the bar while a single GENERIC one ("district") does not — the separation the
# plain keyword-count gate cannot make. Kept independent of query_keywords /
# max_sentence_score so global retrieval/reranking behavior is unchanged.

def _normalize_token(word: str) -> str:
    """Light singular/plural fold: drop a single trailing 's' on longer words."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _norm_tokens(text: str) -> set[str]:
    return {_normalize_token(w) for w in re.findall(r"[a-z0-9]+", text.lower())}


def _norm_query_keywords(query: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", query.lower())
    return {
        _normalize_token(w)
        for w in words
        if len(w) > 2 and w not in _STOPWORDS
    }


def corpus_idf(chunk_texts: list[str]) -> dict[str, float]:
    """Smoothed IDF per normalized token over a chunk corpus (chunk = document)."""
    n_docs = len(chunk_texts) or 1
    doc_freq: dict[str, int] = {}
    for text in chunk_texts:
        for tok in _norm_tokens(text):
            doc_freq[tok] = doc_freq.get(tok, 0) + 1
    return {
        tok: math.log((n_docs + 1) / (df + 1)) + 1.0
        for tok, df in doc_freq.items()
    }


def idf_weighted_relevance(text: str, query: str, idf: dict[str, float]) -> float:
    """Fraction of the query's IDF mass matched by the best sentence in a chunk.

    Only query keywords present in the corpus (known IDF) count toward the
    denominator, so out-of-corpus query words don't dilute the score.
    """
    keywords = {k for k in _norm_query_keywords(query) if k in idf}
    if not keywords:
        return 0.0
    total = sum(idf[k] for k in keywords)
    if total <= 0:
        return 0.0
    best = 0.0
    for sent in _sentence_texts(text):
        matched = keywords & _norm_tokens(sent)
        best = max(best, sum(idf[k] for k in matched) / total)
    return best


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
