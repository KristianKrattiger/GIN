"""
sear.corpus
-----------
Token-indexed document store for SEAR's cursor-based attribution system.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|(?<=—)\s+")


def sentence_start_token_positions(
    text: str,
    tokenize: Callable[[bytes], list[int]],
) -> list[int]:
    """Sorted token indices where each sentence begins within chunk text."""
    positions = [0]
    for match in SENTENCE_BOUNDARY.finditer(text):
        prefix = text[: match.end()]
        positions.append(len(tokenize(prefix.encode("utf-8"))))
    return sorted(set(positions))


def sentence_token_spans(
    text: str,
    tokenize: Callable[[bytes], list[int]],
) -> list[tuple[int, int]]:
    """Return (start_pos, end_pos) inclusive for each sentence in chunk text."""
    toks = tokenize(text.encode("utf-8"))
    if not toks:
        return []
    starts = sentence_start_token_positions(text, tokenize)
    spans: list[tuple[int, int]] = []
    for i, start in enumerate(starts):
        end = (starts[i + 1] - 1) if i + 1 < len(starts) else len(toks) - 1
        if start <= end:
            spans.append((start, end))
    return spans


def _sentence_token_positions(text: str, tokenize: Callable[[bytes], list[int]]) -> set[int]:
    return set(sentence_start_token_positions(text, tokenize))


@dataclass
class Corpus:
    docs: list[list[int]]
    doc_names: list[str]
    start_index: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    doc_meta: list[dict] = field(default_factory=list)
    sentence_starts: set[tuple[int, int]] = field(default_factory=set)
    sentence_ends: set[tuple[int, int]] = field(default_factory=set)
    sentence_end_by_start: dict[tuple[int, int], int] = field(default_factory=dict)

    @classmethod
    def from_texts(cls, texts: dict[str, str], tokenize: Callable[[bytes], list[int]]):
        ordered = list(texts.items())
        return cls.from_chunks(ordered, tokenize=tokenize)

    @classmethod
    def from_chunks(
        cls,
        chunks: list[tuple[str, str]],
        tokenize: Callable[[bytes], list[int]],
        *,
        doc_meta: Optional[list[dict]] = None,
    ):
        """Build a token index from stable (chunk_id, text) pairs."""
        docs, names = [], []
        sentence_starts: set[tuple[int, int]] = set()
        sentence_ends: set[tuple[int, int]] = set()
        sentence_end_by_start: dict[tuple[int, int], int] = {}
        for chunk_id, text in chunks:
            names.append(chunk_id)
            docs.append(tokenize(text.encode("utf-8")))
        index: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for d, toks in enumerate(docs):
            chunk_text = chunks[d][1]
            for start, end in sentence_token_spans(chunk_text, tokenize):
                if start < len(toks):
                    sentence_starts.add((d, start))
                    sentence_end_by_start[(d, start)] = end
                    if end < len(toks):
                        sentence_ends.add((d, end))
            for p, t in enumerate(toks):
                index[t].append((d, p))
        meta = doc_meta if doc_meta is not None else [{} for _ in names]
        return cls(
            docs=docs,
            doc_names=names,
            start_index=dict(index),
            doc_meta=meta,
            sentence_starts=sentence_starts,
            sentence_ends=sentence_ends,
            sentence_end_by_start=sentence_end_by_start,
        )

    def continuation(self, doc: int, pos: int) -> Optional[int]:
        nxt = pos + 1
        return self.docs[doc][nxt] if nxt < len(self.docs[doc]) else None
