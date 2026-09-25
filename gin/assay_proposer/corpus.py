"""
gin.assay_proposer.corpus
-------------------------
Build SEAR's token index from the Assay's excerpts, one SEAR doc per line.

A span never leaves the SEAR doc it started in, so making each doc a single
line of page text makes a quote across a layout edge (a stat tile stitched
to the heading under it) unreachable rather than caught afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

from sear.corpus import SENTENCE_BOUNDARY, Corpus

MAX_QUOTE_WORDS = 40


class ExcerptLike(Protocol):
    docId: str
    role: str
    text: str


@dataclass(frozen=True)
class LineSource:
    doc_id: str
    role: str


@dataclass
class LineCorpus:
    corpus: Corpus
    sources: list[LineSource]
    claimant: frozenset[int]
    independent: frozenset[int]
    forbidden_starts: set[tuple[int, int]]


def _sentence_offsets(line: str) -> list[tuple[int, int]]:
    """(start, end) character offsets of each sentence, split as sear.corpus splits."""
    starts = [0] + [m.end() for m in SENTENCE_BOUNDARY.finditer(line)]
    return [
        (s, starts[i + 1] if i + 1 < len(starts) else len(line))
        for i, s in enumerate(starts)
    ]


def build_line_corpus(
    excerpts: Iterable[ExcerptLike],
    tokenize: Callable[[bytes], list[int]],
) -> LineCorpus:
    chunks: list[tuple[str, str]] = []
    sources: list[LineSource] = []
    for ci, ex in enumerate(excerpts):
        for li, line in enumerate(ex.text.split("\n")):
            line = line.rstrip("\r")
            if not line.strip():
                continue
            chunks.append((f"{ex.docId}#{ci}:{li}", line))
            sources.append(LineSource(ex.docId, ex.role))

    corpus = Corpus.from_chunks(chunks, tokenize=tokenize)

    forbidden: set[tuple[int, int]] = set()
    for d, (_name, line) in enumerate(chunks):
        for start, end in _sentence_offsets(line):
            if len(line[start:end].split()) > MAX_QUOTE_WORDS:
                # The same token position sear.corpus records as this sentence's start.
                pos = len(tokenize(line[:start].encode("utf-8"))) if start else 0
                forbidden.add((d, pos))

    return LineCorpus(
        corpus=corpus,
        sources=sources,
        claimant=frozenset(i for i, s in enumerate(sources) if s.role == "claimant"),
        independent=frozenset(i for i, s in enumerate(sources) if s.role == "independent"),
        forbidden_starts=forbidden,
    )
