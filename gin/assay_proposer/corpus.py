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

# The only definition of this -- slots.py imports it from here. Lives here,
# not in slots.py, so this module can forbid a sentence whose token count
# would overrun the quote decode's token budget without slots.py importing
# corpus.py importing slots.py back.
MAX_QUOTE_TOKENS = 160

# The only definition of this -- slots.py imports it from here. A span can
# never legally close under this many tokens (sear/processor.py's
# _span_close_permitted requires span_len >= min_span_len), so a sentence
# shorter than it can never finish a quote: it either dead-ends at the end of
# its line (forcing EOS with too little copied, which _quote rejects) or, if
# a next sentence follows on the same line, bleeds into it instead.
MIN_SPAN_TOKENS = 4


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


def _sentences(line: str, tokenize: Callable[[bytes], list[int]]) -> list[tuple[int, int, str]]:
    """(token_start, token_end_inclusive, sentence_text) for each sentence of one line.

    sear.corpus counts a sentence's start as the tokens before it *including*
    the whitespace that separates it. A subword tokenizer makes that trailing
    space its own token, but in the full line the space fuses with the next
    word (" Acme"), so SEAR's start lands one token late, mid-word. Counting
    the prefix only up to the punctuation gives the token where the next
    sentence, with its leading space, begins.
    """
    toks = tokenize(line.encode("utf-8"))
    bounds = [(0, 0)]  # (text_start, token_start)
    for m in SENTENCE_BOUNDARY.finditer(line):
        bounds.append((m.end(), len(tokenize(line[: m.start()].encode("utf-8")))))
    out: list[tuple[int, int, str]] = []
    for i, (text_start, tok_start) in enumerate(bounds):
        text_end = bounds[i + 1][0] if i + 1 < len(bounds) else len(line)
        tok_end = bounds[i + 1][1] - 1 if i + 1 < len(bounds) else len(toks) - 1
        text = line[text_start:text_end]
        if tok_start <= tok_end and tok_start < len(toks) and text.strip():
            out.append((tok_start, tok_end, text))
    return out


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

    starts: set[tuple[int, int]] = set()
    ends: set[tuple[int, int]] = set()
    end_by_start: dict[tuple[int, int], int] = {}
    forbidden: set[tuple[int, int]] = set()
    for d, (_name, line) in enumerate(chunks):
        for tok_start, tok_end, text in _sentences(line, tokenize):
            starts.add((d, tok_start))
            ends.add((d, tok_end))
            end_by_start[(d, tok_start)] = tok_end
            token_count = tok_end - tok_start + 1
            if (
                len(text.split()) > MAX_QUOTE_WORDS
                or token_count >= MAX_QUOTE_TOKENS
                or token_count < MIN_SPAN_TOKENS
            ):
                forbidden.add((d, tok_start))
    # Override SEAR's own boundaries, which are one token late under subword tokenizers (see _sentences).
    corpus.sentence_starts = starts
    corpus.sentence_ends = ends
    corpus.sentence_end_by_start = end_by_start

    return LineCorpus(
        corpus=corpus,
        sources=sources,
        claimant=frozenset(i for i, s in enumerate(sources) if s.role == "claimant"),
        independent=frozenset(i for i, s in enumerate(sources) if s.role == "independent"),
        forbidden_starts=forbidden,
    )
