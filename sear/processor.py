"""
sear.processor
--------------
Hand-rolled LogitsProcessor implementing cursor-based copy constraints.
Tracks (doc_id, position) cursor tuples through multi-token spans.
Zero live cursors = grounding failure signal: the corpus cannot support
the current continuation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .corpus import Corpus

NEG_INF = -1e30


# --------------------------------------------------------------------------
# The constraint. Maintains a set of live cursors -- (doc, pos) of the last
# matched token -- so the legal continuation set is the union of next tokens
# across all live cursors. A span present in N documents keeps N cursors live
# until the emitted continuation diverges; the surviving set at span-close *is*
# the corpus-situatedness / divergence signal, not noise.
# --------------------------------------------------------------------------
BOUNDARY, IN_SPAN = "BOUNDARY", "IN_SPAN"


@dataclass
class Segment:
    token_ids: list[int]
    sources: list[tuple[int, int, int]]   # (doc_id, start_pos, end_pos_exclusive)
    kind: str                             # "extract" or "connective"


class ExtractiveCopyConstraint:
    def __init__(self, corpus: Corpus, prompt_len: int, eos_id: int,
                 delim_id: int, min_span_len: int = 3):
        self.corpus = corpus
        self.prompt_len = prompt_len
        self.eos_id = eos_id
        self.delim_id = delim_id            # boundary marker the model may emit
        self.min_span_len = min_span_len
        self.structural = {eos_id, delim_id}

        self.mode = BOUNDARY
        self.cursors: list[tuple[int, int]] = []   # live (doc, pos)
        self.span_start: list[tuple[int, int]] = []  # cursors at span start
        self.span_len = 0
        self.segments: list[Segment] = []
        self._cur_tokens: list[int] = []
        self._seen = prompt_len

    # ---- legal next-token set given current state -------------------------
    def _allowed(self) -> set[int]:
        if self.mode == BOUNDARY:
            allowed = set(self.corpus.start_index.keys())  # start any span
            allowed.add(self.eos_id)                       # or stop
            return allowed
        # IN_SPAN: continue any live cursor ...
        allowed = set()
        for (d, p) in self.cursors:
            c = self.corpus.continuation(d, p)
            if c is not None:
                allowed.add(c)
        # ... and, once the span is long enough to be a real extraction,
        # allow closing it (delimiter) or ending generation.
        if self.span_len >= self.min_span_len:
            allowed.add(self.delim_id)
            allowed.add(self.eos_id)
        if not allowed:           # dead end (span ran to doc end, too short)
            allowed.add(self.eos_id)
        return allowed

    # ---- advance state by one *generated* token ---------------------------
    def _consume(self, tok: int):
        if self.mode == BOUNDARY:
            if tok in self.structural:
                if tok == self.delim_id:
                    self.segments.append(
                        Segment([tok], [], "connective"))
                return
            # start a new span
            self.cursors = list(self.corpus.start_index.get(tok, []))
            self.span_start = list(self.cursors)
            self.span_len = 1
            self._cur_tokens = [tok]
            self.mode = IN_SPAN
            return
        # IN_SPAN
        if tok in self.structural:
            self._close_span()
            if tok == self.delim_id:
                self.segments.append(Segment([tok], [], "connective"))
            self.mode = BOUNDARY
            return
        # continue: keep only cursors whose continuation == tok, advance them
        new_cursors = []
        for (d, p) in self.cursors:
            if self.corpus.continuation(d, p) == tok:
                new_cursors.append((d, p + 1))
        self.cursors = new_cursors
        self.span_len += 1
        self._cur_tokens.append(tok)

    def _close_span(self):
        if self.span_len == 0:
            return
        sources = []
        for (d, end_pos) in self.cursors:          # end_pos = last matched
            start = end_pos - (self.span_len - 1)
            sources.append((d, start, end_pos + 1))
        self.segments.append(Segment(list(self._cur_tokens), sources, "extract"))
        self.span_len = 0
        self._cur_tokens = []

    # ---- runtime hook: (input_ids, scores) -> scores ----------------------
    def __call__(self, input_ids, scores):
        ids = list(input_ids)
        # consume any generated tokens we haven't processed yet
        for i in range(self._seen, len(ids)):
            self._consume(ids[i])
        self._seen = len(ids)

        allowed = self._allowed()
        scores = np.asarray(scores, dtype=np.float32)
        mask = np.full(scores.shape, NEG_INF, dtype=np.float32)
        idx = np.fromiter((t for t in allowed if t < scores.shape[0]),
                          dtype=np.int64)
        mask[idx] = scores[idx]
        return mask

    # ---- read the attribution record -------------------------------------
    def finalize(self):
        if self.mode == IN_SPAN:
            self._close_span()
        return self.segments

    def render(self, detok: Callable[[list[int]], str]) -> str:
        out = []
        for seg in self.segments:
            if seg.kind == "connective":
                out.append("  |  ")
                continue
            text = detok(seg.token_ids)
            srcs = ", ".join(f"{self.corpus.doc_names[d]}[{s}:{e}]"
                             for (d, s, e) in seg.sources) or "UNATTRIBUTED"
            tag = "AMBIGUOUS" if len(seg.sources) > 1 else "EXACT"
            out.append(f'"{text}"  <- {tag}: {srcs}')
        return "\n".join(out)
