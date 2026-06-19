"""
sear.corpus
-----------
Token-indexed document store for SEAR's cursor-based attribution system.
The Corpus is the grammar: a LogitsProcessor consults it at every decode step
to determine which token ids are currently grounded in source spans.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional


# --------------------------------------------------------------------------
# Corpus index: token-level, so span boundaries are token boundaries.
# We index the *actual* token sequence of each document, so verbatim copy is
# tokenization-consistent. (Cross-document span starts mid-sentence can still
# tokenize differently due to leading-space BPE; for the baseline we only let
# spans start at real corpus positions and copy token-by-token, which sidesteps
# that. Flag for later: a content-aware start would need re-tokenization care.)
# --------------------------------------------------------------------------
@dataclass
class Corpus:
    docs: list[list[int]]                       # doc_id -> token ids
    doc_names: list[str]
    start_index: dict[int, list[tuple[int, int]]] = field(default_factory=dict)

    @classmethod
    def from_texts(cls, texts: dict[str, str], tokenize: Callable[[bytes], list[int]]):
        docs, names = [], []
        for name, text in texts.items():
            names.append(name)
            docs.append(tokenize(text.encode("utf-8")))
        index: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for d, toks in enumerate(docs):
            for p, t in enumerate(toks):
                index[t].append((d, p))
        return cls(docs=docs, doc_names=names, start_index=dict(index))

    def continuation(self, doc: int, pos: int) -> Optional[int]:
        nxt = pos + 1
        return self.docs[doc][nxt] if nxt < len(self.docs[doc]) else None
