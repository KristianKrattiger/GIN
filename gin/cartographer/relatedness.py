"""Cheap relatedness gate — stage 1 of the Cartographer.

Cuts the O(n²) pair space to candidate pairs worth the expensive relation-type
stage, and emits ``unrelated`` assessments (stored negatives) for the rest. This
stage MAY use the relevance signals the rest of the system has (IDF-weighted token
overlap here); the *relation-type* detector may not (design §2). Lexical/entity
first for a deterministic, model-free harness; embeddings are the production
upgrade. See docs/nc_cartographer_design.plan.md.
"""
from __future__ import annotations

import re
from itertools import combinations
from typing import Callable, Iterable

from gin.corpus.relevance import _norm_tokens, _normalize_token, corpus_idf

from .models import Assessment, LabeledChunk, Relation

# Overlap-coefficient floor: shared IDF mass as a fraction of the lighter chunk's
# mass. A distinctive shared entity ("wildfire") clears it; incidental generic
# overlap does not. Symmetric, so relatedness has no direction.
DEFAULT_RELATEDNESS_FLOOR = 0.20

# Same-story tier: a pair must share at least this many rare tokens (story
# entities) before the relation detector may type a divergence from the band.
# Measured on the 136-chunk scan corpus (run 20260712T074956Z): every
# recoverable gold contradicts pair shares >= 2 rare tokens, 22/24 sampled
# false positives share <= 1.
DEFAULT_STORY_FLOOR = 2


def _doc_freq(corpus_texts: Iterable[str]) -> dict[str, int]:
    df: dict[str, int] = {}
    for text in corpus_texts:
        for tok in _norm_tokens(text):
            df[tok] = df.get(tok, 0) + 1
    return df


def _rare_df_ceiling(n_docs: int) -> int:
    """A token is 'rare' when it appears in at most this many corpus chunks."""
    return max(2, n_docs // 30)


def shared_rare_token_count(
    a_text: str,
    b_text: str,
    corpus_texts: list[str],
    *,
    df_ceiling: int | None = None,
) -> int:
    """Number of corpus-rare tokens (story entities) the two texts share."""
    df = _doc_freq(corpus_texts)
    ceiling = df_ceiling if df_ceiling is not None else _rare_df_ceiling(len(corpus_texts))
    shared = _norm_tokens(a_text) & _norm_tokens(b_text)
    return sum(1 for tok in shared if df.get(tok, 0) <= ceiling)


_ANCHOR_WORD = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_END = re.compile(r"[.!?]\s*$")

# Calendar words are never entity-grade. anchor_tokens' test for a proper noun
# is mid-sentence capitalization, which every weekday and month in English
# prose satisfies -- so a date was anchoring stories to each other. Measured on
# the 24 node5 labels (2026-07-26): "Monday" was the sole anchor holding a
# hospital outbreak to a bridge closure.
#
# Three of these are also ordinary English words: "may" (modal), "march"
# (verb), "august" (adjective). Excluding them costs the anchor signal in a
# story genuinely named for one -- a March on city hall. Accepted, on two
# grounds: this removes only ANCHOR-grade status, not the token's
# rare-shared-token contribution, so such a pair can still reach story_floor on
# its other entities; and the lowercase homographs are common enough that their
# document frequency puts them above the rare ceiling in any real corpus, so
# they were rarely anchoring anything.
CALENDAR_WORDS = frozenset(
    "monday tuesday wednesday thursday friday saturday sunday "
    "january february march april may june july august september october "
    "november december".split()
)


def anchor_tokens(text: str) -> set[str]:
    """Normalized tokens with at least one entity-grade occurrence in ``text``.

    An occurrence is entity-grade when it is a mid-sentence capitalized word
    (proper noun), an all-caps word (dateline), or a multi-digit number (story
    figure). Sentence-initial capitalization and decimal fragments carry no
    entity signal — corpus-rare boilerplate ('remain in effect', 'Combined
    reservoir storage...') drove the residual scan false positives
    (run 20260712T091415Z).

    DRIFT POINTER: `_scan()` in `scripts/sweep_same_story.py` mirrors this
    function's classification logic (entity_grade) to expose extra signals for
    the anchor-mode sweep. Editing this function's classification without
    updating that mirror invalidates the sweep's numbers silently.
    """
    out: set[str] = set()
    for m in _ANCHOR_WORD.finditer(text):
        word = m.group(0)
        before = text[: m.start()]
        sentence_initial = (
            not before.strip()
            or bool(_SENTENCE_END.search(before))
            or before.endswith("\n")
        )
        entity_grade = (
            (word.isdigit() and len(word) >= 2)
            or (len(word) > 2 and word.isupper())
            or (word[0].isupper() and not word.isupper() and not sentence_initial)
        )
        if entity_grade:
            token = _normalize_token(word.lower())
            if token not in CALENDAR_WORDS:
                out.add(token)
    return out


def make_same_story(
    corpus_texts: list[str],
    *,
    story_floor: int = DEFAULT_STORY_FLOOR,
    df_ceiling: int | None = None,
    require_anchor: bool = True,
) -> "Callable[[str, str], bool]":
    """Stage-1 same-story predicate: do the two chunks cover one story?

    True when the pair shares >= ``story_floor`` corpus-rare tokens, at least
    one of which is entity-grade (see ``anchor_tokens``). Precomputes document
    frequencies once; the returned callable is the injectable story signal for
    the relation detector's contradicts channels.
    """
    df = _doc_freq(corpus_texts)
    ceiling = df_ceiling if df_ceiling is not None else _rare_df_ceiling(len(corpus_texts))

    def same_story(a_text: str, b_text: str) -> bool:
        shared = _norm_tokens(a_text) & _norm_tokens(b_text)
        rare = {tok for tok in shared if df.get(tok, 0) <= ceiling}
        if len(rare) < story_floor:
            return False
        if not require_anchor:
            return True
        return bool((anchor_tokens(a_text) | anchor_tokens(b_text)) & rare)

    return same_story


def _idf_mass(text: str, idf: dict[str, float]) -> float:
    return sum(idf.get(t, 0.0) for t in _norm_tokens(text))


def idf_relatedness(a_text: str, b_text: str, idf: dict[str, float]) -> float:
    """Symmetric IDF-weighted overlap coefficient in [0, 1].

    shared_mass / min(mass_a, mass_b): rewards distinctive shared tokens, and is
    stable when the two chunks differ greatly in length (a short grassroots line
    vs. a long institutional paragraph).
    """
    a_tokens, b_tokens = _norm_tokens(a_text), _norm_tokens(b_text)
    shared = a_tokens & b_tokens
    shared_mass = sum(idf.get(t, 0.0) for t in shared)
    a_mass = sum(idf.get(t, 0.0) for t in a_tokens)
    b_mass = sum(idf.get(t, 0.0) for t in b_tokens)
    denom = min(a_mass, b_mass)
    if denom <= 0:
        return 0.0
    return min(1.0, shared_mass / denom)


class RelatednessGate:
    """Stage 1: partition chunk pairs into candidates vs. stored negatives."""

    def __init__(
        self,
        chunks: Iterable[LabeledChunk],
        *,
        floor: float = DEFAULT_RELATEDNESS_FLOOR,
        idf_corpus: Iterable[str] | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.floor = floor
        texts = list(idf_corpus) if idf_corpus is not None else [c.text for c in self.chunks]
        self.idf = corpus_idf(texts)

    def assess_pair(self, a: LabeledChunk, b: LabeledChunk) -> Assessment:
        score = idf_relatedness(a.text, b.text, self.idf)
        related = score >= self.floor
        return Assessment(
            src_chunk_id=a.chunk_id,
            dst_chunk_id=b.chunk_id,
            relation=Relation.RELATED_UNTYPED if related else Relation.UNRELATED,
            method="relatedness_gate:idf_overlap",
            confidence=score,
            rationale=(
                f"idf overlap {score:.3f} "
                f"{'>=' if related else '<'} floor {self.floor:.2f}"
            ),
        )

    def assess_all(self) -> list[Assessment]:
        """One assessment per unordered pair — negatives included (stored)."""
        return [
            self.assess_pair(a, b) for a, b in combinations(self.chunks, 2)
        ]

    def candidates(self) -> list[Assessment]:
        return [a for a in self.assess_all() if a.relation == Relation.RELATED_UNTYPED]

    def negatives(self) -> list[Assessment]:
        return [a for a in self.assess_all() if a.relation == Relation.UNRELATED]
