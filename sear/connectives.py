"""Build tokenizer-aligned connective inventories for ExtractiveCopyConstraint."""
from __future__ import annotations

from typing import Callable

from .corpus import Corpus

CONTRASTIVE_PHRASES = [
    " but",
    " however",
    " whereas",
    " by contrast",
    " on the other hand",
    " although",
]
ADDITIVE_PHRASES = [" and", " in addition", " meanwhile"]
CONCESSIVE_PHRASES = [" while", " or"]

DEFAULT_CONNECTIVE_PHRASES = (
    CONTRASTIVE_PHRASES + ADDITIVE_PHRASES + CONCESSIVE_PHRASES
)


def phrases_for_edge_types(edge_types: set[str]) -> list[str]:
    if "contradicts" in edge_types:
        return list(CONTRASTIVE_PHRASES)
    if "cites" in edge_types:
        return list(ADDITIVE_PHRASES) + list(CONCESSIVE_PHRASES)
    if "supersedes" in edge_types:
        return list(CONCESSIVE_PHRASES) + list(ADDITIVE_PHRASES)
    return list(DEFAULT_CONNECTIVE_PHRASES)


def connective_mode_label(edge_types: set[str]) -> str:
    if "contradicts" in edge_types:
        return "contrastive"
    if "cites" in edge_types:
        return "additive"
    if "supersedes" in edge_types:
        return "concessory"
    return "default"


def build_connective_inventory(
    tokenize: Callable[[bytes], list[int]],
    corpus: Corpus,
    phrases: list[str] | None = None,
) -> tuple[frozenset[int], dict[int, frozenset[int]], dict[int, list[int]], frozenset[int]]:
    """
    Tokenize connective phrases and build FSM tables.

    Returns:
        connective_starts, connective_continuations, connective_phrases, force_connective_ids
    """
    phrases = phrases if phrases is not None else DEFAULT_CONNECTIVE_PHRASES
    phrase_map: dict[int, list[int]] = {}
    continuations: dict[int, set[int]] = {}
    force_connective: set[int] = set()

    for phrase in phrases:
        ids = tokenize(phrase.encode("utf-8"))
        if not ids:
            continue
        if len(ids) == 1 and ids[0] in corpus.start_index:
            force_connective.add(ids[0])
            phrase_map[ids[0]] = ids
            continue
        phrase_map[ids[0]] = ids
        for i in range(len(ids) - 1):
            continuations.setdefault(ids[i], set()).add(ids[i + 1])

    starts = frozenset(phrase_map.keys())
    cont_frozen = {k: frozenset(v) for k, v in continuations.items()}
    return starts, cont_frozen, phrase_map, frozenset(force_connective)


def build_cite_ids(
    tokenize: Callable[[bytes], list[int]],
    num_sources: int,
) -> dict[int, int]:
    """Map cite-marker token ids to 0-based doc indices (all BPE tokens in each label)."""
    cite_ids, _, _ = build_cite_inventory(tokenize, num_sources)
    return cite_ids


def build_cite_inventory(
    tokenize: Callable[[bytes], list[int]],
    num_sources: int,
) -> tuple[dict[int, int], dict[int, list[int]], dict[int, frozenset[int]]]:
    """
    Build cite FSM tables for multi-token cite markers like ``[1]``.

    Returns:
        cite_ids: final cite token -> doc index (for structural mask)
        cite_sequences_by_doc: doc index -> full cite token sequence
        cite_continuations: mid-phrase continuation sets (union across docs)
    """
    cite_sequences_by_doc: dict[int, list[int]] = {}
    continuations: dict[int, set[int]] = {}

    for i in range(num_sources):
        label = f"[{i + 1}]"
        ids = tokenize(label.encode("utf-8"))
        if not ids:
            continue
        cite_sequences_by_doc[i] = ids
        for j in range(len(ids) - 1):
            continuations.setdefault(ids[j], set()).add(ids[j + 1])

    cite_ids = {
        seq[-1]: doc
        for doc, seq in cite_sequences_by_doc.items()
        if seq
    }
    cont_frozen = {k: frozenset(v) for k, v in continuations.items()}
    return cite_ids, cite_sequences_by_doc, cont_frozen
