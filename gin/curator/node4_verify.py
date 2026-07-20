"""Hard-gate verifier: do node4's thesis pairs reach the curator backlog?

A genuinely-opposed pair whose cosine is below the residue floor (or that reads
same-story) never surfaces; this catches that at build time so sources can be
sharpened before a human labels. Model-free under test via an injected proposer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from gin.cartographer.combined import CombinedRelationProposer
from gin.cartographer.models import LabeledChunk

from .models import pair_key
from .residue import EscalationResidueCandidateSource


@dataclass(frozen=True)
class TopicResult:
    topic: str
    pro_key: tuple[str, str]
    passed: bool
    rank: Optional[int]


def intended_thesis_pairs(documents: list[dict]) -> dict[str, tuple[str, str]]:
    """Per topic, the pair_key of each side's position-0 (thesis) chunk."""
    thesis_by_topic: dict[str, dict[str, str]] = {}
    for doc in documents:
        topic = doc["metadata"]["topic"]
        stance = doc["metadata"]["stance"]
        zero = next(c for c in doc["chunks"] if str(c["position"]) == "0")
        cid = f"{doc['doc_id']}:{zero['position']}"
        thesis_by_topic.setdefault(topic, {})[stance] = cid
    out: dict[str, tuple[str, str]] = {}
    for topic, sides in thesis_by_topic.items():
        out[topic] = pair_key(sides["pro"], sides["con"])
    return out


def verify_surfacing(
    chunks: list[LabeledChunk],
    documents: list[dict],
    proposer: CombinedRelationProposer,
) -> list[TopicResult]:
    source = EscalationResidueCandidateSource(chunks, proposer=proposer)
    surfaced = [pair_key(a.chunk_id, b.chunk_id) for a, b in source.pairs()]
    rank_of = {key: i for i, key in enumerate(surfaced)}
    results = []
    for topic, key in intended_thesis_pairs(documents).items():
        rank = rank_of.get(key)
        results.append(TopicResult(topic, key, rank is not None, rank))
    return results
