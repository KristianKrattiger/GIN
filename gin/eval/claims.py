"""Claim data contract and segmentation for the eval harness.

A ``RawClaim`` is what an arm emits: a piece of output text plus whatever
source chunks the arm claims/knows it drew on and a span type. The verifier
then turns each ``RawClaim`` into a scored ``ClaimRecord`` carrying:

- ``verdict``          SUPPORTED / UNSUPPORTED / REFUSAL
- ``matched_chunk_id`` the chunk the claim was grounded to (if SUPPORTED)
- ``score``            NLI entailment confidence or overlap similarity
- ``node_scope``       WITHIN_NODE / CROSS_NODE / NONE (federation boundary check)
- ``span_type``        EXACT / AMBIGUOUS today; INFERRED / PARAPHRASE reserved
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Callable, Optional

from sear.processor import Segment


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    REFUSAL = "REFUSAL"


class NodeScope(str, Enum):
    WITHIN_NODE = "WITHIN_NODE"
    CROSS_NODE = "CROSS_NODE"
    NONE = "NONE"


class SpanType(str, Enum):
    EXACT = "EXACT"
    AMBIGUOUS = "AMBIGUOUS"
    GENERATED = "GENERATED"  # unconstrained RAG text, pre-verification
    PARAPHRASE = "PARAPHRASE"  # reserved for Flagged Generation
    INFERRED = "INFERRED"  # reserved for Flagged Generation
    REFUSAL = "REFUSAL"


@dataclass
class RawClaim:
    """Arm output before verification."""

    text: str
    span_type: str = SpanType.GENERATED.value
    # Sources the arm knows (No-Continuation) or claims via [n] (RAG).
    cited_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class ClaimRecord:
    """A verified, scored claim — the core data contract of the harness."""

    claim_text: str
    verdict: str
    matched_chunk_id: Optional[str]
    score: float
    node_scope: str
    span_type: str
    cited_chunk_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --- Node attribution -------------------------------------------------------

NodeOf = Callable[[str], str]


def node_scope_for(chunk_ids: list[str], node_of: NodeOf) -> NodeScope:
    """Classify a claim's grounding as within a single node or across nodes."""
    nodes = {node_of(cid) for cid in chunk_ids if cid}
    if not nodes:
        return NodeScope.NONE
    return NodeScope.WITHIN_NODE if len(nodes) == 1 else NodeScope.CROSS_NODE


# --- Segmentation: No-Continuation (SEAR extract segments) ------------------


def segments_to_raw_claims(
    segments: list[Segment],
    detok: Callable[[list[int]], str],
    doc_index_to_chunk_id: dict[int, str],
) -> list[RawClaim]:
    """One RawClaim per extract Segment; connectives and cites are skipped.

    Span type is EXACT for a single surviving source and AMBIGUOUS when the
    verbatim span matched multiple docs (shared-prefix ledes).
    """
    claims: list[RawClaim] = []
    for seg in segments:
        if seg.kind != "extract":
            continue
        text = detok(seg.token_ids).strip()
        if not text:
            continue
        source_ids = [
            doc_index_to_chunk_id[d]
            for (d, _s, _e) in seg.sources
            if d in doc_index_to_chunk_id
        ]
        span_type = (
            SpanType.AMBIGUOUS.value if len(seg.sources) > 1 else SpanType.EXACT.value
        )
        claims.append(
            RawClaim(text=text, span_type=span_type, cited_chunk_ids=source_ids)
        )
    return claims


# --- Segmentation: RAG (free-form generated text) ---------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CITE_BLOCK = re.compile(r"\[([^\]]+)\]")
_CITE_PART_INDEX = re.compile(r"^(\d+)")


def split_sentences(text: str) -> list[str]:
    """Crude sentence-level split; refine toward proposition-level later."""
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def parse_citation_indices(cite_block: str) -> list[int]:
    """Extract source indices from the interior of a ``[...]`` citation marker."""
    indices: list[int] = []
    seen: set[int] = set()
    for part in cite_block.split(","):
        part = part.strip()
        match = _CITE_PART_INDEX.match(part)
        if not match:
            continue
        idx = int(match.group(1))
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return indices


def extract_citation_indices(sentence: str) -> list[int]:
    """Collect unique source indices from all citation blocks in a sentence."""
    indices: list[int] = []
    seen: set[int] = set()
    for block in _CITE_BLOCK.findall(sentence):
        for idx in parse_citation_indices(block):
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
    return indices


def strip_citation_markers(sentence: str) -> str:
    """Remove ``[n]``, ``[n: chunk_id]``, and ``[n, m]`` style markers."""
    return re.sub(r"\[[^\]]+\]", "", sentence).strip()


def rag_text_to_raw_claims(
    text: str,
    cite_index_to_chunk_id: dict[int, str],
) -> list[RawClaim]:
    """Split RAG output into sentence claims, parsing citation markers."""
    claims: list[RawClaim] = []
    for sentence in split_sentences(text):
        cited = [
            cite_index_to_chunk_id[idx]
            for idx in extract_citation_indices(sentence)
            if idx in cite_index_to_chunk_id
        ]
        cleaned = strip_citation_markers(sentence)
        if not cleaned:
            continue
        claims.append(
            RawClaim(
                text=cleaned,
                span_type=SpanType.GENERATED.value,
                cited_chunk_ids=cited,
            )
        )
    return claims
