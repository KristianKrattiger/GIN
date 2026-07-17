"""Canonical data types for the GIN corpus tier."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal, Optional
from uuid import UUID


class EvalLayer(str, Enum):
    REALISM = "realism"
    COUNTERFACTUAL = "counterfactual"
    OUT_OF_SCOPE = "out_of_scope"
    CONVERGENT = "convergent"


class EdgeType(str, Enum):
    CITES = "cites"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    TRANSLATED_FROM = "translated_from"


SynthesisMode = Literal["convergent", "divergent"]


@dataclass
class DocumentDraft:
    doc_id: str
    outlet: str
    title: str
    eval_layer: EvalLayer
    source_uri: str = ""
    source_type: str = "synthetic"
    chunks: list[str] = field(default_factory=list)
    eval_tag: Optional[str] = None
    domain: str = ""


@dataclass
class ChunkDraft:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    eval_layer: EvalLayer
    head_sentence: str = ""
    eval_tag: Optional[str] = None
    content_hash: str = ""


@dataclass
class EdgeDraft:
    src_chunk_id: str
    dst_chunk_id: str
    edge_type: EdgeType
    note: str = ""


@dataclass
class EdgeRecord:
    src_chunk_id: str
    dst_chunk_id: str
    edge_type: str
    note: Optional[str] = None
    src_anchor: Optional[tuple[int, int]] = None
    dst_anchor: Optional[tuple[int, int]] = None


@dataclass
class DocumentRecord:
    doc_id: UUID
    content_hash: str
    source_uri: str
    source_type: str
    outlet: str
    title: str
    ingested_at: datetime


@dataclass
class ChunkHit:
    chunk_id: str
    doc_id: UUID
    text: str
    head_sentence: str
    eval_layer: str
    eval_tag: Optional[str]
    content_hash: str
    outlet: str
    title: str
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rrf_score: float = 0.0


@dataclass
class SynthesisBundle:
    hits: list[ChunkHit]
    edges: list[EdgeRecord]
    mode: SynthesisMode
    pairs: list[tuple[ChunkHit, ChunkHit, EdgeRecord]] = field(default_factory=list)


@dataclass
class SynthesisContext:
    """Maps SEAR doc index to retrieval metadata for prompting and rendering."""
    doc_index_to_hit: dict[int, ChunkHit]
    cite_index_to_doc: dict[int, int]  # cite label [1] -> doc index
    mode: SynthesisMode
    edges: list[EdgeRecord] = field(default_factory=list)
    required_doc_groups: list[frozenset[int]] = field(default_factory=list)
    preferred_starts: set[tuple[int, int]] = field(default_factory=set)
    ranked_sentence_starts: list[tuple[int, int, float]] = field(default_factory=list)
    divergence_starts: dict[int, set[int]] = field(default_factory=dict)
    forbidden_starts: set[tuple[int, int]] = field(default_factory=set)
    divergence_sentence_ends: dict[int, dict[int, int]] = field(default_factory=dict)
    connective_starts: frozenset[int] = field(default_factory=frozenset)
    connective_continuations: dict[int, frozenset[int]] = field(default_factory=dict)
    connective_phrases: dict[int, list[int]] = field(default_factory=dict)
    force_connective_ids: frozenset[int] = field(default_factory=frozenset)
    active_edge_types: set[str] = field(default_factory=set)
    retrieval_manifest_hash: str = ""
    top_doc_idx: Optional[int] = None


def doc_index_to_chunk_id(ctx: SynthesisContext) -> dict[int, str]:
    return {i: hit.chunk_id for i, hit in ctx.doc_index_to_hit.items()}
