"""Cartographer layer — proposes typed epistemic edges; never writes graph state.

Two stages with deliberately different signals (design §2): a cheap relatedness
gate (may use IDF/overlap) and a relation-type detector (must not). The real
NLI-class relation detector is the next step; this package ships the relatedness
gate, the anti-pattern baseline, and the independent edge-precision harness that
the detector must beat. See docs/nc_cartographer_design.plan.md.
"""
from .evaluation import (
    CartographerMetrics,
    GoldPair,
    default_chunks,
    default_gold_pairs,
    evaluate,
    format_report,
)
from .models import Assessment, EdgeProposal, LabeledChunk, Relation
from .nli import NliRelationProposer
from .proposers import Proposer, RelatednessProposer
from .relatedness import RelatednessGate, idf_relatedness

__all__ = [
    "Assessment",
    "CartographerMetrics",
    "EdgeProposal",
    "GoldPair",
    "LabeledChunk",
    "NliRelationProposer",
    "Proposer",
    "Relation",
    "RelatednessGate",
    "RelatednessProposer",
    "default_chunks",
    "default_gold_pairs",
    "evaluate",
    "format_report",
    "idf_relatedness",
]
