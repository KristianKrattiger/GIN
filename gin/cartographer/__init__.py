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
from .calibration import Sample, calibrate, default_samples, leave_one_out
from .combined import CombinedRelationProposer, Thresholds
from .frame_judge import LlmFrameJudge
from .models import Assessment, EdgeProposal, LabeledChunk, Relation
from .nli import NliRelationProposer
from .proposers import Proposer, RelatednessProposer
from .relatedness import RelatednessGate, idf_relatedness

__all__ = [
    "Assessment",
    "CartographerMetrics",
    "CombinedRelationProposer",
    "EdgeProposal",
    "GoldPair",
    "LabeledChunk",
    "LlmFrameJudge",
    "NliRelationProposer",
    "Proposer",
    "Relation",
    "RelatednessGate",
    "RelatednessProposer",
    "Sample",
    "Thresholds",
    "calibrate",
    "default_chunks",
    "default_gold_pairs",
    "default_samples",
    "evaluate",
    "format_report",
    "idf_relatedness",
    "leave_one_out",
]
