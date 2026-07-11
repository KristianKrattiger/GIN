"""Cartographer design harness — relatedness gate, anti-pattern proposer, and
independent edge-precision measurement.

Step 3 of docs/nc_reasoning_robustness_noisy_edges.plan.md
(docs/nc_cartographer_design.plan.md). Pins the central principle carried from
steps 1–2: relatedness is not relation. Deterministic, no DB/model.
"""
import pytest

from gin.cartographer import (
    EdgeProposal,
    RelatednessGate,
    RelatednessProposer,
    Relation,
    default_chunks,
    default_gold_pairs,
    evaluate,
    idf_relatedness,
)
from gin.cartographer.evaluation import _key


def _gate():
    return RelatednessGate(default_chunks())


def _by_id():
    return {c.chunk_id: c for c in default_chunks()}


def test_relatedness_gate_stores_negatives_for_unrelated_pairs():
    """A cross-topic pair is assessed 'unrelated' — stored, not silent."""
    chunks = _by_id()
    gate = _gate()
    asmt = gate.assess_pair(chunks["inst_wf:0"], chunks["grass_wa:0"])
    assert asmt.relation == Relation.UNRELATED
    # Negatives are first-class graph content.
    assert any(n.relation == Relation.UNRELATED for n in gate.negatives())


def test_relatedness_ranks_agreeing_pair_above_a_real_contradiction():
    """The load-bearing illustration: relatedness is not relation.

    Lexical overlap scores the two agreeing wildfire statistics as MORE related
    than a genuine institutional-vs-grassroots contradiction — so relatedness
    cannot stand in for relation typing, and cannot even reliably gate cross-
    register contradictions (motivating embedding-based relatedness).
    """
    chunks = _by_id()
    idf = _gate().idf
    agreeing = idf_relatedness(
        chunks["inst_wf:0"].text, chunks["inst_wf_fed:0"].text, idf
    )
    real_contradiction = idf_relatedness(
        chunks["inst_wf:0"].text, chunks["grass_wf:0"].text, idf
    )
    assert agreeing > real_contradiction


def test_anti_pattern_fails_class_c_discrimination():
    """relatedness-only mints a contradicts edge on the agreeing pair."""
    metrics = evaluate(RelatednessProposer().propose(default_chunks()), default_gold_pairs())
    assert metrics.class_c_discrimination == 0.0
    # The single false positive is exactly the corroborating (class-C) pair.
    props = {
        _key(p.src_chunk_id, p.dst_chunk_id): p.relation
        for p in RelatednessProposer().propose(default_chunks())
    }
    assert props[_key("inst_wf:0", "inst_wf_fed:0")] == Relation.CONTRADICTS


def test_anti_pattern_precision_recall_are_pinned():
    metrics = evaluate(RelatednessProposer().propose(default_chunks()), default_gold_pairs())
    assert metrics.contradicts_precision == 0.5   # emissions TP vs class-C FP
    assert metrics.contradicts_recall == pytest.approx(1 / 3)  # cross-register misses
    assert (metrics.tp, metrics.fp, metrics.fn) == (1, 1, 2)


def test_per_register_breakdown_isolates_the_class_c_register():
    metrics = evaluate(RelatednessProposer().propose(default_chunks()), default_gold_pairs())
    # Emissions is fully recovered; the wildfire register carries the class-C FP.
    assert metrics.by_register["emissions"]["precision"] == 1.0
    assert metrics.by_register["wildfire"]["precision"] == 0.0


def test_edge_proposal_rejects_negative_assessments():
    """Only typed (contradicts/corroborates/supersedes) assessments become edges."""
    negatives = [a for a in _gate().assess_all() if not a.is_typed_edge]
    assert negatives
    with pytest.raises(ValueError):
        EdgeProposal.from_assessment(negatives[0])


def test_edge_proposal_carries_optional_anchors():
    """Anchors exist on the schema now, so no migration after multi-sentence ingest."""
    contra = next(
        p for p in RelatednessProposer().propose(default_chunks())
        if p.relation == Relation.CONTRADICTS
    )
    proposal = EdgeProposal.from_assessment(contra, src_anchor=(0, 12), dst_anchor=(3, 20))
    assert proposal.src_anchor == (0, 12)
    assert proposal.relation == Relation.CONTRADICTS


def test_proposer_is_deterministic():
    a = [(p.src_chunk_id, p.dst_chunk_id, p.relation) for p in RelatednessProposer().propose(default_chunks())]
    b = [(p.src_chunk_id, p.dst_chunk_id, p.relation) for p in RelatednessProposer().propose(default_chunks())]
    assert a == b
