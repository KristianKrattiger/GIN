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


def test_anti_pattern_mints_the_class_c_pair_as_contradicts():
    """relatedness-only types the two agreeing wildfire statistics as contradicts.

    The corroborating pair that clears the relatedness floor is minted as a
    contradicts edge — the exact class-C false positive relatedness cannot avoid.
    """
    props = {
        _key(p.src_chunk_id, p.dst_chunk_id): p.relation
        for p in RelatednessProposer().propose(default_chunks())
    }
    assert props[_key("inst_wf:0", "inst_wf_fed:0")] == Relation.CONTRADICTS


def test_anti_pattern_precision_recall_are_pinned():
    metrics = evaluate(RelatednessProposer().propose(default_chunks()), default_gold_pairs())
    # 2 divergences clear the lexical floor -> TP; two corroborations also clear
    # it -> FP; the other 5 divergences are gated out (cross-register sparse overlap).
    assert metrics.contradicts_precision == pytest.approx(0.5)
    assert metrics.contradicts_recall == pytest.approx(2 / 7)
    assert (metrics.tp, metrics.fp, metrics.fn) == (2, 2, 5)


def test_lexical_gate_under_recalls_divergence_in_every_register():
    """The under-recall is general, not a climate artifact: 5 of 7 true
    divergences across climate/legal/housing fall below the relatedness floor."""
    metrics = evaluate(RelatednessProposer().propose(default_chunks()), default_gold_pairs())
    assert metrics.contradicts_recall < 0.5
    # Housing divergences share only the place entity — both are gated out.
    assert metrics.by_register["housing"]["recall"] == 0.0


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
