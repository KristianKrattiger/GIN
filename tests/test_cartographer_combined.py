"""Combined register-robust relation detector.

Composes an embedding relatedness gate, an NLI propositional-contradiction
channel, and a cosine aspect band (docs/nc_cartographer_design.plan.md §6).
Deterministic via injected scorers reproducing the real embed+NLI measurements on
the 13-pair set: recall 1.0, precision 0.875, the single error being the disputed
inst_em/clim_pledges label.
"""
import pytest

from gin.cartographer import default_chunks, default_gold_pairs, evaluate
from gin.cartographer.calibration import default_samples
from gin.cartographer.combined import CombinedRelationProposer, Thresholds
from gin.cartographer.evaluation import _key
from gin.cartographer.labeled_set import gold
from gin.cartographer.models import LabeledChunk, Relation

# Measured all-MiniLM-L6-v2 cosine per original gold pair (keyed by local id).
_COS_BASE = {
    frozenset({"inst_em", "grass_em"}): 0.390,
    frozenset({"inst_wf", "grass_wf"}): 0.418,
    frozenset({"inst_wa", "grass_wa"}): 0.200,
    frozenset({"disc_nw_pr", "disc_nw_complaint"}): 0.552,
    frozenset({"disc_mer_pr", "disc_mer_complaint"}): 0.415,
    frozenset({"hf_af_staff", "hf_af_tenants"}): 0.211,
    frozenset({"hf_kc_inspection", "hf_kc_tenants"}): 0.339,
    frozenset({"clim_warming1", "clim_warming2"}): 0.654,
    frozenset({"inst_wf", "inst_wf_fed"}): 0.727,
    frozenset({"inst_em", "clim_pledges"}): 0.620,
    frozenset({"inst_wf", "grass_wa"}): 0.080,
    frozenset({"disc_nw_pr", "hf_kc_inspection"}): 0.028,
    frozenset({"inst_em", "disc_mer_complaint"}): 0.024,
}


def _local_key(src_chunk_id: str, dst_chunk_id: str) -> frozenset:
    return frozenset({src_chunk_id.removesuffix(":0"), dst_chunk_id.removesuffix(":0")})


def _cos_table() -> dict[frozenset, float]:
    table = dict(_COS_BASE)
    for (src, dst, _rel, _reg), sample in zip(gold(), default_samples()):
        table.setdefault(_local_key(src, dst), sample.cos)
    return table


_COS = _cos_table()
_NLI_BY_KEY = {
    _local_key(src, dst): sample.p_contra
    for (src, dst, _rel, _reg), sample in zip(gold(), default_samples())
}
_NLI_HIGH = {
    k for k, p in _NLI_BY_KEY.items() if p >= 0.5
}

_ID_BY_TEXT = {c.text: c.chunk_id.removesuffix(":0") for c in default_chunks()}


def _cos(a_text: str, b_text: str) -> float:
    return _COS[frozenset({_ID_BY_TEXT[a_text], _ID_BY_TEXT[b_text]})]


def _nli(a_text: str, b_text: str) -> tuple[float, float, float]:
    key = frozenset({_ID_BY_TEXT[a_text], _ID_BY_TEXT[b_text]})
    p_contra = _NLI_BY_KEY.get(key, 0.05)
    if p_contra >= 0.5:
        return (p_contra, 0.02, 0.08)
    return (p_contra, 0.02, 0.93)


def _proposer() -> CombinedRelationProposer:
    # Use default Thresholds (not JSON file) so tests stay stable with injected scorers.
    return CombinedRelationProposer(
        embed_cos=_cos,
        nli_scores=_nli,
        thresholds=Thresholds(),
    )


def _pairs():
    by_id = {c.chunk_id: c for c in default_chunks()}
    return [(by_id[g.src_chunk_id], by_id[g.dst_chunk_id]) for g in default_gold_pairs()]


def _metrics():
    return evaluate(_proposer().propose_over(_pairs()), default_gold_pairs())


def test_recall_is_perfect_across_registers():
    m = _metrics()
    assert m.contradicts_recall == 1.0
    for reg in ("legal", "housing", "climate"):
        assert m.by_register[reg]["recall"] == 1.0


def test_precision_on_core_contradicts():
    m = _metrics()
    assert m.contradicts_recall == 1.0
    assert m.contradicts_precision is not None and m.contradicts_precision >= 0.85
    assert m.fn == 0


def test_gate_rejects_every_cross_topic_pair():
    props = {
        _key(p.src_chunk_id, p.dst_chunk_id): p
        for p in _proposer().propose_over(_pairs())
    }
    for a, b in (
        ("inst_wf", "grass_wa"),
        ("disc_nw_pr", "hf_kc_inspection"),
        ("inst_em", "disc_mer_complaint"),
    ):
        p = props[_key(f"{a}:0", f"{b}:0")]
        assert p.relation == Relation.UNRELATED
        assert p.method.endswith("gate")


def test_nli_channel_catches_a_high_similarity_contradiction():
    """Legal Northwind: cos 0.552 (corroborate band) but a real contradiction —
    NLI priority over the band is what recovers it."""
    props = {
        _key(p.src_chunk_id, p.dst_chunk_id): p
        for p in _proposer().propose_over(_pairs())
    }
    nw = props[_key("disc_nw_pr:0", "disc_nw_complaint:0")]
    assert nw.relation == Relation.CONTRADICTS
    assert nw.method.endswith("nli")


def test_cosine_band_types_clear_corroborations():
    props = {
        _key(p.src_chunk_id, p.dst_chunk_id): p
        for p in _proposer().propose_over(_pairs())
    }
    for pair in (("clim_warming1", "clim_warming2"), ("inst_wf", "inst_wf_fed")):
        p = props[_key(f"{pair[0]}:0", f"{pair[1]}:0")]
        assert p.relation == Relation.CORROBORATES
        assert p.method.endswith("band")


def test_nli_channel_has_priority_over_corroborate_band():
    """A highly similar pair that NLI calls contradictory types as contradicts."""
    a = LabeledChunk("a:0", "same wording here")
    b = LabeledChunk("b:0", "same wording here")
    prop = CombinedRelationProposer(
        embed_cos=lambda x, y: 0.95, nli_scores=lambda x, y: (0.9, 0.0, 0.1)
    )
    assert prop.type_relation(a.text, b.text)[0] == Relation.CONTRADICTS
