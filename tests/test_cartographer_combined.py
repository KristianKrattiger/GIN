"""Combined register-robust relation detector.

Composes an embedding relatedness gate, an NLI propositional-contradiction
channel, and a cosine aspect band (docs/nc_cartographer_design.plan.md §6).
Deterministic via injected scorers reproducing the real embed+NLI measurements on
the 13-pair set: recall 1.0, precision 0.875, the single error being the disputed
inst_em/clim_pledges label.
"""
import pytest

from gin.cartographer import default_chunks, default_gold_pairs, evaluate
from gin.cartographer.combined import CombinedRelationProposer
from gin.cartographer.evaluation import _key
from gin.cartographer.models import LabeledChunk, Relation

# Measured all-MiniLM-L6-v2 cosine per gold pair (keyed by local id, sans ":0").
_COS = {
    frozenset({"inst_em", "grass_em"}): 0.390,
    frozenset({"inst_wf", "grass_wf"}): 0.418,
    frozenset({"inst_wa", "grass_wa"}): 0.134,
    frozenset({"disc_nw_pr", "disc_nw_complaint"}): 0.552,
    frozenset({"disc_mer_pr", "disc_mer_complaint"}): 0.415,
    frozenset({"hf_af_staff", "hf_af_tenants"}): 0.211,
    frozenset({"hf_kc_inspection", "hf_kc_tenants"}): 0.339,
    frozenset({"clim_warming1", "clim_warming2"}): 0.654,
    frozenset({"inst_wf", "inst_wf_fed"}): 0.727,
    frozenset({"inst_em", "clim_pledges"}): 0.490,
    frozenset({"inst_wf", "grass_wa"}): 0.124,
    frozenset({"disc_nw_pr", "hf_kc_inspection"}): 0.028,
    frozenset({"inst_em", "disc_mer_complaint"}): 0.024,
}
# The two pairs the real NLI rates a propositional contradiction (>= 0.5).
_NLI_HIGH = {
    frozenset({"disc_nw_pr", "disc_nw_complaint"}),
    frozenset({"inst_em", "clim_pledges"}),
}

_ID_BY_TEXT = {c.text: c.chunk_id.removesuffix(":0") for c in default_chunks()}


def _cos(a_text: str, b_text: str) -> float:
    return _COS[frozenset({_ID_BY_TEXT[a_text], _ID_BY_TEXT[b_text]})]


def _nli(a_text: str, b_text: str) -> tuple[float, float, float]:
    key = frozenset({_ID_BY_TEXT[a_text], _ID_BY_TEXT[b_text]})
    return (0.90, 0.02, 0.08) if key in _NLI_HIGH else (0.05, 0.02, 0.93)


def _proposer() -> CombinedRelationProposer:
    return CombinedRelationProposer(embed_cos=_cos, nli_scores=_nli)


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


def test_precision_and_the_single_error_is_the_disputed_pair():
    m = _metrics()
    assert m.contradicts_precision == pytest.approx(0.875)
    assert (m.tp, m.fp, m.fn) == (7, 1, 0)
    props = {
        _key(p.src_chunk_id, p.dst_chunk_id): p
        for p in _proposer().propose_over(_pairs())
    }
    # The lone false positive: inst_em/clim_pledges (gold corroborates, but the
    # NLI channel reads propositional tension — the disputed gold label).
    fp = props[_key("inst_em:0", "clim_pledges:0")]
    assert fp.relation == Relation.CONTRADICTS
    assert "nli" in fp.method


def test_gate_rejects_every_cross_topic_pair():
    props = {
        _key(p.src_chunk_id, p.dst_chunk_id): p
        for p in _proposer().propose_over(_pairs())
    }
    for a, b in (("inst_wf", "grass_wa"), ("disc_nw_pr", "hf_kc_inspection"),
                 ("inst_em", "disc_mer_complaint")):
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
