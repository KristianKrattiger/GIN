"""NLI relation detector — measured finding: NLI-contradiction ≠ GIN divergence.

docs/nc_cartographer_design.plan.md §6. Climate/housing "contradicts" pairs are
both true statements emphasizing different aspects of a shared event, so an
entailment cross-encoder rates them neutral. (On the expanded set the real model
does fire on the *legal* register, where framing divergence coincides with
propositional contradiction — recorded in the design doc; GIN "contradicts" is
heterogeneous.) These deterministic tests use an injected neutral-everywhere
scorer to pin the mapping and the degenerate neutral→zero-recall failure mode; a
synthetic propositional contradiction proves the typing logic itself works.
"""
from gin.cartographer import (
    NliRelationProposer,
    Relation,
    default_chunks,
    default_gold_pairs,
    evaluate,
)
from gin.cartographer.models import LabeledChunk

# The real cross-encoder rated every labeled pair neutral (~0.99). This stub
# reproduces that: framing-divergence and corroboration alike look neutral, while
# a genuine propositional contradiction (passed vs failed) scores high contra.
def _stub_scorer(premise: str, hypothesis: str) -> tuple[float, float, float]:
    text = (premise + " " + hypothesis).lower()
    if "passed" in text and "failed" in text:
        return (0.90, 0.02, 0.08)          # genuine propositional contradiction
    if "identical restatement" in text:
        return (0.02, 0.94, 0.04)          # genuine entailment
    return (0.03, 0.01, 0.96)              # framing divergence / corroboration: neutral


def _proposer() -> NliRelationProposer:
    return NliRelationProposer(scorer=_stub_scorer)


def _labeled_pairs():
    by_id = {c.chunk_id: c for c in default_chunks()}
    return [(by_id[g.src_chunk_id], by_id[g.dst_chunk_id]) for g in default_gold_pairs()]


def test_nli_rates_gin_framing_divergence_as_untyped():
    """The finding: no framing pair is typed contradicts — neutral, like agreement."""
    assessments = _proposer().propose_over(_labeled_pairs())
    assert all(a.relation == Relation.RELATED_UNTYPED for a in assessments)


def test_nli_perfect_discrimination_but_zero_recall():
    """A neutral-everywhere signal 'wins' class-C only by detecting nothing."""
    metrics = evaluate(_proposer().propose_over(_labeled_pairs()), default_gold_pairs())
    assert metrics.class_c_discrimination == 1.0   # never mints a corroborating pair
    assert metrics.contradicts_recall == 0.0       # but never mints a real one either
    assert (metrics.tp, metrics.fp, metrics.fn) == (0, 0, 7)


def test_typing_logic_fires_on_a_genuine_propositional_contradiction():
    """Proves the detector works — the framing miss is a signal property, not a bug."""
    a = LabeledChunk("x:0", "The harbor district referendum passed by 842 votes.")
    b = LabeledChunk("y:0", "The harbor district referendum failed to reach a majority.")
    relation, ev = _proposer().type_relation(a.text, b.text)
    assert relation == Relation.CONTRADICTS
    assert ev["p_contra"] >= 0.5


def test_typing_logic_detects_entailment_as_corroborates():
    a = LabeledChunk("x:0", "Turnout was 61 percent.")
    b = LabeledChunk("y:0", "An identical restatement: turnout reached 61 percent.")
    relation, _ = _proposer().type_relation(a.text, b.text)
    assert relation == Relation.CORROBORATES


def test_relation_typing_is_symmetric_and_deterministic():
    a, b = _labeled_pairs()[0]
    r1, _ = _proposer().type_relation(a.text, b.text)
    r2, _ = _proposer().type_relation(b.text, a.text)
    assert r1 == r2
