"""pair_signals surfaces the cheap detector's cosine / NLI / verdict for display."""
from gin.cartographer.combined import CombinedRelationProposer
from gin.curator.signals import pair_signals


def _proposer(cos, p_contra):
    # Injected scorers make this model-free (same seam combined.py's own tests use).
    return CombinedRelationProposer(
        embed_cos=lambda a, b: cos,
        nli_scores=lambda a, b: (p_contra, 0.0, 1.0 - p_contra),
    )


def test_gated_pair_reports_cosine_and_no_nli():
    sig = pair_signals("x", "y", _proposer(cos=0.05, p_contra=0.9))
    assert sig["cheap_verdict"] == "unrelated"
    assert sig["cosine"] == 0.05
    assert sig["nli_p_contra"] is None  # gate short-circuits before NLI


def test_related_pair_reports_nli_p_contra():
    sig = pair_signals("x", "y", _proposer(cos=0.55, p_contra=0.9))
    assert sig["nli_p_contra"] == 0.9
    assert sig["cheap_verdict"] in {"contradicts", "corroborates", "related_untyped"}
