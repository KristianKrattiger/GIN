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


def test_story_blocked_pair_falls_back_to_nli_p_contra():
    # F1 part 2: type_relation story-blocks the NLI channel for not-same-story
    # pairs (combined.py), so its own evidence dict never carries p_contra for
    # them — but that is exactly the population every residue pair belongs to.
    # The signal panel must not go blind for the very pairs the residue's
    # ranking floated on the strength of a contradiction; fall back to the
    # same cross-encoder call directly.
    prop = CombinedRelationProposer(
        embed_cos=lambda a, b: 0.70,
        same_story=lambda a, b: False,
        nli_scores=lambda a, b: (0.9, 0.0, 0.1),
    )
    sig = pair_signals("x", "y", prop)
    assert sig["same_story"] is False
    assert sig["nli_p_contra"] == 0.9
