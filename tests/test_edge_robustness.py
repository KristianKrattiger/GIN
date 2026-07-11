"""Reasoning-layer robustness under noisy Cartographer edges.

Characterizes how the reasoning layer's divergence gate behaves when fed each
class of bad edge an automated Cartographer could emit. Pins the current
behavior — including the one class the gate structurally cannot catch — so a
future Cartographer/Bookkeeper change that adds relation verification flips these
assertions deliberately. See docs/nc_reasoning_robustness_noisy_edges.plan.md.
"""
from gin.eval.edge_robustness import (
    EdgeNoiseClass,
    default_corpus_idf,
    default_stress_cases,
    run_stress,
)


def _summary():
    # Real two-node corpus IDF (falls back to fixture IDF in a stripped checkout).
    return run_stress(default_stress_cases(), idf=default_corpus_idf())


def _by_id(summary):
    return {r.id: r for r in summary.results}


def test_true_contradictions_are_retained():
    """The gate must not pass the stress test by rejecting everything."""
    summary = _summary()
    assert summary.true_positive_retention == 1.0
    for r in summary.results:
        if r.noise_class == EdgeNoiseClass.TRUE_CONTRADICTION:
            assert r.gate_forced_divergent, f"{r.id} should stay divergent"


def test_irrelevant_partner_edges_are_rejected():
    """An off-query partner drops the pair back to convergent (relevance gate)."""
    results = _by_id(_summary())
    for cid in ("a_wildfire_water", "a_emissions_water"):
        assert not results[cid].gate_forced_divergent


def test_dangling_anchor_is_inert():
    """An endpoint absent from retrieval cannot force divergent mode."""
    assert not _by_id(_summary())["d_dangling"].gate_forced_divergent


def test_mislabeled_corroboration_is_the_uncaught_gap():
    """KNOWN GAP: the gate is a relevance filter, not a relation-type verifier.

    Two agreeing, on-query chunks mistyped as ``contradicts`` both pass the
    relevance gate, so the reasoning layer is driven into spurious divergent
    mode. This is unrecoverable at read time from relevance alone — it is the
    concrete argument for Cartographer edge-precision + Bookkeeper semantic
    admission. When that verification lands, flip this assertion.
    """
    result = _by_id(_summary())["c_wildfire_agree"]
    assert result.gate_forced_divergent is True
    assert result.correct is False


def test_headline_rates_match_current_taxonomy():
    summary = _summary()
    # 3 of 4 should-reject classes caught; the miss is mislabeled corroboration.
    assert summary.noise_rejection_rate == 0.75
    assert summary.true_positive_retention == 1.0


def test_idf_fallback_runs_without_corpus_json():
    """Harness still produces a summary if the corpus JSON is unavailable."""
    idf = default_corpus_idf(paths=[])  # force the fixture-text fallback
    summary = run_stress(default_stress_cases(), idf=idf)
    assert summary.true_positive_retention == 1.0
