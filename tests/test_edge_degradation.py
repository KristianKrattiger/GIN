"""Decode-in-the-loop degradation under a noisy (class-C) edge.

Step 2 of docs/nc_reasoning_robustness_noisy_edges.plan.md. Deterministic:
drives the real materialize + constrained-decode path with GreedyMaskDecoder
(faithful because SEAR's decode is constraint-determined — the real Mistral
produces byte-identical answers). Pins the load-bearing finding: a class-C edge
yields a grounded-but-wrong divergent answer that passes every existing metric.
"""
from gin.eval.edge_degradation import (
    GreedyMaskDecoder,
    default_scenarios,
    run_degradation,
)


def _results():
    return {r.id: r for r in run_degradation(GreedyMaskDecoder())}


def test_clean_agreeing_pair_stays_convergent():
    r = _results()["clean_convergent"]
    assert r.materialized_mode == "convergent"
    assert not r.spurious_divergence
    # A single institutional fact, not two joined by a divergence delimiter.
    assert r.raw_text.count("|") <= 1


def test_noisy_class_c_edge_forces_spurious_divergence():
    r = _results()["noisy_divergent"]
    assert r.materialized_mode == "divergent"
    assert r.spurious_divergence  # ground truth: the pair actually agrees
    # Both agreeing wildfire statistics are quoted as if they diverge.
    assert "56,580 wildfires" in r.raw_text
    assert "federally protected lands" in r.raw_text


def test_the_wrong_answer_passes_every_existing_metric():
    """The whole point: no current metric penalizes the spurious divergence."""
    r = _results()["noisy_divergent"]
    assert r.fabrication_rate == 0.0          # extractive decode cannot fabricate
    assert r.divergence_fidelity == 1.0       # both "sides" cited -> looks perfect
    assert r.supported_irrelevance_rate == 0.0


def test_noisy_and_real_divergence_are_metric_indistinguishable():
    """Only ground-truth relation validity separates them — not any metric.

    This is the measured argument for a divergence-*validity* signal (Cartographer
    edge-precision + Bookkeeper relation verification), since divergence-*fidelity*
    cannot tell a real contradiction from two agreeing chunks.
    """
    res = _results()
    noisy, real = res["noisy_divergent"], res["true_divergent"]
    assert noisy.materialized_mode == real.materialized_mode == "divergent"
    assert noisy.fabrication_rate == real.fabrication_rate == 0.0
    assert noisy.divergence_fidelity == real.divergence_fidelity == 1.0
    # The only difference is the ground-truth flag, which no metric observes.
    assert noisy.spurious_divergence and not real.spurious_divergence


def test_greedy_decode_is_deterministic():
    a = {r.id: r.raw_text for r in run_degradation(GreedyMaskDecoder())}
    b = {r.id: r.raw_text for r in run_degradation(GreedyMaskDecoder())}
    assert a == b


def test_scenarios_cover_the_three_edge_conditions():
    modes = {s.id: s.bundle.mode for s in default_scenarios()}
    assert modes == {
        "clean_convergent": "convergent",
        "noisy_divergent": "divergent",
        "true_divergent": "divergent",
    }
