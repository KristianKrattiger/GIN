"""Dynamic threshold calibration + leave-one-out generalization.

docs/nc_cartographer_design.plan.md §6a. Thresholds are derived from labeled
samples (not hand-picked), with max-margin tie-breaking so an edge threshold
never wins; leave-one-out quantifies the honest out-of-sample gap the small set
imposes. Deterministic (samples are baked measurements).

The historical threshold expectations below (gate 0.140 / ceiling 0.486 /
contra 0.686, LOO 0.897) are pinned against a frozen fixture — the baked
39-sample set as it existed before default_samples() switched to reading the
generated file — so they stay meaningful and model-free regardless of what the
live corpus currently contains.
"""
from pathlib import Path

from gin.cartographer.calibration import Sample, calibrate, leave_one_out
from gin.cartographer.combined import Thresholds, classify_relation
from gin.cartographer.models import Relation

FIXTURE = Path(__file__).parent / "fixtures" / "calibration_samples_fixture.json"


def _fixture_samples():
    from gin.cartographer.calibration_samples import load_samples

    samples, _ = load_samples(FIXTURE)
    return samples


def test_in_sample_calibration_recovers_a_near_perfect_split():
    samples = _fixture_samples()
    t = calibrate(samples)
    correct = sum(
        classify_relation(s.cos, s.p_contra, t, same_story=s.same_story)[0]
        == s.relation
        for s in samples
    )
    # Only the three entity-free climate register pairs are missed: their rare
    # overlap has no anchor token, so the story-gated band cannot reach them.
    # The previously disputed inst_em/clim_pledges corroboration is now correct
    # (its cross-topic NLI artifact is story-blocked).
    assert correct >= len(samples) - 3


def test_calibration_picks_central_thresholds_not_edges():
    """Max-margin calibration keeps thresholds off the sample values.

    An earlier min-margin/edge tie-break drove contra to ~0.04 (barely above a
    near-zero value) and LOO collapsed; central thresholds are the fix.
    """
    t = calibrate(_fixture_samples())
    assert 0.4 < t.contra_threshold < 0.9      # between the 0.47 and 0.90 cluster
    assert 0.45 < t.corroborate_ceiling < 0.65  # between divergent and corroborate bands


def test_leave_one_out_meets_expanded_set_target():
    """LOO on the expanded labeled set should reach the promotion target (>= 0.85).

    Recall is capped at 4/7 in-sample: the three entity-free climate register
    pairs have no anchor token in their rare overlap, so the story-gated band
    structurally cannot type them — the measured cost of ending the scan-scale
    mid-band false-positive flood. LOO may lose one more (hf_af, cos 0.211:
    holding out the lowest story-True cosine lets the max-margin gate drift
    above it), hence the 3/7 floor.
    """
    loo = leave_one_out(_fixture_samples())
    assert loo.accuracy >= 0.85
    assert loo.contradicts_recall is not None and loo.contradicts_recall >= 3 / 7
    assert loo.contradicts_precision is not None and loo.contradicts_precision >= 0.85


def test_leave_one_out_is_honestly_below_or_equal_in_sample():
    """Expose the generalization gap — LOO should not exceed in-sample accuracy."""
    samples = _fixture_samples()
    t = calibrate(samples)
    in_sample = sum(
        classify_relation(s.cos, s.p_contra, t, same_story=s.same_story)[0]
        == s.relation
        for s in samples
    ) / len(samples)
    loo = leave_one_out(samples)
    assert loo.accuracy <= in_sample + 1e-9


def test_gate_thresholds_vary_across_loo_folds():
    """Gate cut points should shift across leave-one-out folds on a small set."""
    loo = leave_one_out(_fixture_samples())
    gates = [f.gate_floor for f in loo.fold_thresholds]
    assert max(gates) - min(gates) > 0.0


def test_calibrate_is_deterministic():
    a = calibrate(_fixture_samples())
    b = calibrate(_fixture_samples())
    assert a == b


def test_empty_samples_fall_back_to_defaults():
    assert calibrate([]) == Thresholds()


def test_default_samples_reads_the_generated_file(tmp_path, monkeypatch):
    # default_samples() must read the generated file, not a baked literal.
    import json

    from gin.cartographer import calibration
    from gin.cartographer.combined import DEFAULT_EMBED_MODEL, DEFAULT_NLI_MODEL

    payload = {
        "manifest": {
            "embed_model": DEFAULT_EMBED_MODEL, "nli_model": DEFAULT_NLI_MODEL,
            "n_samples": 2, "class_counts": {"contradicts": 1, "unrelated": 1},
            "excluded_eval_pairs": 45, "git_sha": "test",
            "created_utc": "2026-07-25T00:00:00Z",
        },
        "samples": [
            {"cos": 0.9, "p_contra": 0.9, "same_story": True, "relation": "contradicts"},
            {"cos": 0.01, "p_contra": 0.01, "same_story": False, "relation": "unrelated"},
        ],
    }
    path = tmp_path / "samples.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(calibration, "DEFAULT_SAMPLES_PATH", path)
    samples = calibration.default_samples()
    assert len(samples) == 2
    assert samples[0].cos == 0.9


def test_baked_measured_literal_is_gone():
    from gin.cartographer import calibration

    assert not hasattr(calibration, "_MEASURED")
