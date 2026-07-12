"""Dynamic threshold calibration + leave-one-out generalization.

docs/nc_cartographer_design.plan.md §6a. Thresholds are derived from labeled
samples (not hand-picked), with max-margin tie-breaking so an edge threshold
never wins; leave-one-out quantifies the honest out-of-sample gap the small set
imposes. Deterministic (samples are baked measurements).
"""
from gin.cartographer.calibration import (
    Sample,
    calibrate,
    default_samples,
    leave_one_out,
)
from gin.cartographer.combined import Thresholds, classify_relation
from gin.cartographer.models import Relation


def test_in_sample_calibration_recovers_a_near_perfect_split():
    samples = default_samples()
    t = calibrate(samples)
    correct = sum(
        classify_relation(s.cos, s.p_contra, t)[0] == s.relation for s in samples
    )
    # Only the disputed inst_em/clim_pledges corroboration pair is missed in-sample.
    assert correct >= len(samples) - 1


def test_calibration_picks_central_thresholds_not_edges():
    """Max-margin calibration keeps thresholds off the sample values.

    An earlier min-margin/edge tie-break drove contra to ~0.04 (barely above a
    near-zero value) and LOO collapsed; central thresholds are the fix.
    """
    t = calibrate(default_samples())
    assert 0.4 < t.contra_threshold < 0.9      # between the 0.47 and 0.90 cluster
    assert 0.45 < t.corroborate_ceiling < 0.65  # between divergent and corroborate bands


def test_leave_one_out_meets_expanded_set_target():
    """LOO on the expanded labeled set should reach the promotion target (>= 0.85)."""
    loo = leave_one_out(default_samples())
    assert loo.accuracy >= 0.85
    assert loo.contradicts_recall is not None and loo.contradicts_recall >= 0.85
    assert loo.contradicts_precision is not None and loo.contradicts_precision >= 0.85


def test_leave_one_out_is_honestly_below_or_equal_in_sample():
    """Expose the generalization gap — LOO should not exceed in-sample accuracy."""
    samples = default_samples()
    t = calibrate(samples)
    in_sample = sum(
        classify_relation(s.cos, s.p_contra, t)[0] == s.relation for s in samples
    ) / len(samples)
    loo = leave_one_out(samples)
    assert loo.accuracy <= in_sample + 1e-9


def test_gate_thresholds_vary_across_loo_folds():
    """Gate cut points should shift across leave-one-out folds on a small set."""
    loo = leave_one_out(default_samples())
    gates = [f.gate_floor for f in loo.fold_thresholds]
    assert max(gates) - min(gates) > 0.0


def test_calibrate_is_deterministic():
    a = calibrate(default_samples())
    b = calibrate(default_samples())
    assert a == b


def test_empty_samples_fall_back_to_defaults():
    assert calibrate([]) == Thresholds()
