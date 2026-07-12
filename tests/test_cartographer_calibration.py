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
    assert correct == 12  # only the disputed inst_em/clim_pledges pair is missed


def test_calibration_picks_central_thresholds_not_edges():
    """Max-margin calibration keeps thresholds off the sample values.

    An earlier min-margin/edge tie-break drove contra to ~0.04 (barely above a
    near-zero value) and LOO collapsed; central thresholds are the fix.
    """
    t = calibrate(default_samples())
    assert 0.4 < t.contra_threshold < 0.9      # between the 0.47 and 0.90 cluster
    assert 0.45 < t.corroborate_ceiling < 0.65  # between divergent and corroborate bands


def test_leave_one_out_is_honestly_below_in_sample():
    """The point of calibration: expose the generalization gap on 13 pairs."""
    loo = leave_one_out(default_samples())
    # In-sample accuracy is 12/13 = 0.92; LOO is materially lower.
    assert loo.accuracy < 0.85
    assert loo.contradicts_recall is not None and loo.contradicts_recall < 1.0
    # Still well above chance (3 classes) — the architecture holds; thresholds
    # need more data.
    assert loo.accuracy > 0.5


def test_gate_is_the_fragile_threshold():
    """The water divergence (cos 0.134) sits one hundredth above the unrelated
    ceiling (0.124), so some folds misgate it — the documented fragility."""
    loo = leave_one_out(default_samples())
    gates = [f.gate_floor for f in loo.fold_thresholds]
    # Folds disagree on the gate across the 0.124/0.134 boundary.
    assert max(gates) - min(gates) > 0.0
    errors = [(g, p) for g, p in loo.predictions if g != p]
    assert any(
        g == Relation.CONTRADICTS and p == Relation.UNRELATED for g, p in errors
    )


def test_calibrate_is_deterministic():
    a = calibrate(default_samples())
    b = calibrate(default_samples())
    assert a == b


def test_empty_samples_fall_back_to_defaults():
    assert calibrate([]) == Thresholds()
