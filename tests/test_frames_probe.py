"""Stage-0 gate: is DIVERGENT linearly recoverable from frozen embeddings?"""
import numpy as np

from gin.frames.probe import (
    PROBE_FLOOR,
    PROBE_PASS,
    divergent_vs_rest,
    run_probe,
)


def test_divergent_vs_rest_is_binary():
    y = np.array(["DIVERGENT", "AGREE", "UNRELATED", "RELATED_UNTYPED", "DIVERGENT"])
    assert list(divergent_vs_rest(y)) == [1, 0, 0, 0, 1]


def _separable(n=40, dim=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim))
    y = np.where(np.arange(n) % 2 == 0, "DIVERGENT", "AGREE")
    X[y == "DIVERGENT"] += 6.0  # plainly separable
    return X, y


def test_separable_data_passes():
    result = run_probe(*_separable())
    assert result.balanced_accuracy >= PROBE_PASS
    assert result.verdict == "pass"
    assert result.passed is True


def test_noise_does_not_pass():
    # Pure noise sits near 0.50 but bounces; the load-bearing claim is that it
    # never clears the gate, not that it lands in a specific band.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 6))
    y = np.where(np.arange(60) % 2 == 0, "DIVERGENT", "AGREE")
    result = run_probe(X, y)
    assert result.balanced_accuracy < PROBE_PASS
    assert result.verdict != "pass"
    assert result.passed is False


def test_verdict_bands_are_contiguous():
    assert PROBE_FLOOR < PROBE_PASS


def test_reports_counts_and_baseline():
    X, y = _separable()
    result = run_probe(X, y)
    assert result.n == 40
    assert result.n_positive == 20
    assert 0.3 <= result.baseline <= 0.7  # stratified-random sits near chance


def test_is_deterministic_for_a_seed():
    X, y = _separable()
    assert run_probe(X, y, seed=0).balanced_accuracy == run_probe(X, y, seed=0).balanced_accuracy
