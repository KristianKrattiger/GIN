"""Bar scoring, LOO across seeds, and the pre-registered decision rule."""
import numpy as np
import pytest

from gin.frames.eval import (
    BAR_METRIC_KEYS,
    PUBLISHED_BASELINES,
    bar_all_green,
    bar_metrics,
    decide,
    loo_report,
)


def _green():
    return {"issue_frame_recall": 1.0, "class_c_discrimination": 1.0,
            "unrelated_discrimination": 1.0, "direction_flip_count": 0}


def test_bar_metric_keys_match_the_spec_bar():
    assert BAR_METRIC_KEYS == (
        "issue_frame_recall", "class_c_discrimination",
        "unrelated_discrimination", "direction_flip_count",
    )


def test_all_green_requires_every_metric():
    assert bar_all_green(_green()) is True
    for key, bad in [("issue_frame_recall", 0.75), ("class_c_discrimination", 0.9),
                     ("unrelated_discrimination", 0.5), ("direction_flip_count", 1)]:
        metrics = _green() | {key: bad}
        assert bar_all_green(metrics) is False


def test_none_metric_is_not_green():
    assert bar_all_green(_green() | {"issue_frame_recall": None}) is False


def test_published_baselines_include_the_failed_judges():
    names = {row["model"] for row in PUBLISHED_BASELINES}
    assert "Qwen2.5-14B dense" in names
    assert "Opus 4.8" in names
    opus = next(r for r in PUBLISHED_BASELINES if r["model"] == "Opus 4.8")
    assert opus["issue_frame_recall"] == 0.00


def test_decision_rule_bands():
    green = _green()
    assert decide(green, 0.62) == "success"
    assert decide(green, 0.50) == "success"
    assert decide(green, 0.45) == "success_caveated"
    assert decide(green, 0.30) == "suspect"
    assert decide(_green() | {"direction_flip_count": 2}, 0.9) == "bar_failed"


def test_loo_report_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    labels = np.array(["DIVERGENT", "AGREE", "RELATED_UNTYPED", "UNRELATED"] * 10)
    for offset, name in enumerate(["DIVERGENT", "AGREE", "RELATED_UNTYPED", "UNRELATED"]):
        X[labels == name] += offset * 5.0
    report = loo_report(X, labels, kind="linear", seeds=(0, 1))
    assert set(report) == {"balanced_accuracy_mean", "balanced_accuracy_spread",
                           "per_seed", "per_class_recall", "n"}
    assert report["n"] == 40
    assert len(report["per_seed"]) == 2
    assert report["balanced_accuracy_mean"] > 0.9  # separable by construction


def test_bar_metrics_runs_db_free_with_a_stub_judge():
    # Proves the bar is scorable without Postgres.
    metrics = bar_metrics(lambda a, b: "DIVERGENT")
    assert metrics["issue_frame_recall"] == 1.0        # constant judge catches all gold
    assert metrics["class_c_discrimination"] == 0.0    # and fails every control
    assert metrics["issue_frame_scorable_count"] == 4
