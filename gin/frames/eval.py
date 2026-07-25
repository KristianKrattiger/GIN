"""Measurement: the pre-registered bar, honest cross-validation, baselines.

The bar stays the headline gate so the comparison with the 2026-07-13 judge
sweep is apples-to-apples. But it is 14 pairs, 4 of them issue_frame, so a pass
can be luck — hence LOO alongside, and a decision rule fixed BEFORE the numbers
are seen. Precedent: calibration.leave_one_out reported 0.69 against 0.875
in-sample, and the honest number was the valuable one.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import LeaveOneOut

from gin.cartographer.escalation_eval import (
    default_calibration_sets,
    evaluate_escalation_judge,
)

from .dataset import default_text_index
from .head import train_head
from .labels import TRAINING_CLASSES

BAR_METRIC_KEYS: tuple[str, ...] = (
    "issue_frame_recall",
    "class_c_discrimination",
    "unrelated_discrimination",
    "direction_flip_count",
)

# Measured 2026-07-13 (data/eval_runs/). Reported alongside every result.
PUBLISHED_BASELINES: tuple[dict, ...] = (
    {"model": "Mistral-7B dense", "issue_frame_recall": 0.50,
     "class_c_discrimination": 0.67, "unrelated_discrimination": 0.25,
     "direction_flip_count": 7},
    {"model": "Qwen3.6-14B-A3B MoE", "issue_frame_recall": 0.25,
     "class_c_discrimination": 0.50, "unrelated_discrimination": 0.50,
     "direction_flip_count": 7},
    {"model": "Qwen2.5-14B dense", "issue_frame_recall": 0.50,
     "class_c_discrimination": 0.33, "unrelated_discrimination": 1.00,
     "direction_flip_count": 3},
    {"model": "Opus 4.8", "issue_frame_recall": 0.00,
     "class_c_discrimination": 0.67, "unrelated_discrimination": 1.00,
     "direction_flip_count": 3},
)

LOO_SUCCESS = 0.50
LOO_SUSPECT = 0.40


def bar_metrics(
    judge: Callable[[str, str], str],
    text_index: Optional[dict[str, str]] = None,
    both_directions: bool = True,
) -> dict:
    """Score a judge on the fixed escalation bar, without touching Postgres."""
    text = default_text_index() if text_index is None else text_index
    sets = default_calibration_sets()
    return evaluate_escalation_judge(
        judge,
        text,
        issue_frame_pairs=sets["issue_frame"],
        corroboration_pairs=sets["corroboration"],
        unrelated_pairs=sets["unrelated"],
        labeled_pairs=None,
        both_directions=both_directions,
    )


def bar_all_green(metrics: dict) -> bool:
    """1.0 on all three discrimination metrics and zero direction flips."""
    for key in ("issue_frame_recall", "class_c_discrimination", "unrelated_discrimination"):
        value = metrics.get(key)
        if value is None or value < 1.0:
            return False
    return metrics.get("direction_flip_count") == 0


def loo_report(
    X: np.ndarray,
    y: np.ndarray,
    kind: str = "linear",
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> dict:
    """Leave-one-out 4-way balanced accuracy, averaged across seeds.

    A single-seed number at this sample size is not trustworthy, so spread is
    reported and a result quoted from one seed is treated as unreported.
    """
    per_seed: list[float] = []
    last_predictions = None
    for seed in seeds:
        # Collected as a list, not np.empty_like(y): y is a fixed-width unicode
        # array, so assigning into it silently truncates "RELATED_UNTYPED"
        # whenever the held-out split happens to lack that class.
        held_out: list[str] = []
        for train_idx, test_idx in LeaveOneOut().split(X):
            model = train_head(X[train_idx], y[train_idx], kind=kind, seed=seed)
            held_out.append(str(model.predict(X[test_idx])[0]))
        predictions = np.array(held_out)
        per_seed.append(float(balanced_accuracy_score(y, predictions)))
        last_predictions = predictions

    class_names = [c.value for c in TRAINING_CLASSES]
    recalls = recall_score(
        y, last_predictions, labels=class_names, average=None, zero_division=0
    )
    return {
        "n": int(len(y)),
        "per_seed": per_seed,
        "balanced_accuracy_mean": float(np.mean(per_seed)),
        "balanced_accuracy_spread": float(np.max(per_seed) - np.min(per_seed)),
        "per_class_recall": {n: float(r) for n, r in zip(class_names, recalls)},
    }


def decide(bar: dict, loo_mean: float) -> str:
    """The rule, fixed in advance so it cannot be renegotiated after the fact."""
    if not bar_all_green(bar):
        return "bar_failed"
    if loo_mean >= LOO_SUCCESS:
        return "success"
    if loo_mean >= LOO_SUSPECT:
        return "success_caveated"
    return "suspect"
