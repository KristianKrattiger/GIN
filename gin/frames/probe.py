"""Stage-0 gate: linear separability of the stance axis in frozen embeddings.

Doubles as the linear baseline. If a logistic regression already clears the bar,
that IS the shipped model and no MLP is built.

Thresholds are fixed before the number is seen so the gate cannot be
renegotiated after the fact. DIVERGENT-vs-rest is binary, so chance balanced
accuracy is 0.50.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut

from .labels import FrameClass

PROBE_PASS = 0.65
PROBE_FLOOR = 0.55
_BASELINE_TRIALS = 200


@dataclass(frozen=True)
class ProbeResult:
    balanced_accuracy: float
    baseline: float
    n: int
    n_positive: int
    verdict: str  # "pass" | "inconclusive" | "fail"

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


def divergent_vs_rest(y: np.ndarray) -> np.ndarray:
    """Collapse the 4-way label vector to the binary stance axis."""
    return (y == FrameClass.DIVERGENT.value).astype(int)


def run_probe(X: np.ndarray, y: np.ndarray, seed: int = 0) -> ProbeResult:
    """Leave-one-out logistic regression on DIVERGENT-vs-rest."""
    target = divergent_vs_rest(y)
    predictions = np.empty_like(target)
    for train_idx, test_idx in LeaveOneOut().split(X):
        clf = LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=seed
        )
        clf.fit(X[train_idx], target[train_idx])
        predictions[test_idx] = clf.predict(X[test_idx])

    accuracy = float(balanced_accuracy_score(target, predictions))

    rng = np.random.default_rng(seed)
    baseline = float(
        np.mean(
            [
                balanced_accuracy_score(target, rng.permutation(target))
                for _ in range(_BASELINE_TRIALS)
            ]
        )
    )

    if accuracy >= PROBE_PASS:
        verdict = "pass"
    elif accuracy < PROBE_FLOOR:
        verdict = "fail"
    else:
        verdict = "inconclusive"

    return ProbeResult(accuracy, baseline, len(target), int(target.sum()), verdict)
