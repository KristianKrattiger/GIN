"""Dynamic threshold calibration for the combined relation detector.

The combined detector (combined.py) had thresholds hand-read off the 13-pair set —
an in-sample choice. This derives them from labeled samples instead, and, more
importantly, reports **leave-one-out cross-validated** performance so the
generalization claim is honest: each pair is predicted by thresholds fitted on the
*other* twelve. See docs/nc_cartographer_design.plan.md §6a.

Calibration works on precomputed ``(cosine, p_contra, gold)`` samples, so it is
model-free and deterministic; gathering the samples (embed + NLI) is a separate
step.
"""
from __future__ import annotations

from dataclasses import dataclass

from .combined import Thresholds, classify_relation
from .models import Relation


@dataclass(frozen=True)
class Sample:
    cos: float
    p_contra: float
    relation: Relation


# Measured all-MiniLM-L6-v2 cosine + max-direction NLI p_contra for the 13 gold
# pairs (gin/cartographer/labeled_set.py), in gold order. Baked so calibration is
# reproducible without the models; regenerate if the labeled set changes.
_MEASURED = [
    ("contradicts", 0.390, 0.068), ("contradicts", 0.418, 0.010),
    ("contradicts", 0.134, 0.006), ("contradicts", 0.552, 0.899),
    ("contradicts", 0.415, 0.473), ("contradicts", 0.211, 0.008),
    ("contradicts", 0.339, 0.003), ("corroborates", 0.654, 0.025),
    ("corroborates", 0.727, 0.006), ("corroborates", 0.490, 0.932),
    ("unrelated", 0.124, 0.050), ("unrelated", 0.028, 0.007),
    ("unrelated", 0.024, 0.004),
]


def default_samples() -> list[Sample]:
    return [Sample(cos, pc, Relation(rel)) for rel, cos, pc in _MEASURED]


def _midpoints(values: list[float]) -> list[float]:
    """Threshold candidates: just below the min, between each adjacent pair,
    and just above the max — every cut point that changes a decision."""
    xs = sorted(set(values))
    if not xs:
        return [0.0]
    cands = [xs[0] - 1e-6]
    cands += [(xs[i] + xs[i + 1]) / 2 for i in range(len(xs) - 1)]
    cands.append(xs[-1] + 1e-6)
    return cands


def _score(samples: list[Sample], t: Thresholds) -> int:
    """Number of samples whose relation the thresholds classify correctly."""
    return sum(
        1 for s in samples if classify_relation(s.cos, s.p_contra, t)[0] == s.relation
    )


def _margin(x: float, values: list[float]) -> float:
    return min(abs(x - v) for v in values) if values else 0.0


def calibrate(samples: list[Sample]) -> Thresholds:
    """Grid-search thresholds that maximize 3-way relation accuracy.

    Candidates are the cut points implied by the observed cosine / p_contra
    values. Among equally accurate threshold sets, pick the **max-margin** one —
    each threshold as far as possible from the nearest sample it separates — since
    an edge threshold (barely above one value) is the least robust choice and
    generalizes worst under leave-one-out.
    """
    if not samples:
        return Thresholds()
    cos_vals = [s.cos for s in samples]
    contra_vals = [s.p_contra for s in samples]
    cos_cands = _midpoints(cos_vals)
    contra_cands = _midpoints(contra_vals)
    best_key: tuple[int, float, float, float, float] | None = None
    best_t = Thresholds()
    for gate in cos_cands:
        for ceiling in cos_cands:
            if ceiling < gate:
                continue
            for contra in contra_cands:
                t = Thresholds(gate, ceiling, contra)
                # Total margin rewards every threshold being central in its gap;
                # a single bottlenecked threshold (the gate sits in a tiny gap)
                # would otherwise leave the others' placement arbitrary.
                margin = (
                    _margin(gate, cos_vals)
                    + _margin(ceiling, cos_vals)
                    + _margin(contra, contra_vals)
                )
                # Maximize accuracy, then total margin; -gate/-ceiling break
                # residual ties deterministically.
                key = (_score(samples, t), margin, -gate, -ceiling)
                if best_key is None or key > best_key:
                    best_key = key
                    best_t = t
    return best_t


@dataclass
class LooResult:
    predictions: list[tuple[Relation, Relation]]  # (gold, predicted)
    fold_thresholds: list[Thresholds]

    def _counts(self) -> tuple[int, int, int, int, int]:
        tp = fp = fn = corr_total = corr_ok = 0
        for gold, pred in self.predictions:
            g_c = gold == Relation.CONTRADICTS
            p_c = pred == Relation.CONTRADICTS
            tp += g_c and p_c
            fp += (not g_c) and p_c
            fn += g_c and (not p_c)
            if gold == Relation.CORROBORATES:
                corr_total += 1
                corr_ok += not p_c
        return tp, fp, fn, corr_total, corr_ok

    @property
    def contradicts_precision(self) -> float | None:
        tp, fp, _, _, _ = self._counts()
        return tp / (tp + fp) if (tp + fp) else None

    @property
    def contradicts_recall(self) -> float | None:
        tp, _, fn, _, _ = self._counts()
        return tp / (tp + fn) if (tp + fn) else None

    @property
    def class_c_discrimination(self) -> float | None:
        _, _, _, total, ok = self._counts()
        return ok / total if total else None

    @property
    def accuracy(self) -> float:
        return sum(g == p for g, p in self.predictions) / len(self.predictions)


def leave_one_out(samples: list[Sample]) -> LooResult:
    """Predict each sample with thresholds calibrated on all the others."""
    preds: list[tuple[Relation, Relation]] = []
    folds: list[Thresholds] = []
    for i, held in enumerate(samples):
        train = samples[:i] + samples[i + 1 :]
        t = calibrate(train)
        folds.append(t)
        preds.append((held.relation, classify_relation(held.cos, held.p_contra, t)[0]))
    return LooResult(preds, folds)
