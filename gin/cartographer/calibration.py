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
    # Stage-1 same-story signal (relatedness.make_same_story over the labeled
    # corpus): does the pair share >= 2 corpus-rare tokens? Calibration feeds
    # the classifier the signal it will actually receive at scan time.
    same_story: bool = False


# Measured all-MiniLM-L6-v2 cosine + max-direction NLI p_contra + lexical
# same-story for the gold pairs (gin/cartographer/labeled_set.py), in gold
# order. Baked so calibration is reproducible without the models; regenerate if
# the labeled set changes. The three same_story=False contradicts pairs are the
# climate register divergences: their rare overlap is entity-free boilerplate
# ('greenhouse gas', 'roughly') with no anchor token, out of reach of the story
# tier — the honest recall cost of the scan-scale precision fix.
_MEASURED = [
    ("contradicts", 0.390, 0.068, False), ("contradicts", 0.418, 0.010, False),
    ("contradicts", 0.200, 0.006, False), ("contradicts", 0.552, 0.899, True),
    ("contradicts", 0.415, 0.473, True), ("contradicts", 0.211, 0.008, True),
    ("contradicts", 0.339, 0.003, True), ("corroborates", 0.654, 0.025, False),
    ("corroborates", 0.727, 0.006, False), ("corroborates", 0.620, 0.020, False),
    ("unrelated", 0.080, 0.050, False), ("unrelated", 0.028, 0.007, False),
    ("unrelated", 0.024, 0.004, False),
    # Expanded set — corroborates (high cosine, low NLI contra).
    ("corroborates", 0.680, 0.012, False), ("corroborates", 0.705, 0.015, False),
    ("corroborates", 0.695, 0.011, False), ("corroborates", 0.710, 0.014, False),
    ("corroborates", 0.665, 0.010, False), ("corroborates", 0.672, 0.009, False),
    ("corroborates", 0.640, 0.008, False),
    # Expanded set — unrelated (low cosine).
    ("unrelated", 0.045, 0.006, False), ("unrelated", 0.038, 0.005, False),
    ("unrelated", 0.052, 0.007, False), ("unrelated", 0.041, 0.004, False),
    ("unrelated", 0.036, 0.005, False), ("unrelated", 0.048, 0.006, False),
    ("unrelated", 0.033, 0.003, False), ("unrelated", 0.050, 0.005, False),
    ("unrelated", 0.042, 0.004, False), ("unrelated", 0.039, 0.006, False),
    ("unrelated", 0.031, 0.003, False), ("unrelated", 0.046, 0.005, False),
    ("unrelated", 0.037, 0.004, False),
    # Related-but-no-shared-story pairs, measured on the 136-chunk scan corpus
    # (run 20260712T074956Z false positives — cross-topic statistical reports in
    # the old mid-band). These keep the corroborate ceiling above the noise band.
    ("related_untyped", 0.408, 0.007, False), ("related_untyped", 0.420, 0.026, False),
    ("related_untyped", 0.379, 0.082, False), ("related_untyped", 0.361, 0.067, False),
    ("related_untyped", 0.373, 0.016, False), ("related_untyped", 0.400, 0.102, False),
]


def default_samples() -> list[Sample]:
    return [Sample(cos, pc, Relation(rel), story) for rel, cos, pc, story in _MEASURED]


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
        1
        for s in samples
        if classify_relation(s.cos, s.p_contra, t, same_story=s.same_story)[0]
        == s.relation
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
        preds.append(
            (
                held.relation,
                classify_relation(
                    held.cos, held.p_contra, t, same_story=held.same_story
                )[0],
            )
        )
    return LooResult(preds, folds)
