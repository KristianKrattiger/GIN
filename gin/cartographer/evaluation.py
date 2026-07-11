"""Independent edge-precision measurement for the Cartographer.

Scores a proposer's typed assessments against a labeled pair set — precision /
recall / F1 for the ``contradicts`` relation, plus the headline
``class_c_discrimination``: of the corroborating pairs (the step-2 failure case),
the fraction the proposer correctly does NOT type as ``contradicts``. Measured on
its own axis, never through a reasoning-layer metric (design §4). Deterministic
and grounded in the same real divergence-demo corpus text as steps 1–2.

See docs/nc_cartographer_design.plan.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from .models import Assessment, LabeledChunk, Relation


@dataclass(frozen=True)
class GoldPair:
    src_chunk_id: str
    dst_chunk_id: str
    relation: Relation
    register: str


def _key(a: str, b: str) -> frozenset:
    return frozenset({a, b})


@dataclass
class CartographerMetrics:
    contradicts_precision: Optional[float]
    contradicts_recall: Optional[float]
    contradicts_f1: Optional[float]
    class_c_discrimination: Optional[float]
    by_register: dict[str, dict[str, Optional[float]]] = field(default_factory=dict)
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def to_dict(self) -> dict:
        return {
            "contradicts_precision": self.contradicts_precision,
            "contradicts_recall": self.contradicts_recall,
            "contradicts_f1": self.contradicts_f1,
            "class_c_discrimination": self.class_c_discrimination,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "by_register": self.by_register,
        }


def _prf(tp: int, fp: int, fn: int) -> tuple[Optional[float], Optional[float], Optional[float]]:
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate(
    proposals: Iterable[Assessment],
    gold: Iterable[GoldPair],
) -> CartographerMetrics:
    """Score proposals over the labeled pair set only (open pairs are ignored)."""
    gold = list(gold)
    proposed_relation: dict[frozenset, Relation] = {
        _key(p.src_chunk_id, p.dst_chunk_id): p.relation for p in proposals
    }

    tp = fp = fn = 0
    corroborating_total = corroborating_discriminated = 0
    reg_counts: dict[str, dict[str, int]] = {}

    for g in gold:
        got = proposed_relation.get(_key(g.src_chunk_id, g.dst_chunk_id), Relation.UNRELATED)
        gold_is_contra = g.relation == Relation.CONTRADICTS
        got_is_contra = got == Relation.CONTRADICTS

        reg = reg_counts.setdefault(g.register, {"tp": 0, "fp": 0, "fn": 0})
        if gold_is_contra and got_is_contra:
            tp += 1
            reg["tp"] += 1
        elif gold_is_contra and not got_is_contra:
            fn += 1
            reg["fn"] += 1
        elif not gold_is_contra and got_is_contra:
            fp += 1
            reg["fp"] += 1

        if g.relation == Relation.CORROBORATES:
            corroborating_total += 1
            if not got_is_contra:
                corroborating_discriminated += 1

    precision, recall, f1 = _prf(tp, fp, fn)
    by_register: dict[str, dict[str, Optional[float]]] = {}
    for reg, c in sorted(reg_counts.items()):
        p, r, f = _prf(c["tp"], c["fp"], c["fn"])
        by_register[reg] = {"precision": p, "recall": r, "f1": f}

    return CartographerMetrics(
        contradicts_precision=precision,
        contradicts_recall=recall,
        contradicts_f1=f1,
        class_c_discrimination=(
            corroborating_discriminated / corroborating_total
            if corroborating_total
            else None
        ),
        by_register=by_register,
        tp=tp,
        fp=fp,
        fn=fn,
    )


# --- Default labeled set -----------------------------------------------------
# Sourced from gin/cartographer/labeled_set.py: 7 divergent pairs across three
# framing registers (climate/legal/housing, author-labeled from the fixture
# edges), 3 corroborating same-stance pairs, and 3 cross-topic negatives.


def default_chunks() -> list[LabeledChunk]:
    from .labeled_set import chunks

    return chunks()


def default_gold_pairs() -> list[GoldPair]:
    from .labeled_set import gold

    return [GoldPair(s, d, r, reg) for s, d, r, reg in gold()]


def format_report(name: str, metrics: CartographerMetrics) -> str:
    def _f(v: Optional[float]) -> str:
        return "n/a" if v is None else f"{v:.3f}"

    lines = [f"# Cartographer edge-precision — {name}", ""]
    lines.append(f"- contradicts precision: {_f(metrics.contradicts_precision)}")
    lines.append(f"- contradicts recall:    {_f(metrics.contradicts_recall)}")
    lines.append(f"- contradicts F1:        {_f(metrics.contradicts_f1)}")
    lines.append(f"- class_c_discrimination: {_f(metrics.class_c_discrimination)}")
    lines.append(f"- tp/fp/fn: {metrics.tp}/{metrics.fp}/{metrics.fn}")
    lines.append("")
    lines.append("| register | precision | recall | f1 |")
    lines.append("|---|---|---|---|")
    for reg, m in metrics.by_register.items():
        lines.append(
            f"| {reg} | {_f(m['precision'])} | {_f(m['recall'])} | {_f(m['f1'])} |"
        )
    return "\n".join(lines)
