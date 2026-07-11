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


# --- Default labeled set (real divergence-demo corpus text) ------------------
# Three genuine institutional-vs-grassroots contradictions (gold from
# data/corpus_edges.yaml), one corroborating institutional pair (the step-2
# class-C case), and one cross-topic unrelated pair the gate should reject.

_INST_EMISSIONS = (
    "Global low-carbon transformations are needed to deliver cuts to predicted "
    "2030 greenhouse gas emissions of roughly 28 percent for a 2 degree C pathway "
    "and 42 percent for a 1.5 degree C pathway."
)
_GRASS_EMISSIONS = (
    "Indigenous-led resistance efforts are estimated to have stopped or delayed "
    "greenhouse gas pollution equivalent to roughly one-quarter of annual U.S. "
    "and Canadian emissions."
)
_INST_WILDFIRE = (
    "In 2023, 56,580 wildfires burned 2,693,910 acres across the United States, "
    "with acreage burned below both the five- and ten-year averages."
)
_INST_WILDFIRE_FEDERAL = (
    "About one-quarter of the nation's wildfires in 2023 occurred on federally "
    "protected lands."
)
_GRASS_WILDFIRE = (
    "Elderly, immunocompromised, and low-income populations face heightened risk "
    "from wildfire smoke exposure."
)
_INST_WATER = (
    "As of April 3, 2023, California's statewide snowpack held a snow water "
    "equivalent of 61.1 inches, or 237 percent of the April 1 average, one of the "
    "largest snowpacks on record."
)
_GRASS_WATER = (
    "Disadvantaged and cumulatively burdened communities are found to be "
    "disproportionately affected by water shortages, reflecting underlying "
    "inequities in water resource management."
)

_CHUNKS = {
    "inst_em:0": _INST_EMISSIONS,
    "grass_em:0": _GRASS_EMISSIONS,
    "inst_wf:0": _INST_WILDFIRE,
    "inst_wf_fed:0": _INST_WILDFIRE_FEDERAL,
    "grass_wf:0": _GRASS_WILDFIRE,
    "inst_wa:0": _INST_WATER,
    "grass_wa:0": _GRASS_WATER,
}


def default_chunks() -> list[LabeledChunk]:
    return [LabeledChunk(cid, text) for cid, text in _CHUNKS.items()]


def default_gold_pairs() -> list[GoldPair]:
    return [
        GoldPair("inst_em:0", "grass_em:0", Relation.CONTRADICTS, "emissions"),
        GoldPair("inst_wf:0", "grass_wf:0", Relation.CONTRADICTS, "wildfire"),
        GoldPair("inst_wa:0", "grass_wa:0", Relation.CONTRADICTS, "water"),
        # Class-C: two agreeing 2023 wildfire statistics — must NOT be contradicts.
        GoldPair("inst_wf:0", "inst_wf_fed:0", Relation.CORROBORATES, "wildfire"),
        # Cross-topic: the relatedness gate should reject this outright.
        GoldPair("inst_wf:0", "grass_wa:0", Relation.UNRELATED, "cross"),
    ]


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
