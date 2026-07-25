"""Export curator labels as unmeasured calibration rows.

Lives in gin.curator because it reads the label store; gin.cartographer may not
import gin.curator, so the cartographer-side code only handles schema and I/O.

Signal computation is injected rather than imported, so every test here runs
without embed or NLI models. The real scorer is wired in
scripts/regen_calibration_samples.py.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional

from gin.cartographer.eval_pairs import eval_pair_keys
from gin.cartographer.models import Relation

from .store import Store
from .text_index import default_text_index

# (a_text, b_text) -> (cos, p_contra, same_story)
SignalsFn = Callable[[str, str], tuple[float, float, bool]]

# Relations the threshold classifier can emit. SUPERSEDES is a graph relation,
# not a detector output, so it is not a calibration target.
_CLASSIFIER_RELATIONS = frozenset(
    {Relation.CONTRADICTS, Relation.CORROBORATES, Relation.RELATED_UNTYPED, Relation.UNRELATED}
)


@dataclass(frozen=True)
class ExportReport:
    rows: list[dict]        # calibration rows
    eval_rows: list[dict]   # held-out rows, measured but never calibrated on
    drops: dict[str, int]

    @property
    def class_counts(self) -> dict[str, int]:
        return dict(Counter(r["relation"] for r in self.rows))


def export_calibration_rows(
    store: Store,
    signals_fn: SignalsFn,
    text_index: Optional[dict[str, str]] = None,
) -> ExportReport:
    """Fold the store into calibration rows, excluding every eval pair."""
    text = default_text_index() if text_index is None else text_index
    eval_keys = eval_pair_keys()
    drops: Counter[str] = Counter()
    rows: list[dict] = []
    eval_rows: list[dict] = []

    for src, dst, relation, _relation_class in sorted(
        store.gold(), key=lambda row: tuple(sorted((row[0], row[1])))
    ):
        if relation not in _CLASSIFIER_RELATIONS:
            drops["not_a_classifier_output"] += 1
            continue
        is_eval_pair = frozenset((src, dst)) in eval_keys
        if src not in text or dst not in text:
            drops["text_unresolved" if not is_eval_pair else "eval_pair"] += 1
            continue
        cos, p_contra, same_story = signals_fn(text[src], text[dst])
        measured = {
            "cos": float(cos),
            "p_contra": float(p_contra),
            "same_story": bool(same_story),
            "relation": relation.value,
        }
        if is_eval_pair:
            # Measured for the held-out score, then kept out of calibration.
            drops["eval_pair"] += 1
            eval_rows.append({"src": src, "dst": dst, **measured})
            continue
        rows.append(measured)

    if not rows:
        raise ValueError(f"no calibration rows after filtering (drops: {dict(drops)})")
    return ExportReport(rows, eval_rows, dict(drops))
