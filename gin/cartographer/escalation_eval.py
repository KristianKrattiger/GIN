"""Escalation frame-judge calibration — issue_frame gold + expanded controls.

Separate from scan_eval on purpose: scan_eval's CLASS_C metric denominator is
pinned by closed label decisions, while this module owns the escalation
calibration bar. Pass = issue_frame_recall 1.0 AND class_c_discrimination 1.0
AND unrelated_discrimination 1.0, with mixed labels on the 33-pair in-memory
labeled set (the anti-collapse breadth diagnostic). Consumed by
``scripts/cartographer_eval_escalation.py``.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional

from .frame_judge import FrameJudge
from .gold_edges import (
    CLASS_C_CONTROLS,
    ESCALATION_CLASS_C_EXTRA,
    ESCALATION_UNRELATED_CONTROLS,
    gold_pairs,
)
from .scan_eval import split_gold_by_class

REASONING_SNIPPET_CHARS = 800

Pair = tuple[str, str, str]  # (src_chunk_id, dst_chunk_id, register)


def default_calibration_sets() -> dict[str, list[Pair]]:
    """Assemble the calibration pair lists from curated gold + control tuples."""
    _machine, curated = split_gold_by_class(gold_pairs())
    return {
        "issue_frame": [
            (g.src_chunk_id, g.dst_chunk_id, g.register) for g in curated
        ],
        "corroboration": list(CLASS_C_CONTROLS) + list(ESCALATION_CLASS_C_EXTRA),
        "unrelated": list(ESCALATION_UNRELATED_CONTROLS),
    }


def labeled_set_pairs() -> list[tuple[str, str, str]]:
    """(a_text, b_text, expected_label) over the 33-pair in-memory labeled set.

    Texts are inline — no DB needed — so this doubles as a breadth diagnostic
    that any judge can run anywhere.
    """
    from .labeled_set import chunks, gold
    from .models import Relation

    text_by_id = {c.chunk_id: c.text for c in chunks()}
    label_for = {
        Relation.CONTRADICTS: "DIVERGENT",
        Relation.CORROBORATES: "AGREE",
        Relation.UNRELATED: "UNRELATED",
    }
    return [(text_by_id[s], text_by_id[d], label_for[r]) for s, d, r, _reg in gold()]


def _row_block(
    judge: FrameJudge,
    text_by_chunk: dict[str, str],
    pairs: list[Pair],
    *,
    both_directions: bool,
    control: bool,
) -> tuple[list[dict], int, int]:
    """Judge pairs; return (rows, hits, flips).

    hits = DIVERGENT count for gold blocks, pass (not-DIVERGENT) count for
    control blocks. Reverse-direction labels are diagnostic only.
    """
    rows: list[dict] = []
    hits = flips = 0
    for src, dst, reg in pairs:
        if src not in text_by_chunk or dst not in text_by_chunk:
            row = {
                "src": src,
                "dst": dst,
                "register": reg,
                "label": None,
                "label_reverse": None,
                "flip": None,
                "reasoning": None,
                "skipped": "chunk_not_in_db",
            }
            if control:
                row["pass"] = None
            rows.append(row)
            continue
        a, b = text_by_chunk[src], text_by_chunk[dst]
        label = judge(a, b)
        reasoning = getattr(judge, "last_completion_text", None)
        if reasoning:
            reasoning = reasoning.strip()[:REASONING_SNIPPET_CHARS]
        label_reverse = None
        flip = None
        if both_directions:
            label_reverse = judge(b, a)
            flip = label_reverse != label
            if flip:
                flips += 1
        row = {
            "src": src,
            "dst": dst,
            "register": reg,
            "label": label,
            "label_reverse": label_reverse,
            "flip": flip,
            "reasoning": reasoning,
            "skipped": None,
        }
        if control:
            row["pass"] = label != "DIVERGENT"
            if row["pass"]:
                hits += 1
        elif label == "DIVERGENT":
            hits += 1
        rows.append(row)
    return rows, hits, flips


def evaluate_escalation_judge(
    judge: FrameJudge,
    text_by_chunk: dict[str, str],
    *,
    issue_frame_pairs: Iterable[Pair],
    corroboration_pairs: Iterable[Pair],
    unrelated_pairs: Iterable[Pair],
    labeled_pairs: Optional[list[tuple[str, str, str]]] = None,
    both_directions: bool = True,
) -> dict:
    """Score a frame judge for escalation duty.

    Primary metrics score the forward (as-listed) direction, matching
    production's single call per pair; reverse labels and flips are reported
    as an order-stability diagnostic.
    """
    issue_frame_pairs = list(issue_frame_pairs)
    corroboration_pairs = list(corroboration_pairs)
    unrelated_pairs = list(unrelated_pairs)

    gold_rows, gold_divergent, gold_flips = _row_block(
        judge, text_by_chunk, issue_frame_pairs,
        both_directions=both_directions, control=False,
    )
    cc_rows, cc_pass, cc_flips = _row_block(
        judge, text_by_chunk, corroboration_pairs,
        both_directions=both_directions, control=True,
    )
    un_rows, un_pass, un_flips = _row_block(
        judge, text_by_chunk, unrelated_pairs,
        both_directions=both_directions, control=True,
    )

    def _scorable(rows: list[dict]) -> list[dict]:
        return [r for r in rows if r["label"] is not None]

    sg, sc, su = _scorable(gold_rows), _scorable(cc_rows), _scorable(un_rows)
    # Forward-direction labels only, so runs stay comparable.
    label_distribution = dict(Counter(r["label"] for r in sg + sc + su))

    metrics: dict = {
        "issue_frame_gold_count": len(issue_frame_pairs),
        "issue_frame_scorable_count": len(sg),
        "issue_frame_divergent_count": gold_divergent,
        "issue_frame_recall": gold_divergent / len(sg) if sg else None,
        "class_c_total": len(corroboration_pairs),
        "class_c_scorable_count": len(sc),
        "class_c_pass": cc_pass,
        "class_c_discrimination": cc_pass / len(sc) if sc else None,
        "unrelated_total": len(unrelated_pairs),
        "unrelated_scorable_count": len(su),
        "unrelated_pass": un_pass,
        "unrelated_discrimination": un_pass / len(su) if su else None,
        "direction_flip_count": gold_flips + cc_flips + un_flips,
        "label_distribution": label_distribution,
        "gold_labels": gold_rows,
        "class_c_labels": cc_rows,
        "unrelated_labels": un_rows,
    }

    if labeled_pairs is not None:
        by_expected: dict[str, dict] = {}
        correct = 0
        for a_text, b_text, expected in labeled_pairs:
            got = judge(a_text, b_text)
            slot = by_expected.setdefault(
                expected, {"total": 0, "correct": 0, "got": Counter()}
            )
            slot["total"] += 1
            slot["got"][got] += 1
            if got == expected:
                slot["correct"] += 1
                correct += 1
        for slot in by_expected.values():
            slot["got"] = dict(slot["got"])
        metrics["labeled_set"] = {
            "total": len(labeled_pairs),
            "accuracy": correct / len(labeled_pairs) if labeled_pairs else None,
            "by_expected": by_expected,
        }

    return metrics
