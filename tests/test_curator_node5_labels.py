"""The node5 label fold and the pre-registered metric arithmetic.

The metric is what the whole sub-project is judged on, so it is under test
rather than living in a print statement.
"""
from __future__ import annotations

import pytest

from gin.cartographer.models import Relation
from gin.curator.node5_labels import (
    BASELINE_P,
    BASELINE_P_ALL,
    BASELINE_R,
    HELD_OUT_EVENTS,
    MetricScore,
    Node5Pair,
    node5_pairs,
    node5_texts,
    score,
)


def _pair(relation: Relation, *, event: str = "e1", within: bool = True) -> Node5Pair:
    return Node5Pair(
        src="n5_doc_001:0", dst="n5_doc_002:0", relation=relation,
        event=event, within_event=within, held_out=event in HELD_OUT_EVENTS,
    )


# --- the metric --------------------------------------------------------------

def test_score_counts_tp_fp_fn():
    rows = [
        (_pair(Relation.CONTRADICTS), True),    # tp
        (_pair(Relation.CORROBORATES), True),   # fp
        (_pair(Relation.CONTRADICTS), False),   # fn
        (_pair(Relation.SUPERSEDES), False),    # true negative, counted nowhere
    ]
    assert score(rows) == MetricScore(tp=1, fp=1, fn=1)


def test_precision_and_recall():
    s = MetricScore(tp=12, fp=7, fn=0)
    assert s.precision == pytest.approx(12 / 19)
    assert s.recall == 1.0


def test_precision_and_recall_are_nan_when_undefined():
    # A rule that types nothing CONTRADICTS has undefined precision. Returning
    # NaN rather than 0.0 keeps "emitted nothing" distinguishable from "emitted
    # only wrong answers" -- they are different failures.
    import math
    assert math.isnan(MetricScore(tp=0, fp=0, fn=5).precision)
    assert math.isnan(MetricScore(tp=0, fp=3, fn=0).recall)


def test_baselines_are_the_measured_degenerate_branch():
    # combined.py's unconditional `if same_story: return CONTRADICTS`, measured
    # on these 24 labels at ebceb46.
    assert BASELINE_P == pytest.approx(12 / 19)
    # The degenerate branch typed EVERY same-story pair CONTRADICTS, so it never
    # missed a true one -- recall was perfect and precision was the whole defect.
    assert BASELINE_R == 1.0
    assert BASELINE_P_ALL == pytest.approx(12 / 24)


# --- the fold, against the real store ---------------------------------------

def test_node5_pairs_reads_the_24_curator_labels():
    pairs = node5_pairs()
    assert len(pairs) == 24
    counts = {}
    for p in pairs:
        counts[p.relation] = counts.get(p.relation, 0) + 1
    assert counts == {
        Relation.CONTRADICTS: 12,
        Relation.SUPERSEDES: 5,
        Relation.UNRELATED: 5,
        Relation.CORROBORATES: 2,
    }


def test_within_and_cross_event_split_is_19_and_5():
    pairs = node5_pairs()
    within = [p for p in pairs if p.within_event]
    cross = [p for p in pairs if not p.within_event]
    assert len(within) == 19
    assert len(cross) == 5
    # Every cross-event pair is one the curator called unrelated -- they are
    # stage-1 false positives, not a fifth relation class.
    assert {p.relation for p in cross} == {Relation.UNRELATED}


def test_held_out_split_is_three_events_and_six_pairs():
    within = [p for p in node5_pairs() if p.within_event]
    held = [p for p in within if p.held_out]
    dev = [p for p in within if not p.held_out]
    assert len(held) == 6
    assert len(dev) == 13
    assert {p.event for p in held} == set(HELD_OUT_EVENTS)
    assert len({p.event for p in dev}) == 7


def test_node5_texts_resolves_every_labeled_endpoint():
    texts = node5_texts()
    for pair in node5_pairs():
        assert pair.src in texts
        assert pair.dst in texts


def test_cross_event_pairs_are_never_marked_held_out():
    # held_out names membership in the pre-registered held-out EVENT split, and
    # that split only applies to within-event pairs. A cross-event pair spans
    # two events, so the field must not depend on which endpoint is src.
    for pair in node5_pairs():
        if not pair.within_event:
            assert pair.held_out is False, f"{pair.src} <-> {pair.dst}"


def test_baseline_p_is_reproduced_by_the_degenerate_rule():
    # Sanity-check the fold against the number the spec pre-registered: the old
    # branch typed EVERY same-story pair CONTRADICTS, and all 24 are same-story.
    within = [p for p in node5_pairs() if p.within_event]
    s = score([(p, True) for p in within])
    assert s.precision == pytest.approx(BASELINE_P)
    all_pairs = node5_pairs()
    assert score([(p, True) for p in all_pairs]).precision == pytest.approx(BASELINE_P_ALL)
