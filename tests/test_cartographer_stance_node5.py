"""Pins the stance channel's measured outcome on the 24 node5 curator labels.

Regenerate the numbers with:
    venv/Scripts/python.exe scripts/eval_node5_stance.py

Two figures matter and they answer different questions, so both are pinned here:

* **Stance arm in isolation** (this module's main test). Cosine and NLI are
  injected so the NLI channel never fires, which isolates the branch this work
  changed. The stance arm contributes **zero** within-event false positives.
* **End to end with real models** (`scripts/eval_node5_stance.py`). `P` 1.000,
  `P_all` 0.857 -- as of sub-project F (2026-07-27), which let the stance
  channel veto a firing NLI on same-story disagreement. Before that (E,
  shipped), `P` was 0.857 / `P_all` 0.750, because NLI kept unconditional
  priority over the stance arm and produced 3 of the 4 residual false
  positives. Now 2 residual false positives remain: one stance-channel, one
  via the `stance=None` pre-veto fallback (not "NLI priority"). Not pinned
  here: it needs models, so it lives in the script.

Conflating the two would credit the stance channel with NLI's errors or blame it
for them. The isolation test is the one that regresses if the stance rule breaks.
"""
from __future__ import annotations

from gin.cartographer.combined import CombinedRelationProposer
from gin.cartographer.models import Relation
from gin.cartographer.relatedness import make_same_story
from gin.curator.node5_labels import (
    BASELINE_P,
    BASELINE_P_ALL,
    MetricScore,
    node5_pairs,
    node5_texts,
    score,
)
from gin.curator.text_index import default_text_index


def _isolated_proposer(texts: dict[str, str]) -> CombinedRelationProposer:
    """Real same-story predicate, injected cosine and NLI.

    cos 0.95 clears the gate; p_contra 0.05 keeps the NLI channel silent. What is
    under test is the stance arm, not the embedding or the cross-encoder.

    The predicate is built over ``default_text_index()`` ALONE. It already
    contains node5 since CORPUS_NODES registered it, so adding node5's texts on
    top would double its document frequencies and mask cross-event false
    positives -- see docs/superpowers/specs/2026-07-26-stage1-anchor-findings.md.
    """
    return CombinedRelationProposer(
        embed_cos=lambda a, b: 0.95,
        nli_scores=lambda a, b: (0.05, 0.05, 0.90),
        same_story=make_same_story(list(default_text_index().values())),
    )


def _typed_rows(texts, proposer):
    return [
        (pair, proposer.type_relation(texts[pair.src], texts[pair.dst])[0]
         is Relation.CONTRADICTS)
        for pair in node5_pairs()
    ]


def test_stance_arm_beats_the_pre_registered_floor():
    texts = node5_texts()
    rows = _typed_rows(texts, _isolated_proposer(texts))
    within = score([(p, t) for p, t in rows if p.within_event])
    overall = score(rows)

    assert within.precision > BASELINE_P, f"within-event precision regressed: {within.precision:.3f}"
    assert overall.precision > BASELINE_P_ALL, f"overall precision regressed: {overall.precision:.3f}"
    assert within.recall >= 0.75, f"recall floor breached: {within.recall:.3f}"
    # Measured 2026-07-26. Pinned as exact counts, not ratios, so a change that
    # preserves the ratio while moving WHICH pairs are right still fails.
    assert within == MetricScore(tp=12, fp=0, fn=0)


def test_the_two_residual_false_positives_are_the_pre_registered_ones():
    """Both are cross-event, and each illustrates a different known cost.

    ``n5_doc_023 <-> 024`` fires through the **stance** channel and needs BOTH
    known weaknesses at once: stage 1's union anchor passes it ("Union Yard" in
    a transit report against "the union local" in a port-strike report), and at
    ALIGN_FLOOR 0.05 two unrelated counts clear the measure-overlap test. It is
    the concrete instance of the low-floor hazard the plan pre-registered, and
    either fix removes it.

    ``n5_doc_023 <-> 026`` fires through the **band** channel with
    ``stance=None`` -- the pre-stance branch, reached because one side states no
    quantity. That is the deliberate price of the None-versus-UNALIGNED split:
    keeping the None path is what preserves the three gold contradicts pairs
    that contradict qualitatively, and the same path necessarily preserves the
    degenerate branch for quantity-free pairs. Recorded as a measured cost of
    that decision, not a defect.

    Pinned by name and channel so a change that swaps one false positive for a
    different one cannot hide behind an unchanged total.
    """
    texts = node5_texts()
    proposer = _isolated_proposer(texts)
    false_positives = {}
    for pair in node5_pairs():
        typed, ev = proposer.type_relation(texts[pair.src], texts[pair.dst])
        if typed is Relation.CONTRADICTS and not pair.gold_contradicts:
            false_positives[frozenset((pair.src, pair.dst))] = ev["channel"]

    assert false_positives == {
        frozenset(("n5_doc_023:0", "n5_doc_024:0")): "stance",
        frozenset(("n5_doc_023:0", "n5_doc_026:0")): "band",
    }
    # Both are cross-event, so neither costs within-event precision.
    within_event_fps = [
        p for p in node5_pairs()
        if p.within_event and frozenset((p.src, p.dst)) in false_positives
    ]
    assert within_event_fps == []


def test_every_gold_contradicts_pair_is_still_found():
    # R must stay at 1.000: the channel removes wrong edges without losing any
    # of the 12 real conflicts. Stated separately from the precision test so a
    # regression names which half moved.
    texts = node5_texts()
    rows = _typed_rows(texts, _isolated_proposer(texts))
    missed = [p.src + " <-> " + p.dst for p, typed in rows
              if p.gold_contradicts and not typed]
    assert missed == [], f"lost real conflicts: {missed}"
