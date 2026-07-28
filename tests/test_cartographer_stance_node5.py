"""Pins the stance channel's measured outcome on the 24 node5 curator labels.

Regenerate the numbers with:
    venv/Scripts/python.exe scripts/eval_node5_stance.py

Two figures matter and they answer different questions, so both are pinned here:

* **Stance arm in isolation** (this module's main test). Cosine and NLI are
  injected so the NLI channel never fires, which isolates the branch this work
  changed. The stance arm contributes **zero** within-event false positives.
* **End to end with real models** (`scripts/eval_node5_stance.py`). `P` 1.000,
  `R` 1.000, `P_all` **1.000** (tp 12, fp 0) -- as of the variant-D anchor
  test (2026-07-28), which removed the last false positive. The history, each
  step measured: sub-project F shipped at `P_all` 0.857 with two cross-event
  false positives; sub-project G (2026-07-27) stripped the hedge-adverb
  "roughly" from measure tokens, removing `n5_doc_023 <-> 024` (0.923); the
  variant-D mixed anchor test (entity-grade in one text, at least capitalized
  in the other, docs/superpowers/specs/2026-07-26-stage1-anchor-findings.md)
  removed `n5_doc_023 <-> 026`, whose 'Union Yard' / 'the union local'
  collision was the stage-1 defect that let a quantity-free pair reach the
  typing channels at all. Not pinned here: it needs models, so it lives in
  the script.

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


def test_no_residual_false_positives_remain():
    """Every historical node5 false positive now has a pinned fix.

    ``n5_doc_023 <-> 024`` was fixed in sub-project G (`docs/superpowers/specs/
    2026-07-27-quantity-hedge-word-stopword-design.md`): the shared hedge-word
    "roughly" was the entire measure overlap between an unrelated dockworker
    headcount and a transit delay in minutes. Stripping it from
    ``_STOPWORDS`` moves this pair's stance to ``UNALIGNED``, which correctly
    abstains.

    ``n5_doc_023 <-> 026`` -- the last one, previously typed through the band
    channel with ``stance=None`` because 026 states no quantity -- was a
    stage-1 defect, not a stance-layer one: 'Union Yard' (proper noun) in one
    text anchored against 'the union local' (common noun) in the other, so the
    pair counted as one story. The variant-D anchor test shipped 2026-07-28
    (entity-grade in one text, at least capitalized in the other -- see
    docs/superpowers/specs/2026-07-26-stage1-anchor-findings.md) rejects the
    pair at stage 1, so it never reaches any typing channel.

    Pinned as an exact empty dict, with the channel recorded per pair, so a
    new false positive cannot hide behind an aggregate ratio.
    """
    texts = node5_texts()
    proposer = _isolated_proposer(texts)
    false_positives = {}
    for pair in node5_pairs():
        typed, ev = proposer.type_relation(texts[pair.src], texts[pair.dst])
        if typed is Relation.CONTRADICTS and not pair.gold_contradicts:
            false_positives[frozenset((pair.src, pair.dst))] = ev["channel"]

    assert false_positives == {}


def test_every_gold_contradicts_pair_is_still_found():
    # R must stay at 1.000: the channel removes wrong edges without losing any
    # of the 12 real conflicts. Stated separately from the precision test so a
    # regression names which half moved.
    texts = node5_texts()
    rows = _typed_rows(texts, _isolated_proposer(texts))
    missed = [p.src + " <-> " + p.dst for p, typed in rows
              if p.gold_contradicts and not typed]
    assert missed == [], f"lost real conflicts: {missed}"
