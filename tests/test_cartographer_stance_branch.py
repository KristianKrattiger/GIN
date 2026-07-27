"""The stance-gated CONTRADICTS branch.

Before this, classify_relation typed ANY same-story pair CONTRADICTS on story
membership alone -- precision 12/24 on the node5 labels. `stance=None` must
still reproduce that behavior exactly, so the committed 39-sample calibration
fixture and the 14-pair bar pin stay valid unedited.
"""
import pytest

from gin.cartographer.combined import Thresholds, classify_relation
from gin.cartographer.models import Relation

T = Thresholds(gate_floor=0.14, corroborate_ceiling=0.486, contra_threshold=0.686)


# --- stance=None reproduces the old rule exactly -----------------------------

@pytest.mark.parametrize(
    "cos,p_contra,same_story,expected_relation,expected_channel",
    [
        (0.05, 0.90, True, Relation.UNRELATED, "gate"),        # gate wins first
        (0.60, 0.90, None, Relation.CONTRADICTS, "nli"),       # NLI, no story evidence
        (0.60, 0.90, True, Relation.CONTRADICTS, "nli"),       # NLI keeps priority
        (0.60, 0.90, False, Relation.CORROBORATES, "band"),    # NLI story-blocked
        (0.60, 0.10, True, Relation.CONTRADICTS, "band"),      # THE degenerate branch
        (0.60, 0.10, False, Relation.CORROBORATES, "band"),
        (0.30, 0.10, False, Relation.RELATED_UNTYPED, "band"),
        (0.30, 0.10, None, Relation.RELATED_UNTYPED, "band"),
    ],
)
def test_stance_none_reproduces_the_current_truth_table(
    cos, p_contra, same_story, expected_relation, expected_channel
):
    relation, channel = classify_relation(cos, p_contra, T, same_story=same_story)
    assert (relation, channel) == (expected_relation, expected_channel)


# --- the new arms ------------------------------------------------------------

def test_conflict_evidence_types_contradicts_on_the_stance_channel():
    relation, channel = classify_relation(0.60, 0.10, T, same_story=True, stance="conflict")
    assert (relation, channel) == (Relation.CONTRADICTS, "stance")


@pytest.mark.parametrize("stance", ["revision", "partial"])
def test_non_conflict_evidence_abstains(stance):
    relation, channel = classify_relation(0.95, 0.10, T, same_story=True, stance=stance)
    assert (relation, channel) == (Relation.RELATED_UNTYPED, "abstain")


def test_agreement_above_the_ceiling_corroborates():
    relation, channel = classify_relation(0.95, 0.10, T, same_story=True, stance="agreement")
    assert (relation, channel) == (Relation.CORROBORATES, "band")


def test_agreement_below_the_ceiling_abstains():
    relation, channel = classify_relation(0.30, 0.10, T, same_story=True, stance="agreement")
    assert (relation, channel) == (Relation.RELATED_UNTYPED, "abstain")


def test_stance_disagreement_overrules_a_firing_nli():
    # Measured 2026-07-27: on the 24 node5 labels, every pair where NLI fires
    # and stance disagrees is wrong (a supersedes and a corroborates, both
    # scored CONTRADICTS by NLI alone); every pair where they agree is right.
    # The stance channel now wins on disagreement instead of NLI.
    relation, channel = classify_relation(0.60, 0.90, T, same_story=True, stance="partial")
    assert (relation, channel) == (Relation.RELATED_UNTYPED, "abstain")


from gin.cartographer.quantity import UNALIGNED


@pytest.mark.parametrize("stance", ["revision", "partial", "agreement", UNALIGNED])
def test_any_disagreeing_stance_overrules_a_firing_nli(stance):
    relation, channel = classify_relation(0.60, 0.90, T, same_story=True, stance=stance)
    assert (relation, channel) == (Relation.RELATED_UNTYPED, "abstain")


def test_agreeing_stance_leaves_nli_priority_untouched():
    # stance == "conflict" agrees with a firing NLI, so the veto never
    # applies -- channel stays "nli", not "stance", matching today's
    # attribution exactly.
    relation, channel = classify_relation(0.60, 0.90, T, same_story=True, stance="conflict")
    assert (relation, channel) == (Relation.CONTRADICTS, "nli")


def test_stance_is_ignored_when_stage_one_says_not_one_story():
    relation, channel = classify_relation(0.60, 0.10, T, same_story=False, stance="conflict")
    assert (relation, channel) == (Relation.CORROBORATES, "band")


from gin.cartographer.combined import CombinedRelationProposer


def test_proposer_wires_the_real_stance_provider_by_default():
    prop = CombinedRelationProposer(
        embed_cos=lambda a, b: 0.95,
        nli_scores=lambda a, b: (0.10, 0.10, 0.80),
        same_story=lambda a, b: True,
    )
    a = "Officials confirmed 34 people were evacuated from nearby buildings."
    b = "Officials confirmed 19 people were evacuated from the surrounding block."
    relation, ev = prop.type_relation(a, b)
    assert relation is Relation.CONTRADICTS
    assert ev["channel"] == "stance"
    assert ev["stance"] == "conflict"


def test_proposer_stance_provider_none_restores_the_old_branch():
    prop = CombinedRelationProposer(
        embed_cos=lambda a, b: 0.95,
        nli_scores=lambda a, b: (0.10, 0.10, 0.80),
        same_story=lambda a, b: True,
        stance_provider=None,
    )
    relation, ev = prop.type_relation("no numbers here at all", "none here either")
    assert relation is Relation.CONTRADICTS
    assert ev["channel"] == "band"
    assert "stance" not in ev


def test_unaligned_abstains_while_none_defers_to_the_pre_stance_branch():
    # The two cases must not be collapsed. UNALIGNED means the channel looked
    # and found no shared fact -> no edge. None means it had nothing to look at
    # -> leave the caller's existing behaviour alone.
    from gin.cartographer.quantity import UNALIGNED

    assert classify_relation(0.95, 0.10, T, same_story=True, stance=UNALIGNED) == (
        Relation.RELATED_UNTYPED, "abstain",
    )
    assert classify_relation(0.95, 0.10, T, same_story=True, stance=None) == (
        Relation.CONTRADICTS, "band",
    )


def test_gold_same_story_contradicts_survive_the_stance_channel():
    """Guards the REJECTED alternative: abstaining whenever nothing aligned.

    Four gold contradicts pairs pass the story gate, and three state no
    quantities at all -- housing habitability disputes and a securities PR
    versus a complaint -- so they contradict qualitatively. A blanket "abstain
    when nothing aligned" rule would discard them, which is why `None` defers to
    the pre-stance branch instead.

    Note this test does NOT exercise the None/UNALIGNED split: none of the four
    pairs ever yields UNALIGNED. That is covered by
    test_examined_but_unaligned_same_story_pairs_abstain_end_to_end.
    """
    from gin.cartographer.labeled_set import chunks as gold_chunks
    from gin.cartographer.labeled_set import gold
    from gin.cartographer.relatedness import make_same_story
    from gin.curator.text_index import default_text_index

    index = {c.chunk_id: c.text for c in gold_chunks()}
    same_story = make_same_story(list(default_text_index().values()))
    proposer = CombinedRelationProposer(
        embed_cos=lambda a, b: 0.95,
        nli_scores=lambda a, b: (0.05, 0.05, 0.90),
        same_story=same_story,
    )

    checked = 0
    for src, dst, relation, _register in gold():
        if src not in index or dst not in index:
            continue
        if relation is not Relation.CONTRADICTS:
            continue
        if not same_story(index[src], index[dst]):
            continue
        checked += 1
        typed, ev = proposer.type_relation(index[src], index[dst])
        assert typed is Relation.CONTRADICTS, f"{src} <-> {dst} ev={ev}"
    assert checked == 4, f"expected 4 gold same-story contradicts, found {checked}"


def test_examined_but_unaligned_same_story_pairs_abstain_end_to_end():
    """The regression the None/UNALIGNED split exists to prevent.

    Before the split, a same-story pair whose quantities did not align returned
    None and fell through to the pre-stance branch, emitting a CONTRADICTS edge
    on a pair the channel had examined and found nothing in. Three real corpus
    pairs are in exactly that position -- two cross-event, and n5_doc_036 vs
    n5_doc_037, a `corroborates` pair whose figures describe different measures
    (total capacity including standing room vs fixed seats in the bowl).

    Asserting stance is UNALIGNED rather than merely "not conflict" is what makes
    this non-vacuous: under the collapsed behaviour these pairs return None and
    the assertion on the channel fails.
    """
    from gin.cartographer.quantity import UNALIGNED
    from gin.cartographer.relatedness import make_same_story
    from gin.curator.node5_labels import node5_pairs, node5_texts
    from gin.curator.text_index import default_text_index

    texts = node5_texts()
    same_story = make_same_story(list(default_text_index().values()))
    proposer = CombinedRelationProposer(
        embed_cos=lambda a, b: 0.95,
        nli_scores=lambda a, b: (0.05, 0.05, 0.90),
        same_story=same_story,
    )

    unaligned = []
    for pair in node5_pairs():
        if not same_story(texts[pair.src], texts[pair.dst]):
            continue
        typed, ev = proposer.type_relation(texts[pair.src], texts[pair.dst])
        if ev.get("stance") != UNALIGNED:
            continue
        unaligned.append((pair.src, pair.dst))
        assert typed is Relation.RELATED_UNTYPED, f"{pair.src} <-> {pair.dst} ev={ev}"
        assert ev["channel"] == "abstain", f"{pair.src} <-> {pair.dst} ev={ev}"
        assert pair.relation is not Relation.CONTRADICTS, (
            f"{pair.src} <-> {pair.dst} is gold CONTRADICTS but the channel found "
            "no aligned fact -- that would be a real miss, not a saved false positive"
        )

    # Pinned so the test fails loudly if the corpus, the alignment floor, or the
    # extraction rules change which pairs land here, rather than passing on an
    # empty set.
    assert len(unaligned) == 3, f"expected 3 UNALIGNED same-story pairs, got {unaligned}"
