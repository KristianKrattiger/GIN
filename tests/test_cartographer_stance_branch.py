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


def test_nli_still_outranks_the_stance_branch():
    # The NLI channel owns the legal/securities register it was calibrated on;
    # stance evidence does not override a confident propositional contradiction.
    relation, channel = classify_relation(0.60, 0.90, T, same_story=True, stance="partial")
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
    """The regression this rule exists to prevent.

    Four gold contradicts pairs pass the story gate. Three state no quantities
    at all -- housing habitability disputes and a securities PR versus a
    complaint -- and contradict qualitatively. An unconditional abstain on "no
    aligned quantity" would discard them.
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
