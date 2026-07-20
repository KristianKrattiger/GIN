"""F1 regression: a pre_ranked source's own ordering must survive to the front
of what next-pairs returns — not get re-sorted through order_backlog's
informativeness heuristic (which collapses to mid-band-first once NLI's
p_contra is None, exactly inverting the residue's evidence-based ranking).

fastapi isn't installed in this environment, so app.py can't be imported here
(test_curator_app.py fails collection for the same reason). This tests the
pure ordering decision app.next_pairs delegates to for pre_ranked sources —
gin.curator.candidates.pre_ranked_unlabeled_pairs — directly.
"""
from gin.cartographer.combined import CombinedRelationProposer
from gin.cartographer.models import LabeledChunk
from gin.curator.candidates import (
    OfflineCandidateSource,
    order_backlog,
    pre_ranked_unlabeled_pairs,
)
from gin.curator.models import pair_key
from gin.curator.residue import EscalationResidueCandidateSource

A = LabeledChunk("n1_doc_005:0", "institutional framing of the issue")
B = LabeledChunk("n2_doc_001:0", "grassroots framing of the same issue")
C = LabeledChunk("n1_doc_008:0", "an unrelated topic entirely")


def _proposer(same_story, cos, contra=None):
    contra = contra or {}
    return CombinedRelationProposer(
        embed_cos=lambda a, b: cos.get(frozenset({a, b}), 0.0),
        same_story=lambda a, b: same_story.get(frozenset({a, b}), False),
        nli_scores=lambda p, h: (contra.get(frozenset({p, h}), 0.0), 0.0, 0.0),
    )


def test_offline_source_is_not_pre_ranked():
    assert OfflineCandidateSource([A, B, C]).pre_ranked is False


def test_residue_source_declares_pre_ranked():
    src = EscalationResidueCandidateSource([A, B, C], proposer=_proposer({}, {}))
    assert src.pre_ranked is True


def test_floated_contradiction_survives_to_front_of_pre_ranked_walk():
    # Same fixture as test_curator_residue's
    # test_pairs_surfaces_high_cosine_contradiction: a high-cosine NLI
    # contradiction (A,B) must rank ahead of two mid-band pairs. Every
    # residue pair is not-same-story by construction, so type_relation's
    # p_contra is None for all three (combined.py's story-gate) — the exact
    # condition that made informativeness() collapse them all to the
    # mid-band tier and invert the ranking (F1).
    cos = {
        frozenset({A.text, B.text}): 0.70,  # high cosine, real contradiction
        frozenset({A.text, C.text}): 0.30,  # mid-band noise
        frozenset({B.text, C.text}): 0.32,  # mid-band noise
    }
    contra = {frozenset({A.text, B.text}): 0.9}
    src = EscalationResidueCandidateSource(
        [A, B, C], proposer=_proposer({}, cos, contra), cos_floor=0.20,
    )
    ranked = src.pairs()
    assert ranked, "residue produced no pairs"

    # The pre_ranked walk (what app.next_pairs now does) preserves the
    # source's own ranking: the contradiction pair is first.
    walked = pre_ranked_unlabeled_pairs(src, already_labeled=set())
    assert pair_key(walked[0][0].chunk_id, walked[0][1].chunk_id) == pair_key(
        A.chunk_id, B.chunk_id
    )
    assert [pair_key(a.chunk_id, b.chunk_id) for a, b in walked] == [
        pair_key(a.chunk_id, b.chunk_id) for a, b in ranked
    ]

    # Demonstrate the bug this fixes: re-sorting the same pairs through
    # order_backlog with the signals type_relation actually reports (p_contra
    # None for every not-same-story pair) buries the contradiction — proving
    # the pre_ranked branch is load-bearing, not redundant with the old path.
    scored = [
        (pair, {"cosine": cos[frozenset({pair[0].text, pair[1].text})], "nli_p_contra": None})
        for pair in ranked
    ]
    inverted = order_backlog(scored, already_labeled=set())
    inverted_key = pair_key(inverted[0][0][0].chunk_id, inverted[0][0][1].chunk_id)
    assert inverted_key != pair_key(A.chunk_id, B.chunk_id)


def test_pre_ranked_walk_skips_already_labeled_pairs():
    cos = {
        frozenset({A.text, B.text}): 0.70,
        frozenset({A.text, C.text}): 0.30,
        frozenset({B.text, C.text}): 0.32,
    }
    contra = {frozenset({A.text, B.text}): 0.9}
    src = EscalationResidueCandidateSource(
        [A, B, C], proposer=_proposer({}, cos, contra), cos_floor=0.20,
    )
    already_labeled = {pair_key(A.chunk_id, B.chunk_id)}
    walked = pre_ranked_unlabeled_pairs(src, already_labeled)
    assert pair_key(A.chunk_id, B.chunk_id) not in [
        pair_key(a.chunk_id, b.chunk_id) for a, b in walked
    ]
