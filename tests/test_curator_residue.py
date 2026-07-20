"""EscalationResidueCandidateSource reuses escalation_candidates (model-free via
an injected proposer)."""
from gin.cartographer.combined import CombinedRelationProposer
from gin.cartographer.models import LabeledChunk
from gin.curator.models import pair_key
from gin.curator.residue import EscalationResidueCandidateSource

A = LabeledChunk("n1_doc_005:0", "institutional framing of the issue")
B = LabeledChunk("n2_doc_001:0", "grassroots framing of the same issue")
C = LabeledChunk("n1_doc_008:0", "an unrelated topic entirely")


def _proposer(same_story, cos, contra=None):
    # Injected scorers => model-free. escalation_candidates needs same_story wired;
    # nli_scores keeps the NLI-aware ranking model-free too (default: no contradiction).
    contra = contra or {}
    return CombinedRelationProposer(
        embed_cos=lambda a, b: cos.get(frozenset({a, b}), 0.0),
        same_story=lambda a, b: same_story.get(frozenset({a, b}), False),
        nli_scores=lambda p, h: (contra.get(frozenset({p, h}), 0.0), 0.0, 0.0),
    )


def test_pairs_excludes_same_story_and_below_floor():
    cos = {frozenset({A.text, B.text}): 0.40,   # residue: not same-story, above floor
           frozenset({A.text, C.text}): 0.05,   # below floor -> dropped
           frozenset({B.text, C.text}): 0.50}   # same-story -> dropped
    same_story = {frozenset({B.text, C.text}): True}
    src = EscalationResidueCandidateSource(
        [A, B, C], proposer=_proposer(same_story, cos), cos_floor=0.30,
    )
    keys = {pair_key(a.chunk_id, b.chunk_id) for a, b in src.pairs()}
    assert keys == {pair_key("n1_doc_005:0", "n2_doc_001:0")}


def test_pairs_prioritizes_high_cosine_within_residue():
    # Corrected model: real issue_frame is HIGH cosine (shared vocabulary,
    # opposed stance), so among non-contradiction residue pairs the rank is
    # cosine-descending and a tight cap keeps the highest-cosine pair — not the
    # mid-band cross-topic noise the module originally favored.
    cos = {frozenset({A.text, B.text}): 0.30,   # mid-band cross-topic noise
           frozenset({A.text, C.text}): 0.72,   # high cosine (issue_frame zone)
           frozenset({B.text, C.text}): 0.68}
    src = EscalationResidueCandidateSource(
        [A, B, C], proposer=_proposer({}, cos), cos_floor=0.20, max_candidates=1,
    )
    kept = [pair_key(a.chunk_id, b.chunk_id) for a, b in src.pairs()]
    assert kept == [pair_key("n1_doc_005:0", "n1_doc_008:0")]  # the 0.72 high-cosine pair


def test_pairs_surfaces_high_cosine_contradiction():
    # A high-cosine pair that NLI flags as contradiction (the issue_frame
    # signature: shared vocabulary, opposed stance) must outrank a mid-band
    # pair under a tight cap — otherwise real issue_frame pairs get truncated.
    cos = {frozenset({A.text, B.text}): 0.70,   # high cosine (shared policy vocab)
           frozenset({A.text, C.text}): 0.30,   # mid-band
           frozenset({B.text, C.text}): 0.32}
    contra = {frozenset({A.text, B.text}): 0.9}  # NLI: A vs B contradict
    src = EscalationResidueCandidateSource(
        [A, B, C], proposer=_proposer({}, cos, contra),
        cos_floor=0.20, max_candidates=1,
    )
    kept = [pair_key(a.chunk_id, b.chunk_id) for a, b in src.pairs()]
    assert kept == [pair_key(A.chunk_id, B.chunk_id)]  # the high-cos contradiction


def test_cross_topic_nli_artifact_does_not_float_above_real_issue_frame():
    # Regression for the cross-topic numeric-claim artifact: NLI scores some
    # topically-unrelated pairs as near-certain contradictions. Measured in the
    # real corpus_node1-4 residue: a monetary-policy x climate pair at cos 0.357
    # scored p_contra 0.932 — higher than any genuine issue_frame pair. Without
    # a cosine floor on the contradiction float it would head the curator queue.
    # A-B is that artifact; A-C is a real issue_frame pair (high cosine).
    cos = {frozenset({A.text, B.text}): 0.357,   # cross-topic, below the ceiling
           frozenset({A.text, C.text}): 0.60,    # topically close: contradiction plausible
           frozenset({B.text, C.text}): 0.31}
    contra = {frozenset({A.text, B.text}): 0.932,  # artifact: very high p_contra
              frozenset({A.text, C.text}): 0.80}   # genuine, lower p_contra
    src = EscalationResidueCandidateSource(
        [A, B, C], proposer=_proposer({}, cos, contra), cos_floor=0.20,
    )
    kept = [pair_key(a.chunk_id, b.chunk_id) for a, b in src.pairs()]
    # The genuine high-cosine contradiction leads despite its LOWER p_contra;
    # the artifact is demoted to the cosine-ranked remainder.
    assert kept[0] == pair_key(A.chunk_id, C.chunk_id)
    assert kept.index(pair_key(A.chunk_id, B.chunk_id)) > 0


def test_chunks_returns_input():
    src = EscalationResidueCandidateSource([A, B, C], proposer=_proposer({}, {}))
    assert src.chunks() == [A, B, C]


def test_pairs_bounds_nli_to_rank_limit():
    # The ~7min-hang fix: NLI (a per-pair model call) must be consulted only
    # for the nli_rank_limit highest-cosine residue pairs. Pairs beyond the
    # limit are ranked by cosine alone and must NEVER trigger an NLI call —
    # enforced here by raising if the injected scorer sees any other pair.
    cos = {frozenset({A.text, B.text}): 0.72,   # top cosine -> only this is consulted
           frozenset({A.text, C.text}): 0.50,
           frozenset({B.text, C.text}): 0.40}
    top_pair = frozenset({A.text, B.text})
    consulted: list[frozenset] = []

    def nli_scores(p, h):
        key = frozenset({p, h})
        consulted.append(key)
        if key != top_pair:
            raise AssertionError(f"NLI consulted beyond rank limit: {p!r}, {h!r}")
        return (0.0, 0.0, 1.0)  # below threshold -> no contradiction float

    proposer = CombinedRelationProposer(
        embed_cos=lambda a, b: cos.get(frozenset({a, b}), 0.0),
        same_story=lambda a, b: False,
        nli_scores=nli_scores,
    )
    src = EscalationResidueCandidateSource(
        [A, B, C], proposer=proposer, cos_floor=0.20, nli_rank_limit=1,
    )
    kept = [pair_key(a.chunk_id, b.chunk_id) for a, b in src.pairs()]
    # No contradiction floats (p_contra stayed 0.0), so the order is pure
    # cosine-descending: A-B (0.72) > A-C (0.50) > B-C (0.40).
    assert kept == [
        pair_key(A.chunk_id, B.chunk_id),
        pair_key(A.chunk_id, C.chunk_id),
        pair_key(B.chunk_id, C.chunk_id),
    ]
    assert consulted, "expected the top-cosine pair to be consulted at least once"
    assert set(consulted) == {top_pair}
