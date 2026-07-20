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


def test_chunks_returns_input():
    src = EscalationResidueCandidateSource([A, B, C], proposer=_proposer({}, {}))
    assert src.chunks() == [A, B, C]
