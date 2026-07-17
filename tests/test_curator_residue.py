"""EscalationResidueCandidateSource reuses escalation_candidates (model-free via
an injected proposer)."""
from gin.cartographer.combined import CombinedRelationProposer
from gin.cartographer.models import LabeledChunk
from gin.curator.models import pair_key
from gin.curator.residue import EscalationResidueCandidateSource

A = LabeledChunk("n1_doc_005:0", "institutional framing of the issue")
B = LabeledChunk("n2_doc_001:0", "grassroots framing of the same issue")
C = LabeledChunk("n1_doc_008:0", "an unrelated topic entirely")


def _proposer(same_story, cos):
    # Injected scorers => model-free. escalation_candidates needs same_story wired.
    return CombinedRelationProposer(
        embed_cos=lambda a, b: cos.get(frozenset({a, b}), 0.0),
        same_story=lambda a, b: same_story.get(frozenset({a, b}), False),
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


def test_pairs_sorted_by_cosine_desc():
    cos = {frozenset({A.text, B.text}): 0.40,
           frozenset({A.text, C.text}): 0.60,
           frozenset({B.text, C.text}): 0.50}
    src = EscalationResidueCandidateSource(
        [A, B, C], proposer=_proposer({}, cos), cos_floor=0.30,
    )
    ordered = [(a.chunk_id, b.chunk_id) for a, b in src.pairs()]
    assert ordered[0] == ("n1_doc_005:0", "n1_doc_008:0")  # 0.60 first


def test_chunks_returns_input():
    src = EscalationResidueCandidateSource([A, B, C], proposer=_proposer({}, {}))
    assert src.chunks() == [A, B, C]
