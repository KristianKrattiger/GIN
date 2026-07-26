"""SameStoryCandidateSource: selects same-story pairs, ranks conflicts first."""
from gin.cartographer.models import LabeledChunk
from gin.curator.same_story import SameStoryCandidateSource


def _chunks(*ids):
    return [LabeledChunk(chunk_id=i, text=f"text {i}") for i in ids]


def _same_story_for(pairs):
    """True only for the given unordered text pairs."""
    keys = {frozenset(p) for p in pairs}

    def predicate(a_text, b_text):
        return frozenset((a_text, b_text)) in keys

    return predicate


def test_only_same_story_pairs_are_offered():
    chunks = _chunks("a:0", "b:0", "c:0")
    src = SameStoryCandidateSource(
        chunks, same_story=_same_story_for([("text a:0", "text b:0")])
    )
    pairs = src.pairs()
    assert len(pairs) == 1
    assert {pairs[0][0].chunk_id, pairs[0][1].chunk_id} == {"a:0", "b:0"}


def test_negatives_are_included_not_filtered():
    # The whole point of the corpus is same-story pairs that are NOT conflicts.
    # A source that kept only high-p_contra pairs would drop them.
    chunks = _chunks("a:0", "b:0", "c:0", "d:0")
    src = SameStoryCandidateSource(
        chunks,
        same_story=_same_story_for([("text a:0", "text b:0"), ("text c:0", "text d:0")]),
        p_contra=lambda x, y: 0.9 if "a:0" in x else 0.01,
    )
    assert len(src.pairs()) == 2


def test_conflicts_rank_before_negatives():
    chunks = _chunks("lo1:0", "lo2:0", "hi1:0", "hi2:0")
    src = SameStoryCandidateSource(
        chunks,
        same_story=_same_story_for(
            [("text lo1:0", "text lo2:0"), ("text hi1:0", "text hi2:0")]
        ),
        p_contra=lambda x, y: 0.95 if "hi" in x else 0.02,
    )
    first = src.pairs()[0]
    assert {first[0].chunk_id, first[1].chunk_id} == {"hi1:0", "hi2:0"}


def test_is_pre_ranked_so_the_app_does_not_resort():
    assert SameStoryCandidateSource.pre_ranked is True


def test_chunks_round_trip():
    chunks = _chunks("a:0", "b:0")
    src = SameStoryCandidateSource(chunks, same_story=lambda x, y: True)
    assert [c.chunk_id for c in src.chunks()] == ["a:0", "b:0"]


def test_max_candidates_caps_the_backlog():
    chunks = _chunks(*[f"c{i}:0" for i in range(10)])
    src = SameStoryCandidateSource(
        chunks, same_story=lambda x, y: True, p_contra=lambda x, y: 0.5,
        max_candidates=4,
    )
    assert len(src.pairs()) == 4


def test_empty_when_nothing_is_same_story():
    src = SameStoryCandidateSource(_chunks("a:0", "b:0"), same_story=lambda x, y: False)
    assert src.pairs() == []
