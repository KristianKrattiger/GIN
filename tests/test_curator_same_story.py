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


def test_truncation_keeps_at_least_one_negative():
    # Three high-p_contra pairs and three low-p_contra pairs; cap below the
    # total. A plain descending-sort-then-slice truncation cuts the tail,
    # which is exactly where the negatives (low p_contra) live, dropping all
    # of them. Band-aware truncation must keep at least one.
    chunks = _chunks(
        "hi1:0", "hi2:0", "hi3:0", "hi4:0", "hi5:0", "hi6:0",
        "lo1:0", "lo2:0", "lo3:0", "lo4:0", "lo5:0", "lo6:0",
    )
    same_story_pairs = [
        ("text hi1:0", "text hi2:0"),
        ("text hi3:0", "text hi4:0"),
        ("text hi5:0", "text hi6:0"),
        ("text lo1:0", "text lo2:0"),
        ("text lo3:0", "text lo4:0"),
        ("text lo5:0", "text lo6:0"),
    ]
    src = SameStoryCandidateSource(
        chunks,
        same_story=_same_story_for(same_story_pairs),
        p_contra=lambda x, y: 0.9 if "hi" in x else 0.01,
        max_candidates=3,
    )
    pairs = src.pairs()
    assert len(pairs) == 3
    ids = [{a.chunk_id, b.chunk_id} for a, b in pairs]
    assert any("lo" in cid for pair_ids in ids for cid in pair_ids), (
        f"expected at least one low-band (negative) pair to survive truncation, got {ids}"
    )


def test_unusable_score_is_retained_not_dropped():
    # A pair whose p_contra returns None, or whose p_contra raises, must still
    # reach the curator (retention over ranking) instead of blowing up the
    # whole backlog with a TypeError from `-row[0]`.
    chunks = _chunks("none:0", "raise:0", "ok1:0", "ok2:0")

    def flaky_p_contra(a_text, b_text):
        if "none" in a_text or "none" in b_text:
            return None
        if "raise" in a_text or "raise" in b_text:
            raise RuntimeError("scorer blew up")
        return 0.5

    src = SameStoryCandidateSource(
        chunks,
        same_story=lambda x, y: True,
        p_contra=flaky_p_contra,
    )
    pairs = src.pairs()
    all_ids = {cid for pair in pairs for chunk in pair for cid in [chunk.chunk_id]}
    # All 4 chunks pair up as C(4,2) = 6 same-story pairs; none should be lost.
    assert len(pairs) == 6
    assert all_ids == {"none:0", "raise:0", "ok1:0", "ok2:0"}


def test_pairs_are_memoized_predicates_run_once():
    chunks = _chunks("a:0", "b:0", "c:0")
    same_story_calls = []
    p_contra_calls = []

    def same_story(a_text, b_text):
        same_story_calls.append((a_text, b_text))
        return True

    def p_contra(a_text, b_text):
        p_contra_calls.append((a_text, b_text))
        return 0.5

    src = SameStoryCandidateSource(chunks, same_story=same_story, p_contra=p_contra)
    first = src.pairs()
    second = src.pairs()
    assert first == second
    # C(3,2) = 3 combinations; predicates must not be invoked again on the
    # second pairs() call.
    assert len(same_story_calls) == 3
    assert len(p_contra_calls) == 3


def test_chunks_returns_a_copy_not_the_internal_list():
    chunks = _chunks("a:0", "b:0")
    src = SameStoryCandidateSource(chunks, same_story=lambda x, y: True)
    returned = src.chunks()
    returned.append(LabeledChunk(chunk_id="injected:0", text="text injected:0"))
    # Mutating the returned list must not affect later pairs() output.
    pairs = src.pairs()
    ids = {cid for pair in pairs for chunk in pair for cid in [chunk.chunk_id]}
    assert "injected:0" not in ids
    assert len(src.chunks()) == 2
