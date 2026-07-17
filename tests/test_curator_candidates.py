"""Offline pair enumeration + hard-cases-first ordering."""
from gin.cartographer.models import LabeledChunk
from gin.curator.candidates import (
    OfflineCandidateSource,
    informativeness,
    order_backlog,
)
from gin.curator.models import pair_key

A = LabeledChunk("a:0", "alpha")
B = LabeledChunk("b:0", "bravo")
C = LabeledChunk("c:0", "charlie")


def test_offline_source_enumerates_unordered_pairs():
    src = OfflineCandidateSource([A, B, C])
    keys = {pair_key(x.chunk_id, y.chunk_id) for x, y in src.pairs()}
    assert keys == {pair_key("a:0", "b:0"), pair_key("a:0", "c:0"), pair_key("b:0", "c:0")}


def test_informativeness_tiers():
    assert informativeness({"cosine": 0.55, "nli_p_contra": 0.9}) == 2.0   # disagreement
    assert informativeness({"cosine": 0.30, "nli_p_contra": None}) == 1.0  # mid-band
    assert informativeness({"cosine": 0.80, "nli_p_contra": 0.05}) == 0.0  # obvious corroboration
    assert informativeness({"cosine": 0.05, "nli_p_contra": None}) == 0.0  # gated


def test_order_ranks_hard_cases_first_and_excludes_labeled():
    disagreement = ((A, B), {"cosine": 0.55, "nli_p_contra": 0.9})
    midband = ((A, C), {"cosine": 0.30, "nli_p_contra": None})
    obvious = ((B, C), {"cosine": 0.80, "nli_p_contra": 0.05})
    ordered = order_backlog([obvious, midband, disagreement], already_labeled=set())
    assert [p for p, _ in ordered] == [(A, B), (A, C), (B, C)]

    # Exclude an already-labeled pair.
    ordered2 = order_backlog(
        [obvious, midband, disagreement],
        already_labeled={pair_key("a:0", "b:0")},
    )
    assert (A, B) not in [p for p, _ in ordered2]
