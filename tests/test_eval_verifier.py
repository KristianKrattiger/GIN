"""Tests for the claim grounding verifier (no model required)."""
import pytest

from gin.eval.claims import Verdict
from gin.eval.verifier import Verifier, token_overlap


CHUNKS = [
    ("c0", "Emergency services confirmed 142 people received treatment at hospitals"),
    ("c1", "A coastal storm is expected to bring sustained winds of 45 miles per hour"),
]
VERBATIM_CHUNK = (
    "c0",
    "Emergency services confirmed 142 people received treatment at area hospitals.",
)
VERBATIM_CLAIM = "Emergency services confirmed 142 people received treatment at area hospitals."


def test_token_overlap_full_containment():
    assert token_overlap("142 people received treatment", CHUNKS[0][1]) == 1.0


def test_token_overlap_partial():
    score = token_overlap("142 people attended the concert", CHUNKS[0][1])
    assert 0.0 < score < 1.0


def test_overlap_verifier_supports_best_chunk():
    verifier = Verifier(mode="overlap", threshold=0.5)
    verdict = verifier.verify("142 people received treatment", CHUNKS)
    assert verdict.verdict == Verdict.SUPPORTED.value
    assert verdict.matched_chunk_id == "c0"
    assert verdict.score >= 0.5


def test_overlap_verifier_unsupported_when_absent():
    verifier = Verifier(mode="overlap", threshold=0.5)
    verdict = verifier.verify("the mars rover sample return launch window", CHUNKS)
    assert verdict.verdict == Verdict.UNSUPPORTED.value
    assert verdict.matched_chunk_id is None


def test_injected_scorer_overrides_backend():
    # nli mode but with an injected scorer so no model is loaded.
    scorer = lambda claim, chunk: 0.9 if "142" in chunk else 0.1
    verifier = Verifier(mode="nli", threshold=0.5, scorer=scorer)
    verdict = verifier.verify("anything", CHUNKS)
    assert verdict.verdict == Verdict.SUPPORTED.value
    assert verdict.matched_chunk_id == "c0"
    assert verdict.score == 0.9


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        Verifier(mode="magic")


def test_nli_entailment_prob_handles_softmax_vector():
    verifier = Verifier(mode="nli", threshold=0.5)
    score = verifier._nli_entailment_prob([0.1, 0.8, 0.1])
    assert score == pytest.approx(0.8, abs=0.01)


def test_nli_entailment_prob_handles_scalar():
    verifier = Verifier(mode="nli", threshold=0.5)
    assert verifier._nli_entailment_prob([0.75]) == pytest.approx(0.75)


def test_nli_verbatim_entailment_with_injected_scorer():
    """Verbatim claim/chunk pairs must score at or above threshold."""
    chunk_text = VERBATIM_CHUNK[1]

    def scorer(claim: str, chunk: str) -> float:
        return 1.0 if claim.strip() == VERBATIM_CLAIM and chunk == chunk_text else 0.0

    verifier = Verifier(mode="nli", threshold=0.5, scorer=scorer)
    verdict = verifier.verify(VERBATIM_CLAIM, [VERBATIM_CHUNK])
    assert verdict.verdict == Verdict.SUPPORTED.value
    assert verdict.score >= 0.5


def test_max_query_overlap():
    from gin.eval.verifier import max_query_overlap

    score = max_query_overlap(
        "sustained wind speed coastal storm",
        CHUNKS,
    )
    assert score > 0.5
