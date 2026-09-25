"""Tests for gin.assay_proposer.corpus — one SEAR doc per line, roles, over-long sentences."""
from dataclasses import dataclass

from gin.assay_proposer.corpus import MAX_QUOTE_WORDS, build_line_corpus

_VOCAB: dict[str, int] = {}


def _tok(b: bytes) -> list[int]:
    return [_VOCAB.setdefault(w, len(_VOCAB) + 2) for w in b.decode("utf-8").split()]


@dataclass
class Ex:
    docId: str
    role: str
    text: str
    label: str = "L"


def test_one_sear_doc_per_nonblank_line_mapped_back_to_its_excerpt():
    lc = build_line_corpus(
        [
            Ex("vendor", "claimant", "Acme is fast.\n\nAcme is cheap."),
            Ex("hn", "independent", "Acme was slow."),
        ],
        _tok,
    )
    assert len(lc.corpus.docs) == 3
    assert [(s.doc_id, s.role) for s in lc.sources] == [
        ("vendor", "claimant"),
        ("vendor", "claimant"),
        ("hn", "independent"),
    ]
    assert lc.claimant == frozenset({0, 1})
    assert lc.independent == frozenset({2})
    assert lc.claimant.isdisjoint(lc.independent)


def test_a_sentence_over_the_word_limit_cannot_start_a_quote():
    long = " ".join(["word"] * (MAX_QUOTE_WORDS + 1)) + "."
    lc = build_line_corpus([Ex("vendor", "claimant", f"Short one here. {long}")], _tok)
    # "Short one here. " is three tokens, so the long sentence starts at position 3 —
    # the same position sear.corpus records as that sentence's start.
    assert lc.forbidden_starts == {(0, 3)}
    assert (0, 3) in lc.corpus.sentence_starts


def test_a_sentence_at_the_limit_may_be_quoted():
    at_limit = " ".join(["word"] * MAX_QUOTE_WORDS) + "."
    lc = build_line_corpus([Ex("vendor", "claimant", at_limit)], _tok)
    assert lc.forbidden_starts == set()
