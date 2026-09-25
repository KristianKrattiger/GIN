"""Tests for gin.assay_proposer.corpus — one SEAR doc per line, roles, over-long sentences."""
import re
from dataclasses import dataclass

from gin.assay_proposer.corpus import MAX_QUOTE_WORDS, build_line_corpus

_VOCAB: dict[str, int] = {}


def _tok(b: bytes) -> list[int]:
    return [_VOCAB.setdefault(w, len(_VOCAB) + 2) for w in b.decode("utf-8").split()]


_SUB_VOCAB: dict[str, int] = {}
_SUB_INV: dict[int, str] = {}


def _subword_tok(b: bytes) -> list[int]:
    """GPT-style pre-tokenization: a word carries its leading space, punctuation is its own
    token, and whitespace at the end of the text is a token of its own -- which is what
    makes SEAR's prefix count one token late."""
    out = []
    for p in re.findall(r" ?[A-Za-z0-9%]+| ?[^\sA-Za-z0-9]|\s+", b.decode("utf-8")):
        if p not in _SUB_VOCAB:
            _SUB_VOCAB[p] = len(_SUB_VOCAB) + 2
            _SUB_INV[_SUB_VOCAB[p]] = p
        out.append(_SUB_VOCAB[p])
    return out


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


def test_crlf_line_endings_do_not_leave_a_carriage_return_token():
    byte_tok = lambda b: list(b)
    lc = build_line_corpus([Ex("vendor", "claimant", "Acme is fast.\r\nAcme is cheap.\r\n")], byte_tok)
    assert len(lc.corpus.docs) == 2
    assert 13 not in lc.corpus.docs[0]
    assert 13 not in lc.corpus.docs[1]
    assert bytes(lc.corpus.docs[0]) == b"Acme is fast."


def test_sentence_boundaries_land_on_word_starts_under_a_subword_tokenizer():
    lc = build_line_corpus([Ex("vendor", "claimant", "Acme is up. Acme is down.")], _subword_tok)
    assert lc.corpus.sentence_starts == {(0, 0), (0, 4)}
    assert lc.corpus.sentence_ends == {(0, 3), (0, 7)}
    assert _SUB_INV[lc.corpus.docs[0][4]] == " Acme"


def test_an_over_long_second_sentence_is_forbidden_at_its_first_word_under_a_subword_tokenizer():
    long = " ".join(["word"] * (MAX_QUOTE_WORDS + 1)) + "."
    lc = build_line_corpus([Ex("vendor", "claimant", f"Short one here. {long}")], _subword_tok)
    assert lc.forbidden_starts == {(0, 4)}
