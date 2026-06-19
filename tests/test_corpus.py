"""Tests for sear.corpus — token indexing and document store behavior."""
import pytest
from sear.corpus import Corpus

# Toy vocab and tokenizer mirroring the selftest in scripts/sear_phase1.py
_VOCAB = {w: i for i, w in enumerate(
    ["<pad>", "<eos>", "|", "the", "fox", "ran", "fast", "dog",
     "slept", "all", "day", "and"])}
_INV = {i: w for w, i in _VOCAB.items()}


def _tok(b: bytes) -> list[int]:
    return [_VOCAB[w] for w in b.decode().split()]


def _make_corpus() -> Corpus:
    return Corpus.from_texts(
        {"A": "the fox ran fast", "B": "the fox ran and the dog slept all day"},
        tokenize=_tok)


def test_document_addition():
    """Corpus stores both documents and their names after from_texts."""
    corpus = _make_corpus()
    assert len(corpus.docs) == 2
    assert corpus.doc_names == ["A", "B"]


def test_token_indexing():
    """start_index maps each token id to all (doc, pos) occurrences in the corpus."""
    corpus = _make_corpus()
    the_id = _VOCAB["the"]
    # "the" appears at pos 0 in A and at pos 0 and pos 4 in B
    entries = corpus.start_index[the_id]
    doc_ids = {d for d, _ in entries}
    assert 0 in doc_ids and 1 in doc_ids
    # "fox" appears in both docs, so it must be in the index
    fox_id = _VOCAB["fox"]
    assert fox_id in corpus.start_index


def test_continuation():
    """continuation(doc, pos) returns the next token id, or None at document end."""
    corpus = _make_corpus()
    ran_id = _VOCAB["ran"]
    # pos 1 in doc A is "fox"; its continuation is "ran" at pos 2
    assert corpus.continuation(0, 1) == ran_id
    # last token of doc A is "fast" at pos 3; no continuation
    assert corpus.continuation(0, 3) is None
