"""Additional tests for sear.corpus.from_chunks."""
from sear.corpus import Corpus


def _tok(data: bytes) -> list[int]:
    return [hash(w) % 1000 for w in data.decode().split()]


def test_from_chunks_roundtrip():
    chunks = [("a:0", "the fox ran"), ("b:0", "the dog slept")]
    corpus = Corpus.from_chunks(chunks, tokenize=_tok)
    assert corpus.doc_names == ["a:0", "b:0"]
    assert len(corpus.docs) == 2
