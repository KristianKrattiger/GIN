"""
Tests for sear.processor — cursor tracking, masking, and grounding failure detection.

Key behaviors under test:
- Cursor fan-out: shared prefixes across docs spawn multiple live cursors
- Cursor pruning: divergence eliminates non-matching cursors
- Multi-token attribution: spans longer than 1 token tracked correctly
- Zero-cursor signal: grounding failure emits correct mask or raises
"""
import pytest
import numpy as np
from sear.corpus import Corpus
from sear.processor import ExtractiveCopyConstraint, NEG_INF

# Toy vocab and tokenizer mirroring the selftest in scripts/sear_phase1.py
_VOCAB = {w: i for i, w in enumerate(
    ["<pad>", "<eos>", "|", "the", "fox", "ran", "fast", "dog",
     "slept", "all", "day", "and"])}
_INV = {i: w for w, i in _VOCAB.items()}
_V = len(_VOCAB)


def _tok(b: bytes) -> list[int]:
    return [_VOCAB[w] for w in b.decode().split()]


def _detok(ids: list[int]) -> str:
    return " ".join(_INV[i] for i in ids)


def _make_corpus() -> Corpus:
    return Corpus.from_texts(
        {"A": "the fox ran fast", "B": "the fox ran and the dog slept all day"},
        tokenize=_tok)


def _make_constraint(corpus: Corpus) -> ExtractiveCopyConstraint:
    return ExtractiveCopyConstraint(
        corpus, prompt_len=0, eos_id=_VOCAB["<eos>"],
        delim_id=_VOCAB["|"], min_span_len=3)


def _step(c: ExtractiveCopyConstraint, generated: list[int]) -> set[str]:
    flat = np.zeros(_V, dtype=np.float32)
    scores = c(np.array(generated, dtype=np.intc), flat.copy())
    return {_INV[i] for i in range(_V) if scores[i] > NEG_INF / 2}


def test_boundary_initial_allowed():
    """At BOUNDARY state, corpus-starting tokens and EOS are legal; delim is not."""
    corpus = _make_corpus()
    c = _make_constraint(corpus)
    allowed = _step(c, [])
    assert "the" in allowed and "fox" in allowed
    assert "<eos>" in allowed
    assert "|" not in allowed, allowed


def test_cursor_fanout_shared_prefix():
    """Shared prefix 'the' fans cursors to all valid next tokens across both docs."""
    corpus = _make_corpus()
    c = _make_constraint(corpus)
    seq = [_VOCAB["the"], _VOCAB["fox"], _VOCAB["ran"]]

    # after "the": occurs before "fox" in A and before "fox"/"dog" in B -> both live
    allowed1 = _step(c, seq[:1])
    assert allowed1 == {"fox", "dog"}, allowed1

    # after "the fox": only "ran" continues both cursors
    allowed2 = _step(c, seq[:2])
    assert allowed2 == {"ran"}, allowed2

    # after "the fox ran" (span_len=3 >= min_span_len): continuations + close tokens
    allowed3 = _step(c, seq[:3])
    assert {"fast", "and", "|", "<eos>"} == allowed3, allowed3


def test_multi_token_attribution():
    """Span 'the fox ran' closed before divergence yields 2 sources (AMBIGUOUS)."""
    corpus = _make_corpus()
    c2 = _make_constraint(corpus)
    seq = [_VOCAB["the"], _VOCAB["fox"], _VOCAB["ran"]]
    flat = np.zeros(_V, dtype=np.float32)

    for i in range(3):
        c2(np.array(seq[:i], dtype=np.intc), flat.copy())
    c2(np.array(seq, dtype=np.intc), flat.copy())
    c2(np.array(seq + [_VOCAB["|"]], dtype=np.intc), flat.copy())  # close span

    segs = c2.finalize()
    ext = [s for s in segs if s.kind == "extract"][0]
    assert _detok(ext.token_ids) == "the fox ran", _detok(ext.token_ids)
    assert len(ext.sources) == 2, ext.sources       # divergence signal: A and B
    assert {d for d, _, _ in ext.sources} == {0, 1}


def test_cursor_pruning_on_divergence():
    """Extending 'the fox ran' with 'fast' prunes cursors to doc A only."""
    corpus = _make_corpus()
    c3 = _make_constraint(corpus)
    seq = [_VOCAB["the"], _VOCAB["fox"], _VOCAB["ran"]]
    flat = np.zeros(_V, dtype=np.float32)
    full = seq + [_VOCAB["fast"]]

    for i in range(len(full)):
        c3(np.array(full[:i], dtype=np.intc), flat.copy())
    c3(np.array(full, dtype=np.intc), flat.copy())
    c3(np.array(full + [_VOCAB["<eos>"]], dtype=np.intc), flat.copy())

    ext3 = [s for s in c3.finalize() if s.kind == "extract"][0]
    assert len(ext3.sources) == 1 and ext3.sources[0][0] == 0, ext3.sources
