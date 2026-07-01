"""Tests for SEAR corpus materialization from tiered storage hits."""
import numpy as np
import pytest

from gin.corpus.materialize import materialize_corpus
from gin.corpus.models import ChunkHit
from sear.processor import ExtractiveCopyConstraint, NEG_INF


def _word_tokenize(data: bytes) -> list[int]:
    words = data.decode("utf-8").split()
    vocab: dict[str, int] = {}
    ids: list[int] = []
    for word in words:
        if word not in vocab:
            vocab[word] = len(vocab)
        ids.append(vocab[word])
    return ids


def _make_news_hits() -> list[ChunkHit]:
    shared = (
        "RIVERPORT — Officials responded to a downtown incident Tuesday evening. "
        "Emergency services confirmed"
    )
    return [
        ChunkHit(
            chunk_id="incident_centralwire:0",
            doc_id="00000000-0000-0000-0000-000000000001",
            text=shared + " 142 people received treatment at area hospitals.",
            head_sentence=shared,
            eval_layer="realism",
            eval_tag="incident_divergence",
            content_hash="a",
            outlet="CentralWire",
            title="Downtown incident",
        ),
        ChunkHit(
            chunk_id="incident_metrodaily:0",
            doc_id="00000000-0000-0000-0000-000000000002",
            text=shared + " 98 people received treatment at area hospitals.",
            head_sentence=shared,
            eval_layer="realism",
            eval_tag="incident_divergence",
            content_hash="b",
            outlet="MetroDaily",
            title="Downtown incident",
        ),
    ]


def test_from_chunks_uses_stable_chunk_ids():
    hits = _make_news_hits()
    corpus = materialize_corpus(hits, _word_tokenize)
    assert corpus.doc_names == ["incident_centralwire:0", "incident_metrodaily:0"]


def test_shared_prefix_fanout_in_materialized_corpus():
    hits = _make_news_hits()
    corpus = materialize_corpus(hits, _word_tokenize)
    constraint = ExtractiveCopyConstraint(
        corpus,
        prompt_len=0,
        eos_id=999,
        delim_id=998,
        min_span_len=5,
    )
    flat = np.zeros(1000, dtype=np.float32)

    tokens = corpus.docs[0][:8]
    for i in range(len(tokens)):
        constraint(np.array(tokens[:i], dtype=np.intc), flat.copy())

    allowed = constraint(np.array(tokens, dtype=np.intc), flat.copy())
    allowed_ids = {i for i in range(1000) if allowed[i] > NEG_INF / 2}
    assert len(allowed_ids) >= 2
