"""Tests for admitted anchor consumption in the read path."""
from uuid import uuid4

from gin.corpus.divergence import (
    divergence_starts_from_edge_anchors,
    sentence_start_for_whitespace_anchor,
)
from gin.corpus.materialize import materialize_synthesis_bundle
from gin.corpus.models import ChunkHit, EdgeRecord, SynthesisBundle
from sear.corpus import Corpus

DOC_ID = uuid4()


def _tok_factory():
    vocab: dict[str, int] = {}

    def tok(b: bytes) -> list[int]:
        return [vocab.setdefault(w, len(vocab) + 1) for w in b.decode().split()]

    return tok


def _hit(chunk_id: str, text: str) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        doc_id=DOC_ID,
        text=text,
        head_sentence="",
        eval_layer="realism",
        eval_tag=None,
        content_hash="",
        outlet="o",
        title="t",
    )


def test_admitted_anchors_seed_divergence_starts():
    tok = _tok_factory()
    institutional = (
        "In 2023 wildfires burned 2,693,910 acres across the United States. "
        "Agency officials credited improved suppression tactics."
    )
    grassroots = (
        "Neighborhood clinics reported elevated respiratory visits. "
        "Elderly residents faced severe smoke exposure inside crowded shelters."
    )
    hits = [_hit("i:0", institutional), _hit("g:0", grassroots)]
    corpus = Corpus.from_chunks([(h.chunk_id, h.text) for h in hits], tokenize=tok)
    edge = EdgeRecord(
        "i:0",
        "g:0",
        "contradicts",
        src_anchor=(0, 12),
        dst_anchor=(0, 10),
    )
    div = divergence_starts_from_edge_anchors(
        hits, [(hits[0], hits[1], edge)], corpus, tok
    )
    assert div.get(0) == {0}
    assert 1 in div and 0 in div[1]


def test_materialize_uses_admitted_anchors_for_divergent_bundle():
    tok = _tok_factory()
    institutional = (
        "In 2023 wildfires burned 2,693,910 acres across the United States. "
        "Agency officials credited improved suppression tactics."
    )
    grassroots = (
        "Neighborhood clinics reported elevated respiratory visits. "
        "Elderly residents faced severe smoke exposure inside crowded shelters."
    )
    left = _hit("i:0", institutional)
    right = _hit("g:0", grassroots)
    edge = EdgeRecord(
        "i:0",
        "g:0",
        "contradicts",
        src_anchor=(0, 12),
        dst_anchor=(0, 10),
    )
    bundle = SynthesisBundle(
        hits=[left, right],
        edges=[edge],
        mode="divergent",
        pairs=[(left, right, edge)],
    )
    _corpus, ctx = materialize_synthesis_bundle(
        bundle, tok, query="wildfire smoke elderly residents"
    )
    assert (1, 0) in ctx.preferred_starts or (1, 0) in {
        p for poses in ctx.divergence_starts.values() for p in poses
    }


def test_sentence_start_for_whitespace_anchor_first_sentence():
    tok = _tok_factory()
    text = "First sentence here. Second sentence follows."
    corpus = Corpus.from_chunks([("c:0", text)], tokenize=tok)
    start = sentence_start_for_whitespace_anchor(text, (0, 3), tok, corpus, 0)
    assert start == 0
