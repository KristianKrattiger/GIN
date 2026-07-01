"""Tests for divergence zone computation."""
from uuid import uuid4

from gin.corpus.divergence import compute_divergence_zones
from gin.corpus.models import ChunkHit, EdgeRecord
from sear.corpus import Corpus

_VOCAB = {w: i for i, w in enumerate(
    ["Officials", "responded", "Emergency", "confirmed", "142", "98", "treatment",
     "Police", "23", "11", "arrests", "The", "mayor", "briefing", "Council", "review"])}
DOC_ID = uuid4()


def _tok(b: bytes) -> list[int]:
    words = []
    for w in b.decode().split():
        words.append(w.rstrip(".,;:"))
    return [_VOCAB[w] for w in words]


def _hit(chunk_id: str, text: str, outlet: str) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        doc_id=DOC_ID,
        text=text,
        head_sentence="",
        eval_layer="realism",
        eval_tag="incident_divergence",
        content_hash="",
        outlet=outlet,
        title="t",
    )


def test_divergence_marks_treatment_and_arrest_sentences():
    central = (
        "Officials responded. Emergency confirmed 142 treatment. "
        "Police 23 arrests. The mayor briefing."
    )
    metro = (
        "Officials responded. Emergency confirmed 98 treatment. "
        "Police 11 arrests. Council review."
    )
    hits = [_hit("c:0", central, "Central"), _hit("m:0", metro, "Metro")]
    corpus = Corpus.from_chunks(
        [(h.chunk_id, h.text) for h in hits], tokenize=_tok
    )
    edge = EdgeRecord("c:0", "m:0", "contradicts")
    pairs = [(hits[0], hits[1], edge)]
    div, forbidden = compute_divergence_zones(hits, pairs, corpus, _tok)
    assert div[0]
    assert div[1]
    assert forbidden
    tail_tokens = {corpus.docs[d][p] for d, p in forbidden}
    assert _VOCAB["The"] in tail_tokens or _VOCAB["Council"] in tail_tokens
