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


def _dynamic_tok_factory():
    """Whitespace tokenizer with a growing vocab, for free-form test text."""
    vocab: dict[str, int] = {}

    def tok(b: bytes) -> list[int]:
        ids = []
        for w in b.decode().split():
            ids.append(vocab.setdefault(w, len(vocab) + 1))
        return ids

    return tok


def test_divergence_fallback_for_structurally_dissimilar_pair():
    # Institutional statistic vs grassroots reframing: no shared lede, near-zero
    # word overlap, so the aligned-sentence >=3-overlap test finds no divergence
    # point. The contradicts pair must still get a two-sided divergence zone --
    # otherwise every doc-unique sentence (including these anchors) falls into
    # the forbidden tail net and the divergent decode refuses.
    tok = _dynamic_tok_factory()
    institutional = "Wildfires burned two million acres nationwide last year."
    grassroots = "Elderly residents faced severe smoke exposure inside crowded shelters."
    hits = [_hit("i:0", institutional, "Bureau"), _hit("g:0", grassroots, "Collective")]
    corpus = Corpus.from_chunks([(h.chunk_id, h.text) for h in hits], tokenize=tok)
    edge = EdgeRecord("i:0", "g:0", "contradicts")
    div, forbidden = compute_divergence_zones(
        hits, [(hits[0], hits[1], edge)], corpus, tok
    )
    assert div.get(0), "institutional side must get a divergence zone"
    assert div.get(1), "grassroots side must get a divergence zone"
    # The anchor sentence starts must remain extractable (not forbidden).
    assert (0, 0) not in forbidden
    assert (1, 0) not in forbidden


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
