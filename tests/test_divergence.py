"""Tests for divergence zone computation."""
from uuid import uuid4

from gin.corpus.divergence import compute_divergence_zones
from gin.corpus.models import ChunkHit, EdgeRecord
from gin.corpus.relevance import corpus_idf, idf_weighted_relevance
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


def test_multi_sentence_fallback_anchors_on_relevant_sentence():
    # Real source paragraphs: a lede/anchor sentence buried among tonally
    # consistent but query-irrelevant filler (org-credit, caveat, tail stat).
    # The forbidden-tail net CANNOT rescue a whole-chunk zone: a sentence that
    # is in the divergence zone is skipped by the net (immune), so whole-chunk
    # marking makes every filler sentence a startable divergence anchor. The
    # IDF scorer must narrow the zone to the one relevant sentence per side.
    tok = _dynamic_tok_factory()
    institutional = (
        "In 2023 wildfires burned 2,693,910 acres across the United States. "     # anchor
        "Acreage burned fell below both the five and ten year averages. "
        "The National Interagency Fire Center credited a wet western spring. "
        "Suppression costs still exceeded three billion dollars for the year."
    )
    grassroots = (
        "Our neighborhood resilience hubs opened three new locations this spring. "
        "Elderly and low income residents face the greatest danger from wildfire smoke exposure. "  # anchor
        "Many members cannot afford air purifiers or evacuation. "
        "We thank the volunteers and clinics who kept shelters running."
    )
    query = "What is the main concern about wildfires in the United States?"
    hits = [_hit("i:0", institutional, "Bureau"), _hit("g:0", grassroots, "Collective")]
    corpus = Corpus.from_chunks([(h.chunk_id, h.text) for h in hits], tokenize=tok)
    pairs = [(hits[0], hits[1], EdgeRecord("i:0", "g:0", "contradicts"))]
    inst_starts = sorted(s for (d, s) in corpus.sentence_starts if d == 0)
    grass_starts = sorted(s for (d, s) in corpus.sentence_starts if d == 1)
    assert len(inst_starts) == 4 and len(grass_starts) == 4
    inst_anchor, grass_anchor = inst_starts[0], grass_starts[1]

    # Whole-chunk (no scorer): every filler sentence is marked AND immune to the
    # forbidden net -> it leaks as a startable anchor. This is the failure mode.
    div_all, forb_all = compute_divergence_zones(hits, pairs, corpus, tok)
    assert set(div_all.get(1, set())) == set(grass_starts)
    leaked = [s for s in grass_starts if s != grass_anchor and (1, s) not in forb_all]
    assert leaked == [s for s in grass_starts if s != grass_anchor], (
        "in-zone filler sentences must be immune to the forbidden net "
        "(demonstrates why whole-chunk marking is unsafe)"
    )

    # IDF scorer: narrow to the one relevant sentence per side; the filler falls
    # back out of the zone and the forbidden net catches it.
    idf = corpus_idf([h.text for h in hits])
    scorer = lambda sent: idf_weighted_relevance(sent, query, idf)
    div, forb = compute_divergence_zones(hits, pairs, corpus, tok, sentence_scorer=scorer)
    assert set(div.get(0, set())) == {inst_anchor}
    assert set(div.get(1, set())) == {grass_anchor}
    for s in grass_starts:
        if s != grass_anchor:
            assert (1, s) in forb, "narrowed-out filler must now be forbidden"


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
