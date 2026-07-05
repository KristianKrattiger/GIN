"""Framing-generalization checks (plan §6 #3, rounds 1 and 2).

The whole real-text divergence result rests on three climate pairs that share a
hidden trait: advocacy-vs-institution framings of one environmental event, with
accidental surface overlap ("wildfire", "water") handing the IDF gate its keyword
mass. These tests stress vocabulary distributions nothing like climate, one
variable per round:

  round 1 — data/fixtures/disclosure_framing.yaml: new REGISTER
    (corporate press release vs. securities-regulator complaint);
  round 2 — data/fixtures/housing_framing.yaml: new DOMAIN with sparse surface
    overlap (zoning-technical vs. tenant-organizing) — the organizing side
    shares essentially only the place entity with the query.

For each contradicts pair the tests check that:

  1. the shared query terms clear DIVERGENCE_IDF_FLOOR on BOTH sides (i.e. 0.13
     is not a climate-corpus artifact), with the measured margin asserted so a
     regression that starves a legitimate side of IDF mass fails loudly; and
  2. compute_divergence_zones' fallback + IDF anchor scorer route the pair to a
     two-sided zone narrowed to the query-bearing anchor sentence, not the
     register filler.

Deterministic: no llama.cpp, no DB. IDF here is computed over each fixture's own
chunks, so the absolute margin differs from a full-corpus retrieval run (the
distinctive terms are rarer corpus-wide, so the real margin is only larger);
this asserts the shared terms carry non-trivial IDF mass in the new register and
guards the fixture. The full divergence_fidelity confirmation is the DB eval
(scripts/eval_run.py --queryset data/eval/queryset_framing{2,3}.yaml), pending a
runnable model on this host.
"""
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from gin.corpus.divergence import compute_divergence_zones
from gin.corpus.models import ChunkHit, EdgeRecord
from gin.corpus.relevance import corpus_idf, idf_weighted_relevance
from gin.corpus.retrieve import DIVERGENCE_IDF_FLOOR
from sear.corpus import Corpus

ROOT = Path(__file__).resolve().parents[1]

ROUNDS = [
    pytest.param(
        ROOT / "data" / "fixtures" / "disclosure_framing.yaml",
        ROOT / "data" / "eval" / "queryset_framing2.yaml",
        id="round1-adversarial-register",
    ),
    pytest.param(
        ROOT / "data" / "fixtures" / "housing_framing.yaml",
        ROOT / "data" / "eval" / "queryset_framing3.yaml",
        id="round2-housing-sparse-overlap",
    ),
]


def _dynamic_tok_factory():
    vocab: dict[str, int] = {}

    def tok(b: bytes) -> list[int]:
        return [vocab.setdefault(w, len(vocab) + 1) for w in b.decode().split()]

    return tok


def _load(fixture: Path, queryset: Path):
    spec = yaml.safe_load(fixture.read_text())
    docs = {d["id"]: d["chunks"][0].strip() for d in spec["documents"]}
    queries = {
        q["contradicts_pairs"][0][0]: q["query"]
        for q in yaml.safe_load(queryset.read_text())["queries"]
    }
    pairs = []
    for e in spec["edges"]:
        left_id, right_id = e["src"].split(":")[0], e["dst"].split(":")[0]
        pairs.append(
            {
                "left": docs[left_id],
                "right": docs[right_id],
                "query": queries[e["src"]],
                "note": e["note"],
            }
        )
    return pairs, list(docs.values())


def _hit(chunk_id: str, text: str, outlet: str) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        doc_id=uuid4(),
        text=text,
        head_sentence="",
        eval_layer="realism",
        eval_tag="framing_divergence",
        content_hash="",
        outlet=outlet,
        title="t",
    )


@pytest.mark.parametrize("fixture, queryset", ROUNDS)
def test_pairs_clear_idf_floor(fixture, queryset):
    # Both sides of each pair must clear DIVERGENCE_IDF_FLOOR, or the retrieval
    # gate silently drops the pair to convergent mode and the divergence is lost.
    pairs, all_texts = _load(fixture, queryset)
    idf = corpus_idf(all_texts)
    assert pairs, "fixture produced no contradicts pairs"
    for p in pairs:
        left = idf_weighted_relevance(p["left"], p["query"], idf)
        right = idf_weighted_relevance(p["right"], p["query"], idf)
        # The institutional/technical (left) side and the advocacy/organizing
        # (right) side stress the floor differently by round; assert BOTH clear
        # it, with the measured value in the failure message.
        assert left >= DIVERGENCE_IDF_FLOOR, (
            f"left side under floor ({left:.3f} < {DIVERGENCE_IDF_FLOOR}) "
            f"for: {p['note']}"
        )
        assert right >= DIVERGENCE_IDF_FLOOR, (
            f"right side under floor ({right:.3f} < {DIVERGENCE_IDF_FLOOR}) "
            f"for: {p['note']}"
        )


@pytest.mark.parametrize("fixture, queryset", ROUNDS)
def test_pairs_route_to_two_sided_anchored_zone(fixture, queryset):
    # The fallback + IDF anchor scorer must give a two-sided divergence zone
    # narrowed to the query-bearing anchor sentence (sentence 0 on each side in
    # these fixtures), never the register-filler second sentence. Mirrors the
    # climate check in test_divergence.py on vocabulary nothing like it.
    pairs, all_texts = _load(fixture, queryset)
    idf = corpus_idf(all_texts)
    for p in pairs:
        tok = _dynamic_tok_factory()
        hits = [_hit("l:0", p["left"], "Institutional"), _hit("r:0", p["right"], "Advocacy")]
        corpus = Corpus.from_chunks([(h.chunk_id, h.text) for h in hits], tokenize=tok)
        edge = EdgeRecord("l:0", "r:0", "contradicts")
        scorer = lambda sent: idf_weighted_relevance(sent, p["query"], idf)
        div, forbidden = compute_divergence_zones(
            hits, [(hits[0], hits[1], edge)], corpus, tok, sentence_scorer=scorer
        )
        assert div.get(0), f"left side got no divergence zone: {p['note']}"
        assert div.get(1), f"right side got no divergence zone: {p['note']}"
        # Anchor sentence (sentence 0) start must stay extractable on both sides,
        # and the IDF scorer must narrow to that single anchor (not whole-chunk).
        left_starts = sorted(s for (d, s) in corpus.sentence_starts if d == 0)
        right_starts = sorted(s for (d, s) in corpus.sentence_starts if d == 1)
        assert (0, left_starts[0]) not in forbidden
        assert (1, right_starts[0]) not in forbidden
        assert set(div.get(0)) == {left_starts[0]}, f"left anchor not narrowed: {p['note']}"
        assert set(div.get(1)) == {right_starts[0]}, f"right anchor not narrowed: {p['note']}"
