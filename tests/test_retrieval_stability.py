"""Retrieval ordering stability across tied relevance scores."""
from uuid import uuid4

from gin.corpus.models import ChunkHit
from gin.corpus.relevance import rerank_hits_by_query_score

DOC = uuid4()


def _hit(chunk_id: str, text: str, score: float = 0.5) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        doc_id=DOC,
        text=text,
        head_sentence=text,
        eval_layer="realism",
        eval_tag=None,
        content_hash=chunk_id,
        outlet="outlet",
        title="title",
        rrf_score=score,
    )


def test_rerank_tie_breaks_on_chunk_id():
    """Tied query-relevance scores must sort deterministically by chunk_id."""
    hits = [
        _hit("z_chunk:0", "wildfire smoke exposure in downtown district"),
        _hit("a_chunk:0", "wildfire smoke exposure in downtown district"),
        _hit("m_chunk:0", "wildfire smoke exposure in downtown district"),
    ]
    query = "downtown wildfire smoke"
    first = [h.chunk_id for h in rerank_hits_by_query_score(hits, query)]
    second = [h.chunk_id for h in rerank_hits_by_query_score(list(reversed(hits)), query)]
    assert first == second == ["a_chunk:0", "m_chunk:0", "z_chunk:0"]
