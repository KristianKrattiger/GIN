"""E2E: Cartographer-discovered edges drive divergent synthesis."""
from pathlib import Path

import yaml

from gin.bookkeeper import Bookkeeper, sync_admissions
from gin.cartographer.evaluation import _key
from gin.cartographer.models import EdgeProposal, Relation
from gin.cartographer.scan import sentence_anchor, whitespace_token_count
from gin.corpus import warm
from gin.corpus.db import connect
from gin.corpus.ingest import _head_sentence
from gin.corpus.models import ChunkDraft, EvalLayer
from gin.corpus.retrieve import retrieve_for_synthesis

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "fixtures" / "disclosure_framing.yaml"


def _ingest_fixture(conn) -> None:
    spec = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    for doc in spec["documents"]:
        doc_uuid = warm.upsert_document(
            conn,
            doc_id=doc["id"],
            content_hash=f"hash-{doc['id']}",
            outlet=doc.get("outlet", ""),
            title=doc.get("title", doc["id"]),
        )
        for i, text in enumerate(doc["chunks"]):
            cid = f"{doc['id']}:{i}"
            warm.upsert_chunk(
                conn,
                ChunkDraft(
                    chunk_id=cid,
                    doc_id=doc["id"],
                    chunk_index=i,
                    text=text.strip(),
                    head_sentence=_head_sentence(text),
                    eval_layer=EvalLayer.REALISM,
                    eval_tag=doc.get("eval_tag"),
                    content_hash=cid,
                ),
                doc_uuid,
            )


def test_cartographer_edges_enable_divergent_synthesis(isolated_db):
    """Admitted contradicts edges enable divergent retrieve_for_synthesis."""
    spec = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    texts = {}
    for doc in spec["documents"]:
        for i, text in enumerate(doc["chunks"]):
            texts[f"{doc['id']}:{i}"] = text.strip()

    nw_id = "disc_northwind_pr:0"
    comp_id = "disc_northwind_complaint:0"
    proposal = EdgeProposal(
        src_chunk_id=nw_id,
        dst_chunk_id=comp_id,
        relation=Relation.CONTRADICTS,
        method="test:fixture",
        confidence=0.95,
        src_anchor=sentence_anchor(texts[nw_id]),
        dst_anchor=sentence_anchor(texts[comp_id]),
    )

    with connect() as conn:
        _ingest_fixture(conn)
        registry = {cid: whitespace_token_count(t) for cid, t in texts.items()}
        bk = Bookkeeper()
        result = bk.admit(proposal, registry=registry)
        assert result.admitted
        sync_admissions(
            conn,
            [result],
            notes={(nw_id, comp_id, "contradicts"): "fixture edge"},
        )
        conn.commit()

        bundle = retrieve_for_synthesis(
            "Northwind third quarter revenue overstated",
            k_seed=8,
            k_max=8,
        )
        assert bundle.mode == "divergent"
        pair_keys = {
            _key(left.chunk_id, right.chunk_id) for left, right, _e in bundle.pairs
        }
        assert _key(nw_id, comp_id) in pair_keys
