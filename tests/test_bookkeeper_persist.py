"""Bookkeeper Postgres persistence."""
from uuid import uuid4

import psycopg

from gin.bookkeeper import (
    AdmissionCode,
    Bookkeeper,
    ensure_edge_schema,
    load_graph,
    sync_admissions,
)
from gin.bookkeeper.models import Provenance, now_iso
from gin.cartographer.models import EdgeProposal, Relation
from gin.corpus.models import ChunkDraft, EvalLayer
from gin.corpus import warm
from gin.corpus.db import connect


def _ingest_one_chunk(conn, doc_uuid, chunk_id: str, text: str) -> None:
    idx = int(chunk_id.rsplit(":", 1)[-1])
    warm.upsert_chunk(
        conn,
        ChunkDraft(
            chunk_id=chunk_id,
            doc_id=str(doc_uuid),
            chunk_index=idx,
            text=text,
            head_sentence=text.split(".")[0] + ".",
            eval_layer=EvalLayer.REALISM,
            eval_tag=None,
            content_hash=chunk_id,
        ),
        doc_uuid,
    )


def test_ensure_edge_schema_idempotent(isolated_db):
    with connect() as conn:
        ensure_edge_schema(conn)
        ensure_edge_schema(conn)
        conn.commit()


def test_sync_admissions_round_trip(isolated_db):
    doc_uuid = uuid4()
    with connect() as conn:
        warm.upsert_document(
            conn,
            doc_id=str(doc_uuid),
            content_hash="hash-a",
            outlet="A",
            title="A",
        )
        row = conn.execute(
            "SELECT doc_id FROM documents WHERE content_hash = %s", ("hash-a",)
        ).fetchone()
        doc_uuid = row[0]
        _ingest_one_chunk(conn, doc_uuid, "a:0", "Alpha sentence one.")
        _ingest_one_chunk(conn, doc_uuid, "b:1", "Beta sentence two.")
        conn.commit()

        proposal = EdgeProposal(
            src_chunk_id="a:0",
            dst_chunk_id="b:1",
            relation=Relation.CONTRADICTS,
            method="test",
            confidence=0.9,
        )
        bk = Bookkeeper()
        registry = {"a:0": 3, "b:1": 3}
        result = bk.admit(proposal, registry=registry)
        assert result.admitted

        counts = sync_admissions(conn, [result], notes={("a:0", "b:1", "contradicts"): "test note"})
        conn.commit()
        assert counts[AdmissionCode.ADMITTED] == 1

        graph = load_graph(conn, edge_types=["contradicts"])
        assert len(graph) == 1
        edge = graph.edges()[0]
        assert edge.provenance.proposer == "test"
        assert edge.provenance.confidence == 0.9
