"""Ingest pipeline — YAML/JSON corpus into cold, warm, and hot tiers."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from . import cold, hot, warm
from .corpus_manager import CorpusManager, IngestResult
from .db import cold_path, transaction
from .models import ChunkDraft, DocumentDraft, EdgeDraft, EdgeType, EvalLayer


def _head_sentence(text: str) -> str:
    match = re.search(r"[^.!?]+[.!?]", text.strip())
    return match.group(0).strip() if match else text.strip().split("\n", 1)[0]


def _parse_eval_layer(raw: str) -> EvalLayer:
    return EvalLayer(raw.strip())


def _parse_edge_type(raw: str) -> EdgeType:
    return EdgeType(raw.strip())


def load_yaml(path: Path) -> tuple[list[DocumentDraft], list[EdgeDraft]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    documents: list[DocumentDraft] = []
    for item in data.get("documents", []):
        chunks = item.get("chunks")
        if chunks is None and "body" in item:
            chunks = [item["body"]]
        if not chunks:
            raise ValueError(f"document {item.get('id')} has no chunks")
        documents.append(
            DocumentDraft(
                doc_id=item["id"],
                outlet=item.get("outlet", ""),
                title=item.get("title", item["id"]),
                eval_layer=_parse_eval_layer(item.get("eval_layer", "realism")),
                source_uri=item.get("source_uri", str(path)),
                source_type=item.get("source_type", "synthetic"),
                chunks=[c.strip() for c in chunks],
                eval_tag=item.get("eval_tag"),
            )
        )

    edges: list[EdgeDraft] = []
    for edge in data.get("edges", []):
        edges.append(
            EdgeDraft(
                src_chunk_id=edge["src"],
                dst_chunk_id=edge["dst"],
                edge_type=_parse_edge_type(edge["type"]),
                note=edge.get("note", ""),
            )
        )
    return documents, edges


def load_json(path: Path) -> tuple[list[DocumentDraft], list[EdgeDraft]]:
    """Load a node corpus manifest (corpus_node*.json) into DocumentDrafts.

    Maps the fetched-corpus schema to the ingest model:
      source   -> title           node              -> outlet (federation node)
      url      -> source_uri       metadata.type     -> source_type
      metadata.category -> eval_tag
    ``outlet`` is the document's node id (node_1_institutional / node_2_grassroots)
    so the eval's chunk->outlet map is the federation boundary; the per-source
    name is preserved in ``title``.
    Chunk objects are ordered by their ``position`` field; chunk ``text`` becomes
    the ingest chunk body (warm-tier chunk ids remain ``<doc_id>:<index>``).
    JSON manifests carry no edges, so an empty edge list is returned.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    documents: list[DocumentDraft] = []
    for item in data.get("documents", []):
        raw_chunks = item.get("chunks")
        if not raw_chunks:
            raise ValueError(f"document {item.get('doc_id')} has no chunks")
        ordered = sorted(raw_chunks, key=lambda c: c.get("position", 0))
        texts = [c["text"].strip() for c in ordered]
        meta = item.get("metadata", {})
        documents.append(
            DocumentDraft(
                doc_id=item["doc_id"],
                outlet=item.get("node") or meta.get("author", ""),
                title=item.get("source", item["doc_id"]),
                eval_layer=_parse_eval_layer(item.get("eval_layer", "realism")),
                source_uri=item.get("url", str(path)),
                source_type=meta.get("type", "curated"),
                chunks=texts,
                eval_tag=meta.get("category"),
            )
        )
    return documents, []


def load_source(path: Path) -> tuple[list[DocumentDraft], list[EdgeDraft]]:
    if path.is_file():
        if path.suffix.lower() == ".json":
            return load_json(path)
        return load_yaml(path)
    docs: list[DocumentDraft] = []
    edges: list[EdgeDraft] = []
    for yaml_file in sorted(path.glob("*.yaml")):
        d, e = load_yaml(yaml_file)
        docs.extend(d)
        edges.extend(e)
    for json_file in sorted(path.glob("*.json")):
        d, e = load_json(json_file)
        docs.extend(d)
        edges.extend(e)
    return docs, edges


def ingest_documents(
    documents: list[DocumentDraft],
    edges: list[EdgeDraft],
    *,
    cold_root: Path | None = None,
    embed: bool = True,
) -> dict[str, Any]:
    root = cold_root or cold_path()
    stats: dict[str, Any] = {
        "documents": 0,
        "chunks": 0,
        "edges": 0,
        "cold_blobs_written": 0,
        "embeddings_written": 0,
    }

    with transaction() as conn:
        run_id = warm.start_ingest_run(conn)
        try:
            for doc in documents:
                full_text = "\n\n".join(doc.chunks)
                doc_bytes = full_text.encode("utf-8")
                doc_hash, doc_created = cold.store(doc_bytes, root)
                if doc_created:
                    stats["cold_blobs_written"] += 1

                doc_uuid = warm.upsert_document(
                    conn,
                    doc_id=doc.doc_id,
                    content_hash=doc_hash,
                    outlet=doc.outlet,
                    title=doc.title,
                    source_uri=doc.source_uri,
                    source_type=doc.source_type,
                )
                stats["documents"] += 1

                for index, chunk_text in enumerate(doc.chunks):
                    chunk_id = f"{doc.doc_id}:{index}"
                    chunk_bytes = chunk_text.encode("utf-8")
                    chunk_hash, chunk_created = cold.store(chunk_bytes, root)
                    if chunk_created:
                        stats["cold_blobs_written"] += 1
                    draft = ChunkDraft(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        chunk_index=index,
                        text=chunk_text,
                        eval_layer=doc.eval_layer,
                        head_sentence=_head_sentence(chunk_text),
                        eval_tag=doc.eval_tag,
                        content_hash=chunk_hash,
                    )
                    warm.upsert_chunk(conn, draft, doc_uuid)
                    stats["chunks"] += 1
                    if embed:
                        hot.embed_and_store(conn, chunk_id, chunk_text)
                        stats["embeddings_written"] += 1

            for edge in edges:
                warm.upsert_edge(conn, edge)
                stats["edges"] += 1

            warm.finish_ingest_run(conn, run_id, "completed", stats)
        except Exception:
            warm.finish_ingest_run(conn, run_id, "failed", stats)
            raise

    return stats


def ingest_path(source: Path, *, embed: bool = True) -> dict[str, Any]:
    documents, edges = load_source(source)
    return ingest_documents(documents, edges, embed=embed)


def ingest_local_directory(
    source: Path,
    *,
    file_format: str = "auto",
    metadata_defaults: dict[str, Any] | None = None,
    dry_run: bool = False,
    manager: CorpusManager | None = None,
) -> IngestResult:
    """Ingest local JSONL/txt docs into immutable store + manifest snapshots."""
    mgr = manager or CorpusManager()
    return mgr.ingest_directory(
        source,
        file_format=file_format,
        metadata_defaults=metadata_defaults,
        dry_run=dry_run,
    )
