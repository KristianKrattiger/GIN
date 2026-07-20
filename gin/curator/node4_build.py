"""Deterministic builder: source manifest -> corpus_node4.json dict.

Pure and network-free. The manifest (data/curator/node4_sources.yaml) is the
reviewable artifact; this module turns it into the same schema node1-3 use, so
load_corpus_chunks and the whole curator path work unchanged.
"""
from __future__ import annotations

import hashlib

NODE_ID = "node_4_contested"
_REQUIRED = ("topic", "stance", "source", "author", "date", "url", "domain", "type", "chunks")
_VALID_DOMAINS = {"climate_policy", "energy_policy", "fiscal_policy"}
_VALID_TYPES = {"opinion", "advocacy", "analysis"}


def compute_global_id(source: str, author: str, date: str) -> str:
    digest = hashlib.sha256(f"{source}|{author}|{date}".encode()).hexdigest()
    return "gid_" + digest[:16]


def _validate(manifest: list[dict]) -> None:
    for i, e in enumerate(manifest):
        for key in _REQUIRED:
            if key not in e:
                raise ValueError(f"manifest entry {i} missing required key {key!r}")
        if e["stance"] not in {"pro", "con"}:
            raise ValueError(f"manifest entry {i} bad stance {e['stance']!r} (pro|con)")
        if e["domain"] not in _VALID_DOMAINS:
            raise ValueError(
                f"manifest entry {i} ({e['topic']}) bad domain {e['domain']!r} "
                f"(expected one of {sorted(_VALID_DOMAINS)})"
            )
        if e["type"] not in _VALID_TYPES:
            raise ValueError(
                f"manifest entry {i} ({e['topic']}) bad type {e['type']!r} "
                f"(expected one of {sorted(_VALID_TYPES)})"
            )
        if not e["chunks"]:
            raise ValueError(f"manifest entry {i} ({e['topic']}) has no chunks")
    # Each topic appears exactly twice: one pro, one con.
    by_topic: dict[str, list[str]] = {}
    for e in manifest:
        by_topic.setdefault(e["topic"], []).append(e["stance"])
    for topic, stances in by_topic.items():
        if sorted(stances) != ["con", "pro"]:
            raise ValueError(f"topic {topic!r} must appear exactly once pro and once con, got {stances}")
    # Topic pairs must be adjacent (entries 2k, 2k+1 share a topic).
    for k in range(0, len(manifest), 2):
        if manifest[k]["topic"] != manifest[k + 1]["topic"]:
            raise ValueError(
                f"topic pair not adjacent at entries {k},{k + 1}: "
                f"{manifest[k]['topic']!r} vs {manifest[k + 1]['topic']!r}"
            )


def build_node4(manifest: list[dict]) -> dict:
    if len(manifest) % 2 != 0:
        raise ValueError(f"manifest must be pro/con pairs (even length), got {len(manifest)}")
    _validate(manifest)
    documents = []
    seen_gids: dict[str, str] = {}
    for idx, e in enumerate(manifest, start=1):
        doc_id = f"n4_doc_{idx:03d}"
        gid = compute_global_id(e["source"], e["author"], e["date"])
        if gid in seen_gids:
            raise ValueError(
                f"global_id collision {gid} between {seen_gids[gid]} and {doc_id} "
                f"(identical source|author|date)"
            )
        seen_gids[gid] = doc_id
        chunks = [
            {"chunk_id": f"{doc_id}_c{j:03d}", "position": j, "text": text}
            for j, text in enumerate(e["chunks"])
        ]
        documents.append({
            "doc_id": doc_id,
            "global_id": gid,
            "source": e["source"],
            "url": e["url"],
            "node": NODE_ID,
            "metadata": {
                "author": e["author"],
                "category": e["topic"],
                "date": e["date"],
                "domain": e["domain"],
                "type": e["type"],
                "stance": e["stance"],
                "topic": e["topic"],
            },
            "chunks": chunks,
        })
    return {"node_id": NODE_ID, "documents": documents}
