#!/usr/bin/env python3
"""Validate a generated corpus JSON file (schema + integrity smoke test).

Runs with no DB or model dependency. Checks:
  - top-level node_id / documents present
  - each document has the required fields
  - doc.node matches top-level node_id
  - doc_id and global_id are unique across the corpus
  - global_id follows the "gid_<16 hex>" form
  - chunk positions are contiguous 0..n-1 and ordered
  - chunk_id follows "<doc_id>_c<NNN>" and text is non-empty

Run: python3 scripts/test_corpus.py --node 1
     python3 scripts/test_corpus.py --file corpus_node2.json
Exit code 0 = all checks pass, 1 = failures found.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GID_RE = re.compile(r"^gid_[0-9a-f]{16}$")
REQUIRED_DOC_FIELDS = ("doc_id", "global_id", "source", "url", "node", "metadata", "chunks")
REQUIRED_META_FIELDS = ("domain", "date", "type", "author", "category")


def validate(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot load {path.name}: {exc}"]

    node_id = data.get("node_id")
    if not node_id:
        errors.append("missing top-level 'node_id'")
    docs = data.get("documents")
    if not isinstance(docs, list) or not docs:
        errors.append("missing or empty 'documents' list")
        return errors

    seen_doc_ids: set[str] = set()
    seen_gids: set[str] = set()

    for i, doc in enumerate(docs):
        tag = doc.get("doc_id", f"index {i}")

        for field in REQUIRED_DOC_FIELDS:
            if field not in doc:
                errors.append(f"{tag}: missing field '{field}'")

        did = doc.get("doc_id")
        if did in seen_doc_ids:
            errors.append(f"{tag}: duplicate doc_id")
        seen_doc_ids.add(did)

        gid = doc.get("global_id", "")
        if not GID_RE.match(gid):
            errors.append(f"{tag}: global_id '{gid}' does not match gid_<16 hex>")
        if gid in seen_gids:
            errors.append(f"{tag}: duplicate global_id '{gid}'")
        seen_gids.add(gid)

        if node_id and doc.get("node") != node_id:
            errors.append(f"{tag}: node '{doc.get('node')}' != top-level '{node_id}'")

        meta = doc.get("metadata", {})
        for field in REQUIRED_META_FIELDS:
            if field not in meta:
                errors.append(f"{tag}: metadata missing '{field}'")

        chunks = doc.get("chunks", [])
        if not chunks:
            errors.append(f"{tag}: no chunks")
        for pos, chunk in enumerate(chunks):
            if chunk.get("position") != pos:
                errors.append(
                    f"{tag}: chunk {pos} has position {chunk.get('position')} (expected {pos})"
                )
            expected_cid = f"{did}_c{pos:03d}"
            if chunk.get("chunk_id") != expected_cid:
                errors.append(
                    f"{tag}: chunk {pos} id '{chunk.get('chunk_id')}' (expected '{expected_cid}')"
                )
            if not (chunk.get("text") or "").strip():
                errors.append(f"{tag}: chunk {pos} has empty text")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a corpus JSON file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--node", choices=["1", "2"], help="Validate corpus_node<N>.json")
    group.add_argument("--file", type=Path, help="Path to a corpus JSON file")
    args = parser.parse_args()

    if args.file:
        path = args.file if args.file.is_absolute() else ROOT / args.file
    else:
        node = args.node or "1"
        path = ROOT / f"corpus_node{node}.json"

    errors = validate(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    n_docs = len(data.get("documents", []))
    n_chunks = sum(len(d.get("chunks", [])) for d in data.get("documents", []))

    print(f"corpus: {path.name}  node_id={data.get('node_id')}")
    print(f"  {n_docs} documents, {n_chunks} chunks")

    if errors:
        print(f"\nFAIL — {len(errors)} problem(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nPASS — all schema and integrity checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
