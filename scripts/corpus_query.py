#!/usr/bin/env python3
"""Ad-hoc hybrid retrieval against the ingested corpus."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gin.corpus.db import DatabaseUnavailableError, ensure_postgres
from gin.corpus.retrieve import retrieve


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the GIN corpus")
    parser.add_argument("query", help="Natural language query")
    parser.add_argument("-k", type=int, default=10, help="Number of results")
    parser.add_argument("--eval-layer", default=None, help="Filter by eval_layer")
    args = parser.parse_args()

    try:
        ensure_postgres()
    except DatabaseUnavailableError as exc:
        print(exc, file=sys.stderr)
        return 1

    filters = {}
    if args.eval_layer:
        filters["eval_layer"] = args.eval_layer

    hits = retrieve(args.query, k=args.k, filters=filters or None)
    if not hits:
        print("No results.")
        return 0

    for i, hit in enumerate(hits, start=1):
        print(f"{i}. [{hit.chunk_id}] rrf={hit.rrf_score:.4f} layer={hit.eval_layer}")
        print(f"   {hit.outlet} — {hit.title}")
        print(f"   {hit.head_sentence}")
        print(f"   hash={hit.content_hash[:12]}...")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
