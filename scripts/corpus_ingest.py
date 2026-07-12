#!/usr/bin/env python3
"""Load synthetic YAML corpus into cold, warm, and hot tiers."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gin.corpus.db import DatabaseUnavailableError, ensure_postgres
from gin.corpus.ingest import ingest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest synthetic corpus YAML into GIN tiers")
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data" / "synthetic",
        help="YAML file or directory of YAML files",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip embedding generation (warm/cold only)",
    )
    parser.add_argument(
        "--no-edges",
        action="store_true",
        help="Ingest documents/chunks only; skip YAML edge rows (use cartographer_scan)",
    )
    args = parser.parse_args()

    try:
        ensure_postgres()
    except DatabaseUnavailableError as exc:
        print(exc, file=sys.stderr)
        return 1

    stats = ingest_path(
        args.source,
        embed=not args.no_embed,
        ingest_edges=not args.no_edges,
    )
    print("Ingest complete:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
