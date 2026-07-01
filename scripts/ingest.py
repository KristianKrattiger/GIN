#!/usr/bin/env python3
"""Ingest local JSONL/txt files into immutable corpus store + manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gin.corpus.corpus_manager import CorpusManager


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest corpus documents from local directory")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing .jsonl or .txt files")
    parser.add_argument("--format", type=str, default="auto", choices=["auto", "jsonl", "txt"])
    parser.add_argument("--manifest-version", type=str, default="latest", help="Reserved for compatibility; emits next version")
    parser.add_argument("--dry-run", action="store_true", help="Parse and compute manifest without writing files")
    parser.add_argument("--metadata-defaults", type=Path, default=None, help="Optional JSON file merged into each doc metadata")
    args = parser.parse_args()

    defaults = None
    if args.metadata_defaults:
        defaults = json.loads(args.metadata_defaults.read_text(encoding="utf-8"))

    manager = CorpusManager()
    result = manager.ingest_directory(
        args.input_dir,
        file_format=args.format,
        metadata_defaults=defaults,
        dry_run=args.dry_run,
    )
    print("Ingest complete:")
    print(f"  docs_seen: {result.docs_seen}")
    print(f"  docs_written: {result.docs_written}")
    print(f"  deduped: {result.deduped}")
    print(f"  manifest_version: {result.manifest.info.version}")
    print(f"  manifest_path: {result.manifest_path}")
    if args.dry_run:
        print("  dry_run: true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
