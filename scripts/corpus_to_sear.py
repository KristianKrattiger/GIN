#!/usr/bin/env python3
"""Retrieve corpus chunks and materialize a SEAR token index."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gin.corpus.db import DatabaseUnavailableError, ensure_postgres
from gin.corpus.materialize import materialize_all, materialize_from_retrieval


def _word_tokenize(data: bytes) -> list[int]:
    words = data.decode("utf-8").split()
    vocab: dict[str, int] = {}
    ids: list[int] = []
    for word in words:
        if word not in vocab:
            vocab[word] = len(vocab)
        ids.append(vocab[word])
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize retrieved chunks into SEAR Corpus")
    parser.add_argument("--query", default=None, help="Retrieve top-k by query")
    parser.add_argument("-k", type=int, default=10, help="Top-k for retrieval mode")
    parser.add_argument("--all", action="store_true", help="Materialize all chunks")
    parser.add_argument(
        "--manifest-version",
        type=int,
        default=None,
        help="Load corpus snapshot from manifest version instead of warm DB",
    )
    args = parser.parse_args()

    try:
        ensure_postgres()
    except DatabaseUnavailableError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.all:
        corpus = materialize_all(_word_tokenize, manifest_version=args.manifest_version)
    elif args.query:
        if args.manifest_version is not None:
            parser.error("--manifest-version is only supported with --all")
        corpus = materialize_from_retrieval(args.query, _word_tokenize, k=args.k)
    else:
        parser.error("Provide --query or --all")

    print(f"Corpus documents: {len(corpus.doc_names)}")
    for name in corpus.doc_names:
        print(f"  - {name}")
    print(f"Distinct start tokens: {len(corpus.start_index)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
