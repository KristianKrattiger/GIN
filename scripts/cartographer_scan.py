"""Batch Cartographer scan: propose edges, admit via Bookkeeper, persist to Postgres."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gin.cartographer.relatedness import DEFAULT_RELATEDNESS_FLOOR
from gin.cartographer.scan import DEFAULT_EXCLUDED_DOC_IDS, run_scan
from gin.corpus.db import DatabaseUnavailableError, ensure_postgres


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-id", type=str, default=None, help="Limit scan to one doc_id prefix")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true", help="Propose and admit in memory only")
    parser.add_argument(
        "--cross-outlet-only",
        action="store_true",
        help="Only scan chunk pairs from different document outlets (framing pairs)",
    )
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="Skip IDF relatedness candidate pruning (scan all pairs)",
    )
    parser.add_argument(
        "--relatedness-floor",
        type=float,
        default=DEFAULT_RELATEDNESS_FLOOR,
        help="IDF overlap floor for stage-1 pair pruning",
    )
    parser.add_argument(
        "--exclude-doc-id",
        action="append",
        default=None,
        dest="exclude_doc_ids",
        metavar="DOC_ID",
        help="Exclude chunks from this doc_id (repeatable; defaults to out_of_scope_stub)",
    )
    parser.add_argument(
        "--no-exclude-defaults",
        action="store_true",
        help="Do not exclude default stub docs from scan",
    )
    parser.add_argument(
        "--no-relation-recheck",
        action="store_true",
        help="Skip Bookkeeper semantic relation re-check",
    )
    args = parser.parse_args()

    exclude_doc_ids: list[str] | None = args.exclude_doc_ids
    if exclude_doc_ids is None and not args.no_exclude_defaults:
        exclude_doc_ids = list(DEFAULT_EXCLUDED_DOC_IDS)
    elif exclude_doc_ids is None:
        exclude_doc_ids = []

    try:
        ensure_postgres()
    except DatabaseUnavailableError as exc:
        print(exc, file=sys.stderr)
        return 1

    result = run_scan(
        doc_id=args.doc_id,
        min_confidence=args.min_confidence,
        dry_run=args.dry_run,
        cross_outlet_only=args.cross_outlet_only,
        prune_relatedness=not args.no_prune,
        relatedness_floor=args.relatedness_floor,
        exclude_doc_ids=exclude_doc_ids or None,
        relation_recheck=not args.no_relation_recheck,
    )
    print("[*] Cartographer scan complete:")
    print(f"    elapsed_seconds: {result.elapsed_seconds:.2f}")
    print(f"    pair_count: {result.pair_count}")
    for key, val in sorted(result.counts.items()):
        print(f"    {key}: {val}")
    if result.admitted_edges:
        print("[*] Admitted edges:")
        for edge in result.admitted_edges:
            print(
                f"    {edge['relation']}: {edge['src_chunk_id']} <-> "
                f"{edge['dst_chunk_id']} (conf={edge['confidence']:.3f}, "
                f"{edge['proposer']})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
