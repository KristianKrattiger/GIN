"""Evaluate Cartographer scan precision/recall against gold hand-curated edges."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gin.cartographer.scan_eval import evaluate_scan_on_conn
from gin.corpus.db import DatabaseUnavailableError, connect, ensure_postgres

DEFAULT_OUT = ROOT / "data" / "eval_runs"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-id", type=str, default=None)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument(
        "--cross-outlet-only",
        action="store_true",
        help="Only score pairs from different document outlets",
    )
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="Skip IDF relatedness candidate pruning",
    )
    parser.add_argument(
        "--no-relation-recheck",
        action="store_true",
        help="Skip Bookkeeper semantic relation re-check",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    try:
        ensure_postgres()
    except DatabaseUnavailableError as exc:
        print(exc, file=sys.stderr)
        return 1

    with connect() as conn:
        result = evaluate_scan_on_conn(
            conn,
            doc_id=args.doc_id,
            min_confidence=args.min_confidence,
            cross_outlet_only=args.cross_outlet_only,
            prune_relatedness=not args.no_prune,
            relation_recheck=not args.no_relation_recheck,
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact = run_dir / "cartographer_scan_metrics.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": ts,
        **result.to_dict(),
    }
    artifact.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    m = result.metrics
    print(f"[*] Cartographer scan evaluation — {ts}")
    print(f"    gold contradicts: {result.gold_count}")
    print(f"    admitted contradicts: {result.admitted_count}")
    print(f"    precision: {m.contradicts_precision}")
    print(f"    recall: {m.contradicts_recall}")
    print(f"    false positives: {len(result.false_positive_keys)}")
    print(f"    anchor discoveries: {len(result.anchor_discovery_keys)}")
    print(f"    missed gold: {len(result.missed_gold_keys)}")
    print(f"    class_c_discrimination: {result.class_c_discrimination}")
    print(f"    artifact: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
