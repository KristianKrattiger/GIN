"""Tag the 7 pre-relation_class seed contradicts by register.

    python scripts/frames_backfill.py

Idempotent: re-running appends nothing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from gin.curator.store import Store
from gin.frames.backfill import backfill_seed_classes

DEFAULT_LOG = Path("data/curator/labels.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill seed relation_class by register")
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = ap.parse_args()
    store = Store(args.log)
    n = backfill_seed_classes(store)
    print(f"appended {n} superseding record(s) to {args.log}")


if __name__ == "__main__":
    main()
