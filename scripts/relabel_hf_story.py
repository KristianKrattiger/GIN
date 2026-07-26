"""Relabel the two housing pairs from issue_frame to story.

    venv/Scripts/python.exe scripts/relabel_hf_story.py

Idempotent: re-running appends nothing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gin.curator.relabel_hf import relabel_hf_to_story
from gin.curator.store import Store

DEFAULT_LOG = ROOT / "data" / "curator" / "labels.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description="Relabel hf_* pairs issue_frame -> story")
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = ap.parse_args()
    print(f"appended {relabel_hf_to_story(Store(args.log))} record(s) to {args.log}")


if __name__ == "__main__":
    main()
