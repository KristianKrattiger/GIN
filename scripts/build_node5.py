"""Build corpus_node5.json from the event manifest.

    venv/Scripts/python.exe scripts/build_node5.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from gin.curator.node5_build import build_node5, pair_inventory


def main() -> None:
    ap = argparse.ArgumentParser(description="Build corpus_node5.json from manifest")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "curator" / "node5_events.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "corpus_node5.json")
    args = ap.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    corpus = build_node5(manifest)
    args.out.write_text(json.dumps(corpus, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"pair inventory: {pair_inventory(manifest)}")
    print(f"wrote {len(corpus['documents'])} docs to {args.out}")


if __name__ == "__main__":
    main()
