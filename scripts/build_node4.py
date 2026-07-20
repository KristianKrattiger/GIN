"""Build corpus_node4.json from the approved source manifest.

    venv/Scripts/python.exe scripts/build_node4.py \
        --manifest data/curator/node4_sources.yaml --out corpus_node4.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from gin.curator.node4_build import build_node4


def main() -> None:
    ap = argparse.ArgumentParser(description="Build corpus_node4.json from manifest")
    ap.add_argument("--manifest", type=Path, default=Path("data/curator/node4_sources.yaml"))
    ap.add_argument("--out", type=Path, default=Path("corpus_node4.json"))
    args = ap.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    corpus = build_node4(manifest)
    args.out.write_text(json.dumps(corpus, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(corpus['documents'])} docs to {args.out}")


if __name__ == "__main__":
    main()
