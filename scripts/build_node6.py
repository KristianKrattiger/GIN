"""Build corpus_node6.json from the collaborator-authored event manifest.

    ./venv/bin/python scripts/build_node6.py

Thin parametrization of the node5 builder (gin/curator/node5_build.py). The
composition floors differ from node5's on purpose: node6 is weighted toward
update/corroboration pairs (the scarce classes), so it needs fewer authored
conflicts — see docs/node6-collaborator-setup.md, "Composition targets".
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
    ap = argparse.ArgumentParser(description="Build corpus_node6.json from manifest")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "curator" / "node6_events.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "corpus_node6.json")
    args = ap.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    corpus = build_node5(
        manifest,
        min_conflicts=6,
        min_negatives=15,
        node_id="node_6_samestory",
        doc_prefix="n6_doc",
        url_tag="node6",
    )
    args.out.write_text(json.dumps(corpus, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"pair inventory: {pair_inventory(manifest)}")
    print(f"wrote {len(corpus['documents'])} docs to {args.out}")


if __name__ == "__main__":
    main()
