"""Hard gate: every authored node5 pair must reach the curator backlog.

    venv/Scripts/python.exe scripts/verify_node5_surfacing.py

Loads the real embedding + NLI proposer. Exit 0 iff every authored pair surfaces.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from gin.cartographer.combined import CombinedRelationProposer
from gin.cartographer.relatedness import make_same_story
from gin.curator.corpus_json import load_corpus_chunks
from gin.curator.node5_verify import verify_surfacing
from gin.curator.same_story import SameStoryCandidateSource


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify node5 pairs surface")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "curator" / "node5_events.yaml")
    ap.add_argument("--corpus", type=Path, default=ROOT / "corpus_node5.json")
    args = ap.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    chunks = load_corpus_chunks([args.corpus])
    proposer = CombinedRelationProposer()
    same_story = make_same_story([c.text for c in chunks])
    source = SameStoryCandidateSource(chunks, same_story=same_story, proposer=proposer)

    offered = {frozenset((a.chunk_id, b.chunk_id)) for a, b in source.pairs()}
    report = verify_surfacing(manifest, offered)

    print(f"authored {report['authored']} | surfaced {report['surfaced']}")
    if report["missing"]:
        print(f"missing by kind: {report['missing_by_kind']}")
        for src, dst, kind in report["missing"]:
            print(f"  MISSING [{kind}] {src} <-> {dst}")
        print("\nA missing negative is as serious as a missing conflict: without")
        print("them the curator never labels a same-story non-contradiction.")
        return 1
    print("PASS: every authored pair reaches the curator backlog")
    return 0


if __name__ == "__main__":
    sys.exit(main())
