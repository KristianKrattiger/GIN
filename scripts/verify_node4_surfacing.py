"""Hard gate: every node4 topic's thesis pair must reach the curator backlog.

    venv/Scripts/python.exe scripts/verify_node4_surfacing.py \
        --corpus corpus_node1.json corpus_node2.json corpus_node3.json corpus_node4.json

Loads the real SentenceTransformer + NLI proposer (no llama_cpp). Exit 0 iff all
node4 thesis pairs PASS; exit 1 lists the sinkers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gin.cartographer.combined import CombinedRelationProposer
from gin.curator.corpus_json import load_corpus_chunks
from gin.curator.node4_verify import verify_surfacing

# A PASS this deep into the residue is reachable in principle (the gate is
# presence-based, deliberately) but not in practice — a human curator paging
# 20 pairs at a time is very unlikely to ever page this far. Flagged, not
# failed: the pass/fail rule stays presence-based by design.
DEEP_RANK_THRESHOLD = 500


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify node4 issue_frame pairs surface")
    ap.add_argument("--corpus", type=Path, nargs="+",
                    default=[Path("corpus_node1.json"), Path("corpus_node2.json"),
                             Path("corpus_node3.json"), Path("corpus_node4.json")])
    ap.add_argument("--node4", type=Path, default=Path("corpus_node4.json"))
    args = ap.parse_args()

    chunks = load_corpus_chunks(args.corpus)
    node4_docs = json.loads(args.node4.read_text(encoding="utf-8"))["documents"]
    proposer = CombinedRelationProposer()  # real embed + NLI, lazily loaded

    results = verify_surfacing(chunks, node4_docs, proposer)
    sinks = [r for r in results if not r.passed]
    shallow = [r for r in results if r.passed and r.rank < DEEP_RANK_THRESHOLD]
    deep = [r for r in results if r.passed and r.rank >= DEEP_RANK_THRESHOLD]
    for r in sorted(results, key=lambda r: (r.passed, r.topic)):
        if not r.passed:
            mark = "SINK"
        elif r.rank >= DEEP_RANK_THRESHOLD:
            mark = f"PASS rank={r.rank} DEEP"
        else:
            mark = f"PASS rank={r.rank}"
        print(f"{mark:<24} {r.topic}")
    print(f"\n{len(results) - len(sinks)}/{len(results)} thesis pairs surfaced"
          f" ({len(shallow)} within first {DEEP_RANK_THRESHOLD}, {len(deep)} deeper)")
    if sinks:
        print("HARD GATE FAILED — sharpen sources for: " + ", ".join(r.topic for r in sinks))
        sys.exit(1)
    print("HARD GATE PASSED")


if __name__ == "__main__":
    main()
