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
from gin.curator.text_index import df_corpus_texts


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify node5 pairs surface")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "curator" / "node5_events.yaml")
    ap.add_argument("--corpus", type=Path, default=ROOT / "corpus_node5.json")
    args = ap.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    chunks = load_corpus_chunks([args.corpus])
    proposer = CombinedRelationProposer()
    # Each event's shared lede appears in that event's 3-4 reports, giving its
    # tokens df 3-4 within the 38 node5 chunks alone -- above _rare_df_ceiling(38)
    # == 2, so the lede cannot anchor its own event. Build the predicate over a
    # realistic corpus (the standard offline text index, which already contains
    # node5 via CORPUS_NODES) so df 3-4 is comfortably rare, matching production
    # where the curator is launched over multiple corpora. df_corpus_texts adds
    # only chunks the index lacks, so registered corpora are never counted twice.
    same_story = make_same_story(df_corpus_texts(c.text for c in chunks))
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
