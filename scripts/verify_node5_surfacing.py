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
from gin.curator.text_index import default_text_index


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
    # realistic corpus (node5 plus the standard offline text index) so df 3-4
    # is comfortably rare, matching production where the curator is launched
    # over multiple corpora.
    #
    # NOTE (was correct before node5 registration, now double-counts node5):
    # this was 38 + 274 = 312 docs, ceiling 10, when default_text_index()
    # already contains node5 (CORPUS_NODES registered it in c039edd). Doubling
    # node5's document frequencies pushes its tokens above the rare ceiling
    # and MASKS cross-event false positives -- it is the reason this gate
    # still passes 42/42 while the true stage-1 false-positive rate on node5
    # is higher. Known, and the user's decision (2026-07-26) is to leave the
    # double-counting in place; see
    # docs/superpowers/specs/2026-07-26-stage1-anchor-findings.md, "Known
    # defect recorded but deliberately NOT fixed."
    same_story = make_same_story([c.text for c in chunks] + list(default_text_index().values()))
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
