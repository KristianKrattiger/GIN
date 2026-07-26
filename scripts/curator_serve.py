"""Launch the local curator labeling app.

    venv/Scripts/python.exe scripts/curator_serve.py

Serves http://127.0.0.1:8600/curator/ over the fixture chunk set, appending
labels to data/curator/labels.jsonl. Seeds the ~33 existing labels on first run
so already-known pairs are not re-surfaced.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

from gin.cartographer import labeled_set
from gin.cartographer.combined import CombinedRelationProposer
from gin.curator.app import create_curator_app
from gin.curator.candidates import OfflineCandidateSource
from gin.curator.corpus_json import load_corpus_chunks
from gin.curator.residue import EscalationResidueCandidateSource
from gin.curator.seed import seed_store
from gin.curator.signals import pair_signals
from gin.curator.store import Store

DEFAULT_LOG = Path("data/curator/labels.jsonl")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="GIN curator labeling app")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--no-seed", action="store_true", help="skip seeding existing gold")
    ap.add_argument("--source", choices=["labeled-set", "escalation-residue", "same-story"],
                    default="labeled-set", help="candidate source")
    ap.add_argument("--corpus", type=Path, nargs="+",
                    default=[Path("corpus_node1.json"), Path("corpus_node2.json"),
                             Path("corpus_node3.json"), Path("corpus_node4.json")],
                    help="corpus_node*.json exports for the escalation-residue source")
    ap.add_argument("--curator", default="kristian",
                    help="name stamped on every LabelRecord this instance writes")
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()

    store = Store(args.log)
    if not args.no_seed:
        added = seed_store(store)
        print(f"seeded {added} existing labels into {args.log}")

    proposer = CombinedRelationProposer()  # real embed + NLI, lazily loaded
    if args.source == "escalation-residue":
        try:
            chunks = load_corpus_chunks(args.corpus)
        except (FileNotFoundError, ValueError) as exc:
            sys.exit(f"error: {exc}")
        # Share the one proposer with the residue source so the whole run loads
        # a single model set and the displayed signals reflect the same
        # story-gating the residue filter uses.
        # Retain the full filtered residue (uncapped) so high-cosine framing
        # issue_frame pairs — which NLI cannot rank to the top — still reach a
        # curator paging the backlog rather than being truncated by the cap.
        cap = max(1, len(chunks) * (len(chunks) - 1) // 2)
        source = EscalationResidueCandidateSource(chunks, proposer=proposer, max_candidates=cap)
        print(f"escalation-residue source over {len(source.chunks())} corpus chunks")
    elif args.source == "same-story":
        try:
            chunks = load_corpus_chunks(args.corpus)
        except (FileNotFoundError, ValueError) as exc:
            sys.exit(f"error: {exc}")
        from gin.cartographer.relatedness import make_same_story
        from gin.curator.same_story import SameStoryCandidateSource
        from gin.curator.text_index import default_text_index

        # Each event's shared lede appears in that event's 3-4 reports, giving
        # its tokens document frequency 3-4 within this corpus's chunks alone --
        # not rare enough to anchor the very event it repeats across. Build the
        # predicate over a realistic corpus (these chunks plus the standard
        # offline text index) so the shared lede is rare relative to the whole
        # corpus, the way it will be in production. Only set it when unset, same
        # guard as wire_same_story, so an injected provider is left untouched.
        if proposer.same_story is None:
            proposer.same_story = make_same_story(
                [ch.text for ch in chunks] + list(default_text_index().values())
            )
        source = SameStoryCandidateSource(chunks, proposer=proposer)
        print(f"same-story source over {len(source.chunks())} corpus chunks")
    else:
        source = OfflineCandidateSource(labeled_set.chunks())
    app = create_curator_app(
        store=store,
        source=source,
        signals_fn=lambda a, b: pair_signals(a, b, proposer),
        curator=args.curator,
    )
    print(f"curator UI: http://{args.host}:{args.port}/curator/")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
