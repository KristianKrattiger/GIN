"""Launch the local curator labeling app.

    venv/Scripts/python.exe scripts/curator_serve.py

Serves http://127.0.0.1:8600/curator/ over the fixture chunk set, appending
labels to data/curator/labels.jsonl. Seeds the ~33 existing labels on first run
so already-known pairs are not re-surfaced.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from gin.cartographer import labeled_set
from gin.cartographer.combined import CombinedRelationProposer
from gin.curator.app import create_curator_app
from gin.curator.candidates import OfflineCandidateSource
from gin.curator.seed import seed_store
from gin.curator.signals import pair_signals
from gin.curator.store import Store

DEFAULT_LOG = Path("data/curator/labels.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser(description="GIN curator labeling app")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--no-seed", action="store_true", help="skip seeding existing gold")
    args = ap.parse_args()

    store = Store(args.log)
    if not args.no_seed:
        added = seed_store(store)
        print(f"seeded {added} existing labels into {args.log}")

    proposer = CombinedRelationProposer()  # real embed + NLI, lazily loaded
    source = OfflineCandidateSource(labeled_set.chunks())
    app = create_curator_app(
        store=store,
        source=source,
        signals_fn=lambda a, b: pair_signals(a, b, proposer),
        curator="kristian",
    )
    print(f"curator UI: http://{args.host}:{args.port}/curator/")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
