"""Serve one GIN federation node.

Usage:
    python scripts/node_serve.py --config config/node_a.yaml

Loads the node config, points the corpus tier at this node's database and
cold store (apply_env BEFORE any DB touch), loads the local model, and serves
POST /v1/federated/query over mutual TLS.
"""
from __future__ import annotations

import argparse
import ssl
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gin.federation.config import apply_env, load_node_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to node YAML")
    args = parser.parse_args()

    config = load_node_config(args.config)
    if not config.peers:
        raise RuntimeError(
            f"node {config.node_id} has no configured peers — refusing to "
            f"serve with no client-cert authentication. Configure at least "
            f"one pinned peer in peers[] before starting this node."
        )
    apply_env(config)

    # Imports after apply_env so the first DB connection sees this node's URL.
    import uvicorn
    from llama_cpp import Llama

    from gin.corpus.fingerprint import corpus_fingerprint
    from gin.corpus.hot import embed_query
    from gin.eval.arms import ArmConfig
    from gin.federation.anchor_store import PostgresPeerAnchorStore, local_anchor_rows
    from gin.federation.certs import build_ca_bundle
    from gin.federation.client import HttpPeerClient
    from gin.federation.peer_summary_store import (
        PostgresPeerSummaryStore, build_local_summary,
    )
    from gin.federation.server import create_app
    from gin.federation.service import answer_query

    print(f"[*] node {config.node_id}: loading {config.model_path} "
          f"(n_gpu_layers={config.n_gpu_layers})")
    llm = Llama(
        model_path=config.model_path,
        n_ctx=config.n_ctx,
        n_gpu_layers=config.n_gpu_layers,
        verbose=False,
    )
    arm_cfg = ArmConfig(chat_template=config.chat_template)
    fingerprint = corpus_fingerprint()
    print(f"[*] node {config.node_id}: corpus fingerprint {fingerprint}")

    app = create_app(
        config,
        answer_fn=lambda q: answer_query(q, llm, arm_cfg),
        peer_client=HttpPeerClient(config.cert_path, config.key_path, config.peer_timeout_s),
        corpus_fingerprint=fingerprint,
        local_anchor_rows=local_anchor_rows,
        peer_anchor_store=PostgresPeerAnchorStore(),
        local_summary=lambda: build_local_summary(config.node_id),
        peer_summary_store=PostgresPeerSummaryStore(),
        embed_query_fn=embed_query,
    )
    ca_bundle = build_ca_bundle(
        [p.pinned_cert_path for p in config.peers],
        Path(config.cert_path).parent / "peer_ca_bundle.pem",
    )
    ssl_kwargs = {"ssl_certfile": config.cert_path, "ssl_keyfile": config.key_path}
    if ca_bundle is not None:
        ssl_kwargs["ssl_ca_certs"] = str(ca_bundle)
        ssl_kwargs["ssl_cert_reqs"] = ssl.CERT_REQUIRED
    else:
        # Unreachable in practice: config.peers is checked non-empty at the
        # top of main(), before the (slow) model load, and build_ca_bundle
        # only returns None for an empty peer list. Kept as a defensive
        # fail-fast in case that invariant is ever broken by future changes.
        raise RuntimeError(
            f"node {config.node_id} has no configured peers — refusing to "
            f"serve with no client-cert authentication. Configure at least "
            f"one pinned peer in peers[] before starting this node."
        )

    uvicorn.run(app, host=config.host, port=config.port, log_level="info", **ssl_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
