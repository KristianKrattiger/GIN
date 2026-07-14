"""Serve one GIN federation node.

Usage:
    python scripts/node_serve.py --config config/node_a.yaml

Loads the node config, points the corpus tier at this node's database and
cold store (apply_env BEFORE any DB touch), loads the local model, and serves
POST /v1/federated/query.
"""
from __future__ import annotations

import argparse
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
    apply_env(config)

    # Imports after apply_env so the first DB connection sees this node's URL.
    import uvicorn
    from llama_cpp import Llama

    from gin.corpus.fingerprint import corpus_fingerprint
    from gin.eval.arms import ArmConfig
    from gin.federation.client import HttpPeerClient
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
        peer_client=HttpPeerClient(config.shared_secret, config.peer_timeout_s),
        corpus_fingerprint=fingerprint,
    )
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
