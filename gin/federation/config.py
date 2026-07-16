"""Per-node configuration.

Each node process is sovereign: its own Postgres database, cold store, model,
port, and peer list. Peer authentication is mutual TLS: each node presents
its own self-signed certificate (cert_path/key_path) and trusts only the
specific pinned certificate configured for each peer — no CA, no shared
secret. The corpus tier reads GIN_DATABASE_URL / GIN_COLD_PATH from the
environment at call time (gin/corpus/db.py), so ``apply_env`` must be
called before the process touches the database.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PeerConfig:
    node_id: str
    url: str  # base URL, no trailing slash
    pinned_cert_path: str = ""


@dataclass(frozen=True)
class NodeConfig:
    node_id: str
    host: str
    port: int
    database_url: str
    cold_path: str
    model_path: str
    n_gpu_layers: int
    n_ctx: int
    cert_path: str
    key_path: str
    peer_timeout_s: float
    peers: tuple[PeerConfig, ...]
    chat_template: str = "mistral"
    anchor_sync_interval_s: float = 30.0
    trust_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    trust_gate_threshold: float = 0.5


def load_node_config(path: str | Path) -> NodeConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    peers = tuple(
        PeerConfig(
            node_id=p["node_id"],
            url=str(p["url"]).rstrip("/"),
            pinned_cert_path=p["pinned_cert_path"],
        )
        for p in raw.get("peers", [])
    )
    return NodeConfig(
        node_id=raw["node_id"],
        host=raw.get("host", "127.0.0.1"),
        port=int(raw["port"]),
        database_url=raw["database_url"],
        cold_path=raw["cold_path"],
        model_path=raw.get("model_path", ""),
        n_gpu_layers=int(raw.get("n_gpu_layers", -1)),
        n_ctx=int(raw.get("n_ctx", 4096)),
        cert_path=raw["cert_path"],
        key_path=raw["key_path"],
        peer_timeout_s=float(raw.get("peer_timeout_s", 300.0)),
        peers=peers,
        chat_template=raw.get("chat_template", "mistral"),
        anchor_sync_interval_s=float(raw.get("anchor_sync_interval_s", 30.0)),
        trust_weights=raw.get("trust_weights") or {},
        trust_gate_threshold=float(raw.get("trust_gate_threshold") or 0.5),
    )


def apply_env(config: NodeConfig) -> None:
    """Point the corpus tier at this node's database and cold store.

    Must run before the first DB connection; db.py reads env at call time.
    """
    os.environ["GIN_DATABASE_URL"] = config.database_url
    os.environ["GIN_COLD_PATH"] = config.cold_path
