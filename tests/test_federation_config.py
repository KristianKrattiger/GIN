# tests/test_federation_config.py
"""NodeConfig loading and env application."""
import os

from gin.federation.config import NodeConfig, PeerConfig, apply_env, load_node_config

_YAML = """\
node_id: node_a
host: 127.0.0.1
port: 8471
database_url: postgresql://gin:gin@localhost:5432/gin_node_a
cold_path: data/cold_node_a
model_path: data/models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf
n_gpu_layers: -1
n_ctx: 4096
cert_path: certs/node_a/cert.pem
key_path: certs/node_a/key.pem
peer_timeout_s: 300
peers:
  - node_id: node_b
    url: http://127.0.0.1:8472/
    pinned_cert_path: certs/node_b/cert.pem
"""


def test_load_node_config(tmp_path):
    p = tmp_path / "node_a.yaml"
    p.write_text(_YAML, encoding="utf-8")
    cfg = load_node_config(p)
    assert cfg.node_id == "node_a"
    assert cfg.port == 8471
    assert cfg.cert_path == "certs/node_a/cert.pem"
    assert cfg.key_path == "certs/node_a/key.pem"
    assert cfg.peers == (
        PeerConfig(
            node_id="node_b", url="http://127.0.0.1:8472",
            pinned_cert_path="certs/node_b/cert.pem",
        ),
    )
    assert cfg.peer_timeout_s == 300.0
    assert cfg.chat_template == "mistral"  # default


def test_anchor_sync_interval_default(tmp_path):
    p = tmp_path / "node_a.yaml"
    p.write_text(_YAML, encoding="utf-8")
    cfg = load_node_config(p)
    assert cfg.anchor_sync_interval_s == 30.0


def test_anchor_sync_interval_override(tmp_path):
    p = tmp_path / "node_a.yaml"
    p.write_text(_YAML + "anchor_sync_interval_s: 5\n", encoding="utf-8")
    cfg = load_node_config(p)
    assert cfg.anchor_sync_interval_s == 5.0


def test_apply_env(tmp_path, monkeypatch):
    p = tmp_path / "node_a.yaml"
    p.write_text(_YAML, encoding="utf-8")

    original_db_url = os.environ.get("GIN_DATABASE_URL")
    original_cold_path = os.environ.get("GIN_COLD_PATH")

    try:
        cfg = load_node_config(p)
        apply_env(cfg)
        assert os.environ["GIN_DATABASE_URL"].endswith("/gin_node_a")
        assert os.environ["GIN_COLD_PATH"] == "data/cold_node_a"
    finally:
        if original_db_url is None:
            os.environ.pop("GIN_DATABASE_URL", None)
        else:
            os.environ["GIN_DATABASE_URL"] = original_db_url
        if original_cold_path is None:
            os.environ.pop("GIN_COLD_PATH", None)
        else:
            os.environ["GIN_COLD_PATH"] = original_cold_path


def test_trust_weights_default_empty(tmp_path):
    p = tmp_path / "node_a.yaml"
    p.write_text(_YAML, encoding="utf-8")
    cfg = load_node_config(p)
    assert cfg.trust_weights == {}
    assert cfg.trust_gate_threshold == 0.5


def test_trust_weights_parsed_from_yaml(tmp_path):
    p = tmp_path / "node_a.yaml"
    p.write_text(
        _YAML + "trust_weights:\n  node_c:\n    monetary_policy: 0.1\n"
        "trust_gate_threshold: 0.6\n",
        encoding="utf-8",
    )
    cfg = load_node_config(p)
    assert cfg.trust_weights == {"node_c": {"monetary_policy": 0.1}}
    assert cfg.trust_gate_threshold == 0.6
