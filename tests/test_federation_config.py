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
shared_secret: dev-federation-secret
peer_timeout_s: 300
peers:
  - node_id: node_b
    url: http://127.0.0.1:8472/
"""


def test_load_node_config(tmp_path):
    p = tmp_path / "node_a.yaml"
    p.write_text(_YAML, encoding="utf-8")
    cfg = load_node_config(p)
    assert cfg.node_id == "node_a"
    assert cfg.port == 8471
    assert cfg.peers == (PeerConfig(node_id="node_b", url="http://127.0.0.1:8472"),)
    assert cfg.peer_timeout_s == 300.0
    assert cfg.chat_template == "mistral"  # default


def test_secret_env_override(tmp_path, monkeypatch):
    p = tmp_path / "node_a.yaml"
    p.write_text(_YAML, encoding="utf-8")
    monkeypatch.setenv("GIN_FED_SECRET", "real-secret")
    cfg = load_node_config(p)
    assert cfg.shared_secret == "real-secret"


def test_apply_env(tmp_path, monkeypatch):
    p = tmp_path / "node_a.yaml"
    p.write_text(_YAML, encoding="utf-8")
    monkeypatch.delenv("GIN_DATABASE_URL", raising=False)
    monkeypatch.delenv("GIN_COLD_PATH", raising=False)
    cfg = load_node_config(p)
    apply_env(cfg)
    assert os.environ["GIN_DATABASE_URL"].endswith("/gin_node_a")
    assert os.environ["GIN_COLD_PATH"] == "data/cold_node_a"
