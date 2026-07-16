# Federation mTLS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace plaintext HTTP + shared-secret bearer auth between federation nodes with mutual TLS over self-signed, pinned peer certificates — no CA, no VPN mesh dependency.

**Architecture:** Each node generates its own self-signed ECDSA cert (`gin/federation/certs.py` + `scripts/node_keygen.py`). Operators exchange certs out-of-band and pin them in `PeerConfig.pinned_cert_path`. The server (uvicorn) requires and verifies client certs against a bundle of pinned peer certs; the client (`HttpPeerClient`) trusts only the specific peer's pinned cert per connection. Rejection happens entirely at the TLS layer — no endpoint handler ever inspects caller identity, so the existing `_check_auth` bearer-token dependency is deleted outright, not replaced.

**Tech Stack:** Python 3.10+, `cryptography` (new dependency, self-signed cert generation), stdlib `ssl`, existing FastAPI/uvicorn/httpx stack (no transport-framework change).

## Global Constraints

- No CA, no Tailscale/VPN dependency — self-signed certs pinned per peer (spec decision 2).
- Pin the full certificate file, not just a fingerprint hash (spec decision 3) — standard TLS libraries validate against a CA bundle, not an arbitrary hash.
- Remove `shared_secret`/`GIN_FED_SECRET` entirely — no dual-mode auth, no back-compat shim (spec decisions 4, 6).
- 10-year cert validity, manual re-pinning for rotation — no automated rotation/revocation/CA in this phase (spec decision 5).
- `cryptography>=41.0` is the one new dependency; everything else (TLS itself) uses the stdlib `ssl` module already exercised transitively by uvicorn/httpx.
- Verified mechanism (see spike results below) — do not deviate from these without re-verifying: self-signed cert needs `BasicConstraints(ca=True)` + `KeyUsage(key_cert_sign=True, ...)` critical extensions to work as its own trust anchor; httpx client **must** disable `check_hostname` (certs carry no IP/DNS SAN, and adding one would conflict with the long-lived/no-rotation decision) or even a correctly pinned peer fails with `CERTIFICATE_VERIFY_FAILED: IP address mismatch`; a TLS rejection surfaces to the caller as `httpx.RemoteProtocolError` (a subclass of `httpx.HTTPError`), not a clean connect-time exception — the existing `except httpx.HTTPError` in `HttpPeerClient` already covers this with no new exception handling needed.
- **Deviation from the spec's data-flow narrative, deliberate and verified:** the spec's design doc describes resolving the caller's `node_id` inside each endpoint handler after mTLS verification. This plan does not do that. Empirically confirmed (installed uvicorn 0.51.0 source, `venv/Lib/site-packages/uvicorn/protocols/http/*.py`) that uvicorn never exposes the verified client certificate into the ASGI scope — subclassing its protocol internals to extract one would be fragile across uvicorn upgrades. Checked against the actual code: no endpoint handler reads caller identity today (`_check_auth` is a pure allow/deny gate; `FederatedQuery.origin_node` already carries caller identity for loop-prevention, unrelated to auth). TLS-layer rejection alone — an unpinned cert never completes the handshake, so it never reaches the ASGI app — satisfies every falsifiable-claim bar in the spec. `_check_auth` is therefore deleted, not replaced with a cert-extraction mechanism.

---

## Task 1: Certificate generation & CA-bundle logic

**Files:**
- Create: `gin/federation/certs.py`
- Test: `tests/test_certs.py`

**Interfaces:**
- Produces: `generate_self_signed_cert(node_id: str, certs_root: str | Path) -> tuple[Path, Path]` (returns `(cert_path, key_path)`, writes `certs_root/node_id/{cert.pem,key.pem}`); `cert_fingerprint(cert_path: str | Path) -> str` (e.g. `"SHA256:ab:cd:..."`); `build_ca_bundle(peer_cert_paths: Sequence[str | Path], bundle_path: str | Path) -> Path | None` (concatenates PEM certs; returns `None` and writes nothing if the list is empty).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_certs.py
"""Self-signed cert generation, fingerprinting, and CA-bundle building."""
import ssl

from cryptography import x509

from gin.federation.certs import build_ca_bundle, cert_fingerprint, generate_self_signed_cert


def test_generate_self_signed_cert_writes_expected_paths(tmp_path):
    cert_path, key_path = generate_self_signed_cert("node_a", tmp_path)
    assert cert_path == tmp_path / "node_a" / "cert.pem"
    assert key_path == tmp_path / "node_a" / "key.pem"
    assert cert_path.exists()
    assert key_path.exists()


def test_generate_self_signed_cert_has_expected_common_name(tmp_path):
    cert_path, _ = generate_self_signed_cert("node_b", tmp_path)
    cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    cn = cert.subject.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)[0].value
    assert cn == "node_b"


def test_cert_fingerprint_is_stable_sha256_format(tmp_path):
    cert_path, _ = generate_self_signed_cert("node_a", tmp_path)
    fp = cert_fingerprint(cert_path)
    assert fp.startswith("SHA256:")
    assert len(fp[len("SHA256:"):].split(":")) == 32  # 32-byte digest


def test_cert_fingerprint_differs_for_different_certs(tmp_path):
    cert_a, _ = generate_self_signed_cert("node_a", tmp_path)
    cert_b, _ = generate_self_signed_cert("node_b", tmp_path)
    assert cert_fingerprint(cert_a) != cert_fingerprint(cert_b)


def test_build_ca_bundle_concatenates_all_peer_certs(tmp_path):
    cert_b, _ = generate_self_signed_cert("node_b", tmp_path)
    cert_c, _ = generate_self_signed_cert("node_c", tmp_path)
    bundle = build_ca_bundle([cert_b, cert_c], tmp_path / "bundle.pem")
    assert bundle.read_text() == cert_b.read_text() + cert_c.read_text()


def test_build_ca_bundle_is_usable_as_a_real_ca_store(tmp_path):
    cert_b, _ = generate_self_signed_cert("node_b", tmp_path)
    cert_c, _ = generate_self_signed_cert("node_c", tmp_path)
    bundle = build_ca_bundle([cert_b, cert_c], tmp_path / "bundle.pem")
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_verify_locations(cafile=str(bundle))  # must not raise


def test_build_ca_bundle_returns_none_for_empty_peer_list(tmp_path):
    assert build_ca_bundle([], tmp_path / "bundle.pem") is None
    assert not (tmp_path / "bundle.pem").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_certs.py -v`
Expected: FAIL / ERROR with `ModuleNotFoundError: No module named 'gin.federation.certs'`

- [ ] **Step 3: Add `cryptography` to requirements.txt**

Edit `requirements.txt`, in the "Federation (Phase 3, v1)" section, add:
```
cryptography>=41.0
```

Run: `./venv/Scripts/pip.exe install cryptography>=41.0` (already installed at 49.0.0 in this environment from the design spike — confirm with `./venv/Scripts/python.exe -c "import cryptography; print(cryptography.__version__)"`)

- [ ] **Step 4: Write the implementation**

```python
# gin/federation/certs.py
"""Self-signed peer certificates for mutual TLS.

Each node is its own certificate authority: it presents a self-signed cert
as its identity and trusts only the specific certificates operators have
pinned for each peer (docs/superpowers/specs/2026-07-16-federation-mtls-design.md).
No CA, no shared secret — the pinned certificate file IS the trust anchor.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Sequence

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

_VALIDITY_DAYS = 3650


def generate_self_signed_cert(node_id: str, certs_root: str | Path) -> tuple[Path, Path]:
    """Write a self-signed ECDSA P-256 cert+key for node_id under
    certs_root/node_id/{cert.pem,key.pem}. Returns (cert_path, key_path).

    BasicConstraints(ca=True) and KeyUsage(key_cert_sign=True) are required
    (and marked critical) for this self-signed cert to work as its own trust
    anchor when pinned directly into a peer's CA bundle — without them,
    OpenSSL's chain validation rejects it even when the exact cert is
    present in the trust store.
    """
    out_dir = Path(certs_root) / node_id
    out_dir.mkdir(parents=True, exist_ok=True)

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, node_id)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=_VALIDITY_DAYS))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, key_cert_sign=True, crl_sign=True,
                key_encipherment=False, content_commitment=False,
                data_encipherment=False, key_agreement=False,
                encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage(
                [ExtendedKeyUsageOID.CLIENT_AUTH, ExtendedKeyUsageOID.SERVER_AUTH]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    cert_path = out_dir / "cert.pem"
    key_path = out_dir / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return cert_path, key_path


def cert_fingerprint(cert_path: str | Path) -> str:
    """SHA-256 fingerprint of the cert at cert_path, for out-of-band pinning
    confirmation (e.g. "SHA256:ab:cd:...")."""
    cert = x509.load_pem_x509_certificate(Path(cert_path).read_bytes())
    digest = cert.fingerprint(hashes.SHA256())
    return "SHA256:" + ":".join(f"{b:02x}" for b in digest)


def build_ca_bundle(
    peer_cert_paths: Sequence[str | Path], bundle_path: str | Path
) -> Path | None:
    """Concatenate pinned peer certs into one CA bundle file for
    ssl_ca_certs. Returns None (writes nothing) if peer_cert_paths is empty
    — an empty bundle file is invalid and crashes
    SSLContext.load_verify_locations with CERTIFICATE_VERIFY_FAILED /
    NO_CERTIFICATE_OR_CRL_FOUND; callers must skip ssl_ca_certs/CERT_REQUIRED
    entirely in that case rather than pass an empty file.
    """
    if not peer_cert_paths:
        return None
    bundle_path = Path(bundle_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    contents = "".join(Path(p).read_text(encoding="utf-8") for p in peer_cert_paths)
    bundle_path.write_text(contents, encoding="utf-8")
    return bundle_path
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_certs.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add requirements.txt gin/federation/certs.py tests/test_certs.py
git commit -m "Self-signed cert generation + CA-bundle logic for federation mTLS (sub-project 5, task 1)."
```

---

## Task 2: `node_keygen.py` CLI

**Files:**
- Create: `scripts/node_keygen.py`
- Test: `tests/test_node_keygen.py`

**Interfaces:**
- Consumes: `gin.federation.certs.generate_self_signed_cert`, `cert_fingerprint` (Task 1).
- Produces: a runnable CLI, `python scripts/node_keygen.py --node-id <id> --out-dir <dir>`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_node_keygen.py
"""node_keygen CLI: writes cert+key and prints the fingerprint."""
import subprocess
import sys

from gin.federation.certs import cert_fingerprint


def test_node_keygen_writes_cert_and_prints_fingerprint(tmp_path):
    out_dir = tmp_path / "certs"
    result = subprocess.run(
        [sys.executable, "scripts/node_keygen.py",
         "--node-id", "node_a", "--out-dir", str(out_dir)],
        capture_output=True, text=True, check=True,
    )

    cert_path = out_dir / "node_a" / "cert.pem"
    key_path = out_dir / "node_a" / "key.pem"
    assert cert_path.exists()
    assert key_path.exists()
    assert cert_fingerprint(cert_path) in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/test_node_keygen.py -v`
Expected: FAIL — `scripts/node_keygen.py` does not exist (non-zero exit from subprocess, `check=True` raises `CalledProcessError`)

- [ ] **Step 3: Write the implementation**

```python
# scripts/node_keygen.py
"""Generate a self-signed identity certificate for one GIN federation node.

Usage:
    python scripts/node_keygen.py --node-id node_a --out-dir certs

Writes certs/<node_id>/{cert.pem,key.pem} and prints the SHA-256 fingerprint
for out-of-band pinning confirmation with peer operators.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gin.federation.certs import cert_fingerprint, generate_self_signed_cert


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-id", required=True, help="this node's node_id")
    parser.add_argument("--out-dir", default="certs", help="output root (default: certs)")
    args = parser.parse_args()

    cert_path, key_path = generate_self_signed_cert(args.node_id, args.out_dir)
    fingerprint = cert_fingerprint(cert_path)

    print(f"[*] wrote {cert_path}")
    print(f"[*] wrote {key_path}")
    print(f"[*] fingerprint: {fingerprint}")
    print("[*] send cert_path to peer operators out-of-band; confirm this "
          "fingerprint matches what they receive before pinning it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/test_node_keygen.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/node_keygen.py tests/test_node_keygen.py
git commit -m "node_keygen CLI for federation mTLS identity generation (sub-project 5, task 2)."
```

---

## Task 3: Config schema swap + server auth removal

This is the pivot task: `NodeConfig.shared_secret` and `_check_auth` are removed together (the dependency reads the field, so they cannot be split across separately-green commits). It touches `config.py`, `server.py`, and every test file that constructs `NodeConfig`/uses bearer auth. Structured as several small commits within the task rather than one — the suite is expected to be red between sub-steps until the last one lands; that's normal here, not a signal to stop.

**Files:**
- Modify: `gin/federation/config.py` (full rewrite below)
- Modify: `gin/federation/server.py` (full rewrite below)
- Modify: `tests/test_federation_config.py`, `tests/test_router_selection.py`, `tests/test_multi_peer_sync.py`, `tests/test_federation_router.py`, `tests/test_trust_gate_wiring.py`, `tests/test_summary_endpoint.py`, `tests/test_anchor_endpoints.py`, `tests/test_federation_server.py`, `tests/test_peer_selection_loop.py`, `tests/test_anchor_sync_loop.py`, `tests/test_federation_loop.py` (construction-site fixes only in this task — Task 5 gives the last three real certs and mTLS wiring)

**Interfaces:**
- Produces: `NodeConfig(..., cert_path: str, key_path: str, ...)` (no `shared_secret`); `PeerConfig(node_id, url, pinned_cert_path: str = "")`.
- Consumes (Task 1, already shipped): nothing yet — real cert files come in Task 4/5.

### Sub-step A: `config.py` + its own test

- [ ] **Step 1: Update the failing test first**

Replace `tests/test_federation_config.py` in full:

```python
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
```

(`test_secret_env_override` is deleted — `GIN_FED_SECRET` no longer exists.)

- [ ] **Step 2: Run to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_config.py -v`
Expected: FAIL — `NodeConfig` has no `cert_path` field yet (`TypeError`)

- [ ] **Step 3: Rewrite `config.py`**

```python
# gin/federation/config.py
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
```

- [ ] **Step 4: Run to verify `test_federation_config.py` passes** (rest of the suite is now red — expected)

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_config.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add gin/federation/config.py tests/test_federation_config.py
git commit -m "Config schema: cert_path/key_path/pinned_cert_path replace shared_secret (sub-project 5, task 3a). Rest of suite red until 3b-3d land."
```

### Sub-step B: `server.py` auth removal + its own test file rewrites

- [ ] **Step 1: Rewrite `server.py`** — remove `_check_auth`, the `Authorization`-checking dependency, and now-unused imports (`hmac`, `Header`, `HTTPException`, `Depends`):

```python
# gin/federation/server.py
"""FastAPI app factory for one federation node.

Peer authentication happens at the TLS layer (mutual TLS, self-signed pinned
certificates — see docs/superpowers/specs/2026-07-16-federation-mtls-design.md
and scripts/node_serve.py's uvicorn.run wiring). A connection that doesn't
present a pinned peer certificate never completes its handshake, so it never
reaches any endpoint here — there is no app-layer auth check to perform.
Guards that DO run here: protocol version (typed refusal) -> hop limit
(typed refusal). hop_count >= 1 requests are answered locally and NEVER
re-delegated — that, plus the router only running at hop 0, is the entire
loop-prevention story. answer_fn / peer_client / corpus_fingerprint are
injected so tests run without a model, a database, or a network.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from contextlib import asynccontextmanager
from typing import Callable, Optional

from fastapi import FastAPI

from gin.corpus.relevance import query_keywords

from .anchor_store import PeerAnchorStore
from .anchor_sync import run_forever
from .anchor_tree import all_bucket_hashes, build_buckets, root_hash
from .client import PeerClient
from .config import NodeConfig, PeerConfig
from .peer_selection import rank_peers
from .peer_summary_store import PeerSummaryStore
from .router import AnswerFn, answer_or_delegate
from .schema import (
    PROTOCOL_VERSION,
    AnchorBucketsResponse,
    AnchorLeaf,
    AnchorLeavesResponse,
    AnchorRootResponse,
    AnchorSyncStats,
    FederatedAnswer,
    FederatedQuery,
    FederatedResponse,
    NodeRefusal,
    PeerSummaryResponse,
)
from .service import claims_to_wire
from .trust_gate import filter_trusted


def create_app(
    config: NodeConfig,
    *,
    answer_fn: AnswerFn,
    peer_client: Optional[PeerClient] = None,
    corpus_fingerprint: Optional[dict] = None,
    local_anchor_rows: Optional[Callable[[], list[AnchorLeaf]]] = None,
    peer_anchor_store: Optional[PeerAnchorStore] = None,
    local_summary: Optional[Callable[[], PeerSummaryResponse]] = None,
    peer_summary_store: Optional[PeerSummaryStore] = None,
    embed_query_fn: Optional[Callable[[str], list[float]]] = None,
) -> FastAPI:
    fingerprint = corpus_fingerprint or {}
    anchor_rows_fn = local_anchor_rows or (lambda: [])
    summary_fn = local_summary or (lambda: PeerSummaryResponse(node_id=config.node_id))

    def _rank_peers_for_query(query: str) -> list[PeerConfig]:
        if (
            peer_summary_store is None
            or embed_query_fn is None
            or len(config.peers) <= 1
        ):
            return list(config.peers)
        summaries = {}
        for p in config.peers:
            s = peer_summary_store.get(p.node_id)
            if s is not None:
                summaries[p.node_id] = s
        if not summaries:
            return list(config.peers)
        order = rank_peers(
            embed_query_fn(query), query_keywords(query),
            summaries, [p.node_id for p in config.peers],
        )
        # Scoped to summaries (synced peers only) — a peer with no cached
        # summary is absent here too, so filter_trusted defaults it to
        # trusted rather than gating on missing information.
        domains_by_peer = {nid: s.domains for nid, s in summaries.items()}
        order = filter_trusted(
            order, domains_by_peer, config.trust_weights, config.trust_gate_threshold
        )
        by_id = {p.node_id: p for p in config.peers}
        return [by_id[nid] for nid in order]

    sync_stats = AnchorSyncStats(
        node_id=config.node_id,
        peer_node_id=config.peers[0].node_id if config.peers else "",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        tasks: list[asyncio.Task] = []
        if peer_anchor_store is not None and peer_client is not None and config.peers:
            for i, peer in enumerate(config.peers):
                stats = sync_stats if i == 0 else AnchorSyncStats(
                    node_id=config.node_id, peer_node_id=peer.node_id
                )
                tasks.append(
                    asyncio.create_task(
                        run_forever(
                            peer, peer_client, peer_anchor_store,
                            config.anchor_sync_interval_s, stats,
                            summary_store=peer_summary_store,
                        )
                    )
                )
        yield
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title=f"GIN federation node {config.node_id}", lifespan=lifespan)

    def _refusal(
        fq: FederatedQuery,
        reason: str,
        detail: str = "",
        peer_reasons: Optional[dict] = None,
    ) -> FederatedResponse:
        return FederatedResponse(
            refusal=NodeRefusal(
                request_id=fq.request_id,
                node_id=config.node_id,
                reason=reason,
                detail=detail,
                peer_reasons=peer_reasons or {},
            )
        )

    @app.post(
        "/v1/federated/query",
        response_model=FederatedResponse,
        response_model_exclude_none=True,
    )
    def federated_query(fq: FederatedQuery) -> FederatedResponse:
        if fq.protocol_version != PROTOCOL_VERSION:
            return _refusal(
                fq, "version_mismatch",
                f"node speaks v{PROTOCOL_VERSION}, got v{fq.protocol_version}",
            )
        if fq.hop_count > 1:
            return _refusal(
                fq, "hop_limit", f"hop_count {fq.hop_count} exceeds max 1"
            )

        started = time.monotonic()

        if fq.hop_count >= 1 or peer_client is None or not config.peers:
            local = answer_fn(fq.query)
            if local.refused:
                return _refusal(fq, local.refusal_reason or "zero_cursors")
            return FederatedResponse(
                answer=FederatedAnswer(
                    request_id=fq.request_id,
                    node_id=config.node_id,
                    answer_text=local.raw_text,
                    claims=claims_to_wire(local),
                    corpus_fingerprint=fingerprint,
                    synthesis_mode=local.synthesis_mode or "unknown",
                    timing_s=time.monotonic() - started,
                )
            )

        routed = answer_or_delegate(
            fq.query,
            config=config,
            answer_fn=answer_fn,
            peer_client=peer_client,
            request_id=fq.request_id,
            peer_ranker=_rank_peers_for_query,
        )
        if routed.refused:
            own = routed.refusal_reasons.get(config.node_id, "zero_cursors")
            peer_reasons = {
                k: v for k, v in routed.refusal_reasons.items()
                if k != config.node_id
            }
            return _refusal(fq, own, peer_reasons=peer_reasons)
        return FederatedResponse(
            answer=FederatedAnswer(
                request_id=fq.request_id,
                node_id=routed.source_node,
                answer_text=routed.answer_text,
                claims=routed.claims,
                corpus_fingerprint=(
                    routed.corpus_fingerprint if routed.federation else fingerprint
                ),
                synthesis_mode=routed.synthesis_mode,
                timing_s=time.monotonic() - started,
            ),
            federation=routed.federation,
        )

    @app.get("/v1/federated/anchors/root", response_model=AnchorRootResponse)
    def anchors_root() -> AnchorRootResponse:
        rows = anchor_rows_fn()
        return AnchorRootResponse(
            node_id=config.node_id,
            root_hash=root_hash(all_bucket_hashes(rows)),
            leaf_count=len(rows),
        )

    @app.get("/v1/federated/anchors/buckets", response_model=AnchorBucketsResponse)
    def anchors_buckets() -> AnchorBucketsResponse:
        rows = anchor_rows_fn()
        return AnchorBucketsResponse(node_id=config.node_id, bucket_hashes=all_bucket_hashes(rows))

    @app.get(
        "/v1/federated/anchors/bucket/{index}", response_model=AnchorLeavesResponse
    )
    def anchors_bucket(index: int) -> AnchorLeavesResponse:
        rows = anchor_rows_fn()
        buckets = build_buckets(rows)
        return AnchorLeavesResponse(
            node_id=config.node_id, bucket_index=index, leaves=buckets.get(index, [])
        )

    @app.get("/v1/federated/anchors/sync_stats", response_model=AnchorSyncStats)
    def anchors_sync_stats() -> AnchorSyncStats:
        return sync_stats

    @app.get("/v1/federated/summary", response_model=PeerSummaryResponse)
    def federated_summary() -> PeerSummaryResponse:
        return summary_fn()

    return app
```

- [ ] **Step 2: Rewrite `tests/test_federation_server.py`** — delete the two bearer-specific tests, drop `AUTH`/headers everywhere, fix `NodeConfig` construction:

```python
# tests/test_federation_server.py
"""Server guards: version, hop limit; local-only for hop>=1."""
from fastapi.testclient import TestClient

from gin.eval.arms import ArmOutput
from gin.eval.claims import RawClaim
from gin.federation.config import NodeConfig, PeerConfig
from gin.federation.schema import FederatedQuery, FederatedResponse
from gin.federation.server import create_app

CFG = NodeConfig(
    node_id="node_b", host="127.0.0.1", port=8472,
    database_url="postgresql://x/gin_node_b", cold_path="data/cold_node_b",
    model_path="", n_gpu_layers=0, n_ctx=4096,
    cert_path="b_cert.pem", key_path="b_key.pem", peer_timeout_s=5.0,
    peers=(PeerConfig(node_id="node_a", url="http://peer-a"),),
)


def _grounded(q: str) -> ArmOutput:
    return ArmOutput(
        raw_text="grounded answer",
        claims=[RawClaim(text="grounded answer", span_type="EXACT",
                         cited_chunk_ids=["n2_doc_002:3"])],
        retrieval_manifest_hash="h",
        synthesis_mode="convergent",
    )


def _refusing(q: str) -> ArmOutput:
    return ArmOutput(
        raw_text="[REFUSAL]", claims=[], retrieval_manifest_hash="",
        refused=True, refusal_reason="zero_cursors",
    )


class ExplodingPeer:
    """Peer client that must never be consulted."""

    def query(self, peer, fq):  # pragma: no cover - failure is the assert
        raise AssertionError("peer consulted on a hop>=1 request")


def _post(client, payload):
    return client.post("/v1/federated/query", json=payload)


def _fq(hop: int) -> dict:
    return FederatedQuery(
        query="q", origin_node="node_a", hop_count=hop
    ).model_dump()


def test_version_mismatch_refused():
    app = create_app(CFG, answer_fn=_grounded, peer_client=ExplodingPeer())
    client = TestClient(app)
    payload = _fq(1)
    payload["protocol_version"] = 99
    r = _post(client, payload)
    assert r.status_code == 200
    resp = FederatedResponse.model_validate(r.json())
    assert resp.refusal.reason == "version_mismatch"


def test_hop_over_limit_refused():
    app = create_app(CFG, answer_fn=_grounded, peer_client=ExplodingPeer())
    client = TestClient(app)
    r = _post(client, _fq(2))
    resp = FederatedResponse.model_validate(r.json())
    assert resp.refusal.reason == "hop_limit"


def test_hop_one_answers_locally_with_fingerprint():
    app = create_app(
        CFG, answer_fn=_grounded, peer_client=ExplodingPeer(),
        corpus_fingerprint={"n_chunks": 46},
    )
    client = TestClient(app)
    r = _post(client, _fq(1))
    resp = FederatedResponse.model_validate(r.json())
    assert resp.answer.node_id == "node_b"
    assert resp.answer.corpus_fingerprint == {"n_chunks": 46}
    assert resp.answer.claims[0].cited_chunk_ids == ["n2_doc_002:3"]
    assert resp.federation is None


def test_hop_one_refusal_never_redelegates():
    app = create_app(CFG, answer_fn=_refusing, peer_client=ExplodingPeer())
    client = TestClient(app)
    r = _post(client, _fq(1))
    resp = FederatedResponse.model_validate(r.json())
    assert resp.refusal.reason == "zero_cursors"
    assert resp.refusal.node_id == "node_b"


def test_hop_zero_local_success_no_federation_layer():
    app = create_app(CFG, answer_fn=_grounded, peer_client=ExplodingPeer())
    client = TestClient(app)
    r = _post(client, _fq(0))
    resp = FederatedResponse.model_validate(r.json())
    assert resp.answer.node_id == "node_b"
    assert resp.federation is None


def test_relayed_answer_keeps_peer_empty_fingerprint():
    from gin.federation.schema import FederatedAnswer, WireClaim

    class FingerprintlessPeer:
        def query(self, peer, fq):
            return FederatedAnswer(
                request_id=fq.request_id, node_id="node_a_peer",
                answer_text="peer answer",
                claims=[WireClaim(text="peer answer", span_type="EXACT",
                                  cited_chunk_ids=["n2_doc_002:3"])],
                corpus_fingerprint={},
                synthesis_mode="convergent",
            )

    app = create_app(
        CFG, answer_fn=_refusing, peer_client=FingerprintlessPeer(),
        corpus_fingerprint={"chunk_count": 999},
    )
    client = TestClient(app)
    r = _post(client, _fq(0))
    resp = FederatedResponse.model_validate(r.json())
    assert resp.answer is not None
    assert resp.federation is not None
    assert resp.answer.corpus_fingerprint == {}
```

- [ ] **Step 3: Rewrite `tests/test_summary_endpoint.py`** — delete `test_summary_endpoint_requires_auth`, drop `AUTH`:

```python
# tests/test_summary_endpoint.py
"""The /v1/federated/summary endpoint: injected summary callable."""
from fastapi.testclient import TestClient

from gin.eval.arms import ArmOutput
from gin.federation.config import NodeConfig, PeerConfig
from gin.federation.schema import PeerSummaryResponse
from gin.federation.server import create_app

CFG = NodeConfig(
    node_id="node_c", host="127.0.0.1", port=8473,
    database_url="postgresql://x/gin_node_c", cold_path="data/cold_node_c",
    model_path="", n_gpu_layers=0, n_ctx=4096,
    cert_path="c_cert.pem", key_path="c_key.pem", peer_timeout_s=5.0,
    peers=(PeerConfig(node_id="node_a", url="http://peer-a"),),
)


def _grounded(q: str) -> ArmOutput:
    return ArmOutput(raw_text="x", claims=[], retrieval_manifest_hash="")


def _summary() -> PeerSummaryResponse:
    return PeerSummaryResponse(
        node_id="node_c", embedding_centroid=[0.1, 0.2, 0.3],
        distinctive_terms={"inflation": 2.0},
    )


def test_summary_endpoint_returns_injected_summary():
    app = create_app(CFG, answer_fn=_grounded, local_summary=_summary)
    client = TestClient(app)
    r = client.get("/v1/federated/summary")
    resp = PeerSummaryResponse.model_validate(r.json())
    assert resp.node_id == "node_c"
    assert resp.distinctive_terms == {"inflation": 2.0}


def test_summary_endpoint_default_is_empty():
    app = create_app(CFG, answer_fn=_grounded)
    client = TestClient(app)
    r = client.get("/v1/federated/summary")
    resp = PeerSummaryResponse.model_validate(r.json())
    assert resp.node_id == "node_c"
    assert resp.embedding_centroid == []
    assert resp.distinctive_terms == {}
```

- [ ] **Step 4: Rewrite `tests/test_anchor_endpoints.py`** — drop `AUTH`/`test_anchors_root_requires_auth`, fix `NodeConfig`:

```python
# tests/test_anchor_endpoints.py
"""Read-only anchor endpoints: root/buckets/bucket/sync_stats, backed by an
injected local_anchor_rows callable."""
from fastapi.testclient import TestClient

from gin.eval.arms import ArmOutput
from gin.federation.anchor_tree import NUM_BUCKETS, all_bucket_hashes, bucket_index, root_hash
from gin.federation.config import NodeConfig, PeerConfig
from gin.federation.schema import (
    AnchorBucketsResponse,
    AnchorLeaf,
    AnchorLeavesResponse,
    AnchorRootResponse,
    AnchorSyncStats,
)
from gin.federation.server import create_app

CFG = NodeConfig(
    node_id="node_a", host="127.0.0.1", port=8471,
    database_url="postgresql://x/gin_node_a", cold_path="data/cold_node_a",
    model_path="", n_gpu_layers=0, n_ctx=4096,
    cert_path="a_cert.pem", key_path="a_key.pem", peer_timeout_s=5.0,
    peers=(PeerConfig(node_id="node_b", url="http://peer-b"),),
)


def _grounded(q: str) -> ArmOutput:
    return ArmOutput(raw_text="x", claims=[], retrieval_manifest_hash="")


def _rows() -> list[AnchorLeaf]:
    return [
        AnchorLeaf(chunk_id="n1_doc_001:0", content_hash="h1", outlet="node_1", title="t1"),
        AnchorLeaf(chunk_id="n1_doc_002:0", content_hash="h2", outlet="node_1", title="t2"),
    ]


def test_anchors_root_matches_pure_computation():
    app = create_app(CFG, answer_fn=_grounded, local_anchor_rows=_rows)
    client = TestClient(app)
    r = client.get("/v1/federated/anchors/root")
    resp = AnchorRootResponse.model_validate(r.json())
    assert resp.root_hash == root_hash(all_bucket_hashes(_rows()))
    assert resp.leaf_count == 2


def test_anchors_buckets_has_16_entries():
    app = create_app(CFG, answer_fn=_grounded, local_anchor_rows=_rows)
    client = TestClient(app)
    r = client.get("/v1/federated/anchors/buckets")
    resp = AnchorBucketsResponse.model_validate(r.json())
    assert len(resp.bucket_hashes) == NUM_BUCKETS


def test_anchors_bucket_returns_only_that_bucket():
    rows = _rows()
    app = create_app(CFG, answer_fn=_grounded, local_anchor_rows=lambda: rows)
    client = TestClient(app)
    idx = bucket_index(rows[0].chunk_id)
    r = client.get(f"/v1/federated/anchors/bucket/{idx}")
    resp = AnchorLeavesResponse.model_validate(r.json())
    assert rows[0].chunk_id in {leaf.chunk_id for leaf in resp.leaves}
    assert resp.bucket_index == idx


def test_anchors_default_empty_when_not_configured():
    app = create_app(CFG, answer_fn=_grounded)
    client = TestClient(app)
    r = client.get("/v1/federated/anchors/root")
    resp = AnchorRootResponse.model_validate(r.json())
    assert resp.leaf_count == 0


def test_sync_stats_defaults_before_any_cycle():
    app = create_app(CFG, answer_fn=_grounded, local_anchor_rows=_rows)
    client = TestClient(app)
    r = client.get("/v1/federated/anchors/sync_stats")
    resp = AnchorSyncStats.model_validate(r.json())
    assert resp.node_id == "node_a"
    assert resp.peer_node_id == "node_b"
    assert resp.cycles_run == 0


import asyncio

from gin.federation.anchor_store import InMemoryPeerAnchorStore
from gin.federation.anchor_tree import all_bucket_hashes, root_hash
from gin.federation.schema import AnchorBucketsResponse


class StubPeerClient:
    """Serves one fixed peer row set over the anchor GET methods only."""

    def __init__(self, rows):
        self.rows = rows

    def get_anchor_root(self, peer):
        return AnchorRootResponse(
            node_id=peer.node_id, root_hash=root_hash(all_bucket_hashes(self.rows)),
            leaf_count=len(self.rows),
        )

    def get_anchor_buckets(self, peer):
        return AnchorBucketsResponse(node_id=peer.node_id, bucket_hashes=all_bucket_hashes(self.rows))

    def get_anchor_bucket(self, peer, index):
        from gin.federation.anchor_tree import bucket_index as _bi
        matches = [r for r in self.rows if _bi(r.chunk_id) == index]
        return AnchorLeavesResponse(node_id=peer.node_id, bucket_index=index, leaves=matches)


def test_lifespan_starts_and_stops_background_sync():
    peer_rows = [AnchorLeaf(chunk_id="p:0", content_hash="h", outlet="o", title="t")]
    store = InMemoryPeerAnchorStore()
    app = create_app(
        CFG, answer_fn=_grounded, peer_client=StubPeerClient(peer_rows),
        peer_anchor_store=store,
    )

    async def _run():
        async with app.router.lifespan_context(app):
            await asyncio.sleep(0.2)

    asyncio.run(_run())
    assert {r.chunk_id for r in store.all_rows("node_b")} == {"p:0"}


def test_no_background_task_without_peer_anchor_store():
    app = create_app(CFG, answer_fn=_grounded, local_anchor_rows=_rows)

    async def _run():
        async with app.router.lifespan_context(app):
            await asyncio.sleep(0.1)

    asyncio.run(_run())
```

- [ ] **Step 5: Rewrite `tests/test_trust_gate_wiring.py`** — drop `SECRET`/`AUTH`, fix `_cfg()`:

```python
# tests/test_trust_gate_wiring.py
"""Trust gate wired into create_app: a peer below the configured trust
threshold for a domain it serves is never contacted, even when it ranks
first on similarity. An unconfigured (empty) trust_weights config must
reproduce sub-project 3's ungated behavior exactly."""
from fastapi.testclient import TestClient

from gin.eval.arms import ArmOutput
from gin.federation.config import NodeConfig, PeerConfig
from gin.federation.peer_summary_store import InMemoryPeerSummaryStore
from gin.federation.schema import (
    FederatedAnswer,
    FederatedQuery,
    FederatedResponse,
    NodeRefusal,
    PeerSummaryResponse,
    WireClaim,
)
from gin.federation.server import create_app


def _cfg(trust_weights=None, trust_gate_threshold=0.5):
    return NodeConfig(
        node_id="node_a", host="127.0.0.1", port=8471,
        database_url="postgresql://x/a", cold_path="data/cold_a",
        model_path="", n_gpu_layers=0, n_ctx=4096,
        cert_path="a_cert.pem", key_path="a_key.pem", peer_timeout_s=5.0,
        peers=(PeerConfig("node_b", "http://b"), PeerConfig("node_c", "http://c")),
        trust_weights=trust_weights or {},
        trust_gate_threshold=trust_gate_threshold,
    )


def _refuse(q):
    return ArmOutput(raw_text="[REFUSAL]", claims=[], retrieval_manifest_hash="",
                     refused=True, refusal_reason="retrieval_floor")


class ScriptedPeer:
    """Answers for `answerer` node_id; refuses for everyone else. Records calls."""

    def __init__(self, answerer):
        self.answerer = answerer
        self.calls = []

    def query(self, peer, fq):
        self.calls.append(peer.node_id)
        if peer.node_id == self.answerer:
            return FederatedAnswer(
                request_id=fq.request_id, node_id=peer.node_id,
                answer_text="grounded", claims=[WireClaim(
                    text="grounded", span_type="EXACT", cited_chunk_ids=["c:0"])],
                corpus_fingerprint={"n": 1}, synthesis_mode="convergent",
            )
        return NodeRefusal(request_id=fq.request_id, node_id=peer.node_id, reason="zero_cursors")


def _summaries():
    store = InMemoryPeerSummaryStore()
    store.set("node_c", PeerSummaryResponse(
        node_id="node_c", embedding_centroid=[1.0, 0.0],
        distinctive_terms={"inflation": 3.0}, domains=["monetary_policy"],
    ))
    store.set("node_b", PeerSummaryResponse(
        node_id="node_b", embedding_centroid=[0.0, 1.0],
        distinctive_terms={"justice": 3.0}, domains=["environmental_impact"],
    ))
    return store


def _embed(q):
    return [1.0, 0.0] if "inflation" in q else [0.0, 1.0]


def test_gated_peer_never_contacted_falls_back_to_refusal():
    peer_client = ScriptedPeer(answerer="node_c")
    app = create_app(
        _cfg(trust_weights={"node_c": {"monetary_policy": 0.1}}),
        answer_fn=_refuse, peer_client=peer_client,
        peer_summary_store=_summaries(), embed_query_fn=_embed,
    )
    client = TestClient(app)
    fq = FederatedQuery(query="what drives inflation", origin_node="d", hop_count=0)
    r = client.post("/v1/federated/query", json=fq.model_dump())
    resp = FederatedResponse.model_validate(r.json())
    assert resp.refusal is not None
    assert peer_client.calls == ["node_b"]
    assert "node_c" not in (resp.refusal.peer_reasons or {})


def test_ungated_query_still_reaches_correct_peer():
    peer_client = ScriptedPeer(answerer="node_c")
    app = create_app(
        _cfg(),
        answer_fn=_refuse, peer_client=peer_client,
        peer_summary_store=_summaries(), embed_query_fn=_embed,
    )
    client = TestClient(app)
    fq = FederatedQuery(query="what drives inflation", origin_node="d", hop_count=0)
    r = client.post("/v1/federated/query", json=fq.model_dump())
    resp = FederatedResponse.model_validate(r.json())
    assert resp.answer.node_id == "node_c"
    assert peer_client.calls == ["node_c"]
```

- [ ] **Step 6: Run the four TestClient-based files**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_server.py tests/test_summary_endpoint.py tests/test_anchor_endpoints.py tests/test_trust_gate_wiring.py -v`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add gin/federation/server.py tests/test_federation_server.py tests/test_summary_endpoint.py tests/test_anchor_endpoints.py tests/test_trust_gate_wiring.py
git commit -m "Remove bearer-token auth from server.py; TLS layer is now the only auth check (sub-project 5, task 3b)."
```

### Sub-step C: Remaining pure-construction fixes (no HTTP/auth involved)

- [ ] **Step 1: Edit `tests/test_router_selection.py`** — replace `_cfg()`:

```python
def _cfg():
    return NodeConfig(
        node_id="node_a", host="127.0.0.1", port=8471,
        database_url="postgresql://x/a", cold_path="data/cold_a",
        model_path="", n_gpu_layers=0, n_ctx=4096,
        cert_path="a_cert.pem", key_path="a_key.pem",
        peer_timeout_s=5.0, peers=(PEER_B, PEER_C),
    )
```

- [ ] **Step 2: Edit `tests/test_federation_router.py`** — replace `CFG`:

```python
CFG = NodeConfig(
    node_id="node_a", host="127.0.0.1", port=8471,
    database_url="postgresql://x/gin_node_a", cold_path="data/cold_node_a",
    model_path="", n_gpu_layers=0, n_ctx=4096,
    cert_path="a_cert.pem", key_path="a_key.pem", peer_timeout_s=5.0, peers=(PEER,),
)
```

(`test_no_peers_refuses_locally`'s `NodeConfig(**{**CFG.__dict__, "peers": ()})` dict-spread construction picks up `cert_path`/`key_path` automatically — no change needed there.)

- [ ] **Step 3: Edit `tests/test_multi_peer_sync.py`** — remove the now-unused `SECRET` constant and fix `_config()`:

```python
def _config(peers: tuple[PeerConfig, ...]) -> NodeConfig:
    return NodeConfig(
        node_id="node_a", host="127.0.0.1", port=8471,
        database_url="postgresql://x/node_a", cold_path="data/cold_node_a",
        model_path="", n_gpu_layers=0, n_ctx=4096,
        cert_path="a_cert.pem", key_path="a_key.pem",
        peer_timeout_s=5.0, peers=peers,
        anchor_sync_interval_s=0.02,
    )
```

Delete the line `SECRET = "multi-peer-secret"` (no longer referenced).

- [ ] **Step 4: Run all three**

Run: `./venv/Scripts/python.exe -m pytest tests/test_router_selection.py tests/test_federation_router.py tests/test_multi_peer_sync.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add tests/test_router_selection.py tests/test_federation_router.py tests/test_multi_peer_sync.py
git commit -m "Fix remaining NodeConfig construction sites for cert_path/key_path (sub-project 5, task 3c)."
```

### Sub-step D: Real-socket test files — construction fix only (mTLS wiring itself is Task 5)

These three still use `shared_secret`/`HttpPeerClient(SECRET, ...)`. This sub-step only makes them *construct* without error using placeholder cert paths; they will not actually pass yet because `HttpPeerClient`'s constructor signature hasn't changed (that's Task 4) and there's no real TLS wiring yet (Task 5). This is expected — Task 5 replaces these files in full anyway.

- [ ] **Step 1: Edit `tests/test_peer_selection_loop.py`** — `_cfg()` signature swap only:

```python
def _cfg(node_id, port, peers):
    return NodeConfig(
        node_id=node_id, host="127.0.0.1", port=port,
        database_url=f"postgresql://x/{node_id}", cold_path=f"data/cold_{node_id}",
        model_path="", n_gpu_layers=0, n_ctx=4096,
        cert_path=f"{node_id}_cert.pem", key_path=f"{node_id}_key.pem",
        peer_timeout_s=10.0, peers=peers,
    )
```

- [ ] **Step 2: Edit `tests/test_anchor_sync_loop.py`** — `_config()` signature swap only:

```python
def _config(node_id: str, port: int, peer: PeerConfig, interval_s: float = 0.05) -> NodeConfig:
    return NodeConfig(
        node_id=node_id, host="127.0.0.1", port=port,
        database_url=f"postgresql://x/{node_id}", cold_path=f"data/cold_{node_id}",
        model_path="", n_gpu_layers=0, n_ctx=4096,
        cert_path=f"{node_id}_cert.pem", key_path=f"{node_id}_key.pem",
        peer_timeout_s=10.0, peers=(peer,),
        anchor_sync_interval_s=interval_s,
    )
```

- [ ] **Step 3: Edit `tests/test_federation_loop.py`** — `_config()` signature swap only:

```python
def _config(node_id: str, port: int, peer: PeerConfig) -> NodeConfig:
    return NodeConfig(
        node_id=node_id, host="127.0.0.1", port=port,
        database_url=f"postgresql://x/{node_id}", cold_path=f"data/cold_{node_id}",
        model_path="", n_gpu_layers=0, n_ctx=4096,
        cert_path=f"{node_id}_cert.pem", key_path=f"{node_id}_key.pem",
        peer_timeout_s=10.0, peers=(peer,),
    )
```

- [ ] **Step 4: Commit** (suite is still red for these three files — expected, resolved in Task 5)

```bash
git add tests/test_peer_selection_loop.py tests/test_anchor_sync_loop.py tests/test_federation_loop.py
git commit -m "NodeConfig construction fix for real-socket tests (sub-project 5, task 3d) — HttpPeerClient/mTLS wiring lands in tasks 4-5."
```

---

## Task 4: `HttpPeerClient` mTLS rewrite

**Files:**
- Modify: `gin/federation/client.py`
- Modify: `tests/test_federation_client.py` (full rewrite)

**Interfaces:**
- Consumes: `gin.federation.certs.generate_self_signed_cert` (Task 1); `PeerConfig.pinned_cert_path` (Task 3).
- Produces: `HttpPeerClient(cert_path: str, key_path: str, timeout_s: float = 300.0, transport: Optional[httpx.BaseTransport] = None)` — same public methods (`query`, `get_anchor_root`, `get_anchor_buckets`, `get_anchor_bucket`, `get_summary`) as before, same `PeerUnreachable`/`PeerClient` Protocol (untouched).

- [ ] **Step 1: Rewrite the failing test file**

```python
# tests/test_federation_client.py
"""HttpPeerClient: parsing, mTLS identity, and failure mapping via MockTransport.

httpx still constructs a real ssl.SSLContext (and loads the cert/key files
from disk) even when a MockTransport intercepts the connection, so these
tests need real — if throwaway — cert fixtures, not arbitrary path strings.
"""
import httpx
import pytest

from gin.federation.certs import generate_self_signed_cert
from gin.federation.client import HttpPeerClient, PeerUnreachable
from gin.federation.config import PeerConfig
from gin.federation.schema import (
    AnchorBucketsResponse,
    AnchorLeaf,
    AnchorLeavesResponse,
    AnchorRootResponse,
    FederatedAnswer,
    FederatedQuery,
    FederatedResponse,
    NodeRefusal,
    PeerSummaryResponse,
)


@pytest.fixture
def own_identity(tmp_path):
    cert_path, key_path = generate_self_signed_cert("node_a", tmp_path)
    return str(cert_path), str(key_path)


@pytest.fixture
def peer(tmp_path):
    cert_path, _ = generate_self_signed_cert("node_b", tmp_path)
    return PeerConfig(node_id="node_b", url="http://peer-b", pinned_cert_path=str(cert_path))


def _fq() -> FederatedQuery:
    return FederatedQuery(query="q", origin_node="node_a", hop_count=1)


def test_returns_parsed_answer_and_sends_no_bearer_header(own_identity, peer):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["url"] = str(request.url)
        body = FederatedResponse(
            answer=FederatedAnswer(
                request_id="r", node_id="node_b", answer_text="grounded",
                claims=[], corpus_fingerprint={"n_chunks": 1},
            )
        )
        return httpx.Response(200, json=body.model_dump())

    client = HttpPeerClient(*own_identity, transport=httpx.MockTransport(handler))
    out = client.query(peer, _fq())
    assert isinstance(out, FederatedAnswer)
    assert out.node_id == "node_b"
    assert seen["auth"] is None  # no bearer header — mTLS is the identity now
    assert seen["url"] == "http://peer-b/v1/federated/query"


def test_returns_parsed_refusal(own_identity, peer):
    def handler(request: httpx.Request) -> httpx.Response:
        body = FederatedResponse(
            refusal=NodeRefusal(
                request_id="r", node_id="node_b", reason="zero_cursors"
            )
        )
        return httpx.Response(200, json=body.model_dump())

    client = HttpPeerClient(*own_identity, transport=httpx.MockTransport(handler))
    out = client.query(peer, _fq())
    assert isinstance(out, NodeRefusal)
    assert out.reason == "zero_cursors"


def test_http_error_maps_to_peer_unreachable(own_identity, peer):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    client = HttpPeerClient(*own_identity, transport=httpx.MockTransport(handler))
    with pytest.raises(PeerUnreachable) as exc:
        client.query(peer, _fq())
    assert exc.value.peer.node_id == "node_b"


def test_connect_error_maps_to_peer_unreachable(own_identity, peer):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = HttpPeerClient(*own_identity, transport=httpx.MockTransport(handler))
    with pytest.raises(PeerUnreachable):
        client.query(peer, _fq())


def test_remote_protocol_error_maps_to_peer_unreachable(own_identity, peer):
    """The real-world shape of a rejected mTLS handshake: httpx surfaces it
    as RemoteProtocolError (subclass of HTTPError), not a connect-time error
    — verified against a real uvicorn+httpx stack during design. The existing
    except httpx.HTTPError must already cover this with no new code."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("Server disconnected without sending a response.")

    client = HttpPeerClient(*own_identity, transport=httpx.MockTransport(handler))
    with pytest.raises(PeerUnreachable):
        client.query(peer, _fq())


def test_get_anchor_root_parses_and_sends_no_bearer(own_identity, peer):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        body = AnchorRootResponse(node_id="node_b", root_hash="abc", leaf_count=50)
        return httpx.Response(200, json=body.model_dump())

    client = HttpPeerClient(*own_identity, transport=httpx.MockTransport(handler))
    out = client.get_anchor_root(peer)
    assert out.root_hash == "abc"
    assert seen["url"] == "http://peer-b/v1/federated/anchors/root"
    assert seen["auth"] is None


def test_get_anchor_buckets_parses(own_identity, peer):
    def handler(request: httpx.Request) -> httpx.Response:
        body = AnchorBucketsResponse(node_id="node_b", bucket_hashes=["h"] * 16)
        return httpx.Response(200, json=body.model_dump())

    client = HttpPeerClient(*own_identity, transport=httpx.MockTransport(handler))
    out = client.get_anchor_buckets(peer)
    assert len(out.bucket_hashes) == 16


def test_get_anchor_bucket_hits_indexed_path(own_identity, peer):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        body = AnchorLeavesResponse(
            node_id="node_b", bucket_index=7,
            leaves=[AnchorLeaf(chunk_id="c", content_hash="h", outlet="o", title="t")],
        )
        return httpx.Response(200, json=body.model_dump())

    client = HttpPeerClient(*own_identity, transport=httpx.MockTransport(handler))
    out = client.get_anchor_bucket(peer, 7)
    assert seen["url"] == "http://peer-b/v1/federated/anchors/bucket/7"
    assert out.leaves[0].chunk_id == "c"


def test_anchor_endpoint_http_error_maps_to_peer_unreachable(own_identity, peer):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    client = HttpPeerClient(*own_identity, transport=httpx.MockTransport(handler))
    with pytest.raises(PeerUnreachable):
        client.get_anchor_root(peer)


def test_get_summary_parses_and_hits_summary_path(own_identity, peer):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        body = PeerSummaryResponse(
            node_id="node_c", embedding_centroid=[0.1, 0.2],
            distinctive_terms={"inflation": 2.0},
        )
        return httpx.Response(200, json=body.model_dump())

    client = HttpPeerClient(*own_identity, transport=httpx.MockTransport(handler))
    out = client.get_summary(peer)
    assert out.node_id == "node_c"
    assert out.distinctive_terms == {"inflation": 2.0}
    assert seen["url"] == "http://peer-b/v1/federated/summary"
    assert seen["auth"] is None


def test_get_summary_http_error_maps_to_peer_unreachable(own_identity, peer):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = HttpPeerClient(*own_identity, transport=httpx.MockTransport(handler))
    with pytest.raises(PeerUnreachable):
        client.get_summary(peer)
```

- [ ] **Step 2: Run to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_client.py -v`
Expected: FAIL — `HttpPeerClient(*own_identity, ...)` passes 2 positional args, current constructor takes 1 (`shared_secret`)

- [ ] **Step 3: Rewrite `client.py`**

```python
# gin/federation/client.py
"""PeerClient: how one node talks to another.

The Protocol is the seam — the router depends on it, tests inject fakes, and
a gRPC/QUIC implementation (the documented institutional target) can replace
HttpPeerClient without touching routing logic. Peer authentication is mutual
TLS: each connection trusts only the specific peer's pinned self-signed
certificate as its CA, and presents this node's own cert as its client
identity — no shared secret, no CA, no hostname check (the pinned cert IS
the identity check; see
docs/superpowers/specs/2026-07-16-federation-mtls-design.md). HTTP failures
of any kind, including TLS handshake/cert-verification rejection (which
httpx surfaces as RemoteProtocolError, itself an httpx.HTTPError), surface
as PeerUnreachable; the caller decides what an unreachable peer means.
"""
from __future__ import annotations

import ssl
from typing import Optional, Protocol, Union, runtime_checkable

import httpx

from .config import PeerConfig
from .schema import (
    AnchorBucketsResponse,
    AnchorLeavesResponse,
    AnchorRootResponse,
    FederatedAnswer,
    FederatedQuery,
    FederatedResponse,
    NodeRefusal,
    PeerSummaryResponse,
)


class PeerUnreachable(Exception):
    def __init__(self, peer: PeerConfig, cause: Exception) -> None:
        self.peer = peer
        self.cause = cause
        super().__init__(
            f"peer {peer.node_id} at {peer.url} unreachable: {cause}"
        )


@runtime_checkable
class PeerClient(Protocol):
    def query(
        self, peer: PeerConfig, fq: FederatedQuery
    ) -> Union[FederatedAnswer, NodeRefusal]: ...
    def get_anchor_root(self, peer: PeerConfig) -> AnchorRootResponse: ...
    def get_anchor_buckets(self, peer: PeerConfig) -> AnchorBucketsResponse: ...
    def get_anchor_bucket(self, peer: PeerConfig, index: int) -> AnchorLeavesResponse: ...
    def get_summary(self, peer: PeerConfig) -> PeerSummaryResponse: ...


class HttpPeerClient:
    """HTTP/JSON implementation of PeerClient, authenticated with mutual TLS.

    ``transport`` is injectable for tests (httpx.MockTransport); production
    uses the default network transport. Each call builds a fresh SSLContext
    trusting only the target peer's pinned certificate.
    """

    def __init__(
        self,
        cert_path: str,
        key_path: str,
        timeout_s: float = 300.0,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self._cert_path = cert_path
        self._key_path = key_path
        self._timeout = timeout_s
        self._transport = transport

    def _ssl_context(self, peer: PeerConfig) -> ssl.SSLContext:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False  # the pinned cert IS the identity check
        ctx.load_verify_locations(cafile=peer.pinned_cert_path)
        ctx.load_cert_chain(certfile=self._cert_path, keyfile=self._key_path)
        return ctx

    def query(
        self, peer: PeerConfig, fq: FederatedQuery
    ) -> Union[FederatedAnswer, NodeRefusal]:
        try:
            with httpx.Client(
                transport=self._transport, timeout=self._timeout,
                verify=self._ssl_context(peer),
            ) as client:
                r = client.post(
                    f"{peer.url}/v1/federated/query",
                    json=fq.model_dump(),
                )
                r.raise_for_status()
        except httpx.HTTPError as exc:
            raise PeerUnreachable(peer, exc) from exc
        resp = FederatedResponse.model_validate(r.json())
        return resp.answer if resp.answer is not None else resp.refusal

    def get_anchor_root(self, peer: PeerConfig) -> AnchorRootResponse:
        return self._get(peer, "/v1/federated/anchors/root", AnchorRootResponse)

    def get_anchor_buckets(self, peer: PeerConfig) -> AnchorBucketsResponse:
        return self._get(peer, "/v1/federated/anchors/buckets", AnchorBucketsResponse)

    def get_anchor_bucket(self, peer: PeerConfig, index: int) -> AnchorLeavesResponse:
        return self._get(peer, f"/v1/federated/anchors/bucket/{index}", AnchorLeavesResponse)

    def get_summary(self, peer: PeerConfig) -> PeerSummaryResponse:
        return self._get(peer, "/v1/federated/summary", PeerSummaryResponse)

    def _get(self, peer: PeerConfig, path: str, model_cls):
        try:
            with httpx.Client(
                transport=self._transport, timeout=self._timeout,
                verify=self._ssl_context(peer),
            ) as client:
                r = client.get(f"{peer.url}{path}")
                r.raise_for_status()
        except httpx.HTTPError as exc:
            raise PeerUnreachable(peer, exc) from exc
        return model_cls.model_validate(r.json())
```

- [ ] **Step 4: Run to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_client.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add gin/federation/client.py tests/test_federation_client.py
git commit -m "HttpPeerClient: mutual TLS with per-peer pinned certs replaces bearer auth (sub-project 5, task 4)."
```

---

## Task 5: Real-socket integration — mTLS wiring + rejection test

**Files:**
- Modify: `tests/test_federation_loop.py`, `tests/test_peer_selection_loop.py`, `tests/test_anchor_sync_loop.py`

**Interfaces:**
- Consumes: `gin.federation.certs.generate_self_signed_cert`, `build_ca_bundle` (Task 1); `HttpPeerClient(cert_path, key_path, ...)` (Task 4).

This is the only tier that can prove a real TLS handshake rejects an unpinned peer — the falsifiable-claim bars from the spec (peer with correctly pinned cert succeeds; peer with unpinned/wrong cert rejected at handshake) live here.

**Shared pattern across all three files:** in each fixture, node A's CA bundle trusts only its own configured peers. Any external "driver" call in these tests that targets node A's server must therefore authenticate as one of node A's already-pinned peers (reusing that peer's generated cert/key as the driver's identity) rather than inventing a separate, unpinned "driver" identity — a driver with an unpinned cert is exactly the rejection case `test_wrong_cert_rejected` covers on purpose.

- [ ] **Step 1: Rewrite `tests/test_federation_loop.py`** in full — generates real certs per test run, wires mTLS into both uvicorn servers, replaces `test_wrong_secret_rejected` with `test_wrong_cert_rejected`:

```python
# tests/test_federation_loop.py
"""End-to-end sovereign delegation over real localhost sockets, mutual TLS.

Two uvicorn servers (node A and node B) with stubbed answer paths — no model,
no database — exercising the full wire: driver -> A (hop 0) -> B (hop 1).
The external "driver" caller authenticates as node_b (reusing its already-
pinned identity) since that's the one cert node A's CA bundle trusts.
"""
import socket
import ssl
import threading
import time

import httpx
import pytest
import uvicorn

from gin.eval.arms import ArmOutput
from gin.eval.claims import RawClaim
from gin.federation.certs import build_ca_bundle, generate_self_signed_cert
from gin.federation.client import HttpPeerClient
from gin.federation.config import NodeConfig, PeerConfig
from gin.federation.schema import FederatedQuery, FederatedResponse
from gin.federation.server import create_app


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _grounded_b(q: str) -> ArmOutput:
    return ArmOutput(
        raw_text="Indigenous-led resistance efforts",
        claims=[RawClaim(text="Indigenous-led resistance efforts",
                         span_type="EXACT", cited_chunk_ids=["n2_doc_001:4"])],
        retrieval_manifest_hash="stub-b",
        synthesis_mode="convergent",
    )


def _grounded_a(q: str) -> ArmOutput:
    return ArmOutput(
        raw_text="2023 anomaly answer",
        claims=[RawClaim(text="2023 anomaly answer", span_type="EXACT",
                         cited_chunk_ids=["n1_doc_002:1"])],
        retrieval_manifest_hash="stub-a",
        synthesis_mode="convergent",
    )


def _refusing(reason: str):
    def fn(q: str) -> ArmOutput:
        return ArmOutput(raw_text="[REFUSAL]", claims=[],
                         retrieval_manifest_hash="", refused=True,
                         refusal_reason=reason)
    return fn


def _config(node_id: str, port: int, peer: PeerConfig, cert_path, key_path) -> NodeConfig:
    return NodeConfig(
        node_id=node_id, host="127.0.0.1", port=port,
        database_url=f"postgresql://x/{node_id}", cold_path=f"data/cold_{node_id}",
        model_path="", n_gpu_layers=0, n_ctx=4096,
        cert_path=str(cert_path), key_path=str(key_path),
        peer_timeout_s=10.0, peers=(peer,),
    )


def _serve(app, port: int, cert_path, key_path, ca_bundle_path) -> uvicorn.Server:
    server = uvicorn.Server(
        uvicorn.Config(
            app, host="127.0.0.1", port=port, log_level="error",
            ssl_certfile=str(cert_path), ssl_keyfile=str(key_path),
            ssl_ca_certs=str(ca_bundle_path), ssl_cert_reqs=ssl.CERT_REQUIRED,
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.05)
    return server


@pytest.fixture
def two_nodes(request, tmp_path):
    """Start node B (grounded) and node A (answer_fn per-test via param).
    Yields (url_a, a_cert, b_cert, b_key): the driver authenticates as
    node_b (the one identity node A's CA bundle trusts) and trusts a_cert
    to validate node A as the server."""
    a_fn, b_fn = request.param
    port_a, port_b = _free_port(), _free_port()
    a_cert, a_key = generate_self_signed_cert("node_a", tmp_path)
    b_cert, b_key = generate_self_signed_cert("node_b", tmp_path)

    cfg_a = _config("node_a", port_a, PeerConfig("node_b", f"http://127.0.0.1:{port_b}", str(b_cert)), a_cert, a_key)
    cfg_b = _config("node_b", port_b, PeerConfig("node_a", f"http://127.0.0.1:{port_a}", str(a_cert)), b_cert, b_key)
    peer_client = HttpPeerClient(str(a_cert), str(a_key), timeout_s=10.0)
    app_a = create_app(cfg_a, answer_fn=a_fn, peer_client=peer_client)
    app_b = create_app(cfg_b, answer_fn=b_fn, peer_client=peer_client)

    a_bundle = build_ca_bundle([b_cert], tmp_path / "a_ca_bundle.pem")
    b_bundle = build_ca_bundle([a_cert], tmp_path / "b_ca_bundle.pem")
    server_a = _serve(app_a, port_a, a_cert, a_key, a_bundle)
    server_b = _serve(app_b, port_b, b_cert, b_key, b_bundle)
    yield f"https://127.0.0.1:{port_a}", a_cert, b_cert, b_key
    server_a.should_exit = True
    server_b.should_exit = True
    time.sleep(0.2)


def _ask(url: str, trust_cert_path, cert_path, key_path, hop: int = 0) -> httpx.Response:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.load_verify_locations(cafile=str(trust_cert_path))
    ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
    fq = FederatedQuery(query="q", origin_node="driver", hop_count=hop)
    with httpx.Client(verify=ctx) as client:
        return client.post(f"{url}/v1/federated/query", json=fq.model_dump(), timeout=15.0)


@pytest.mark.parametrize(
    "two_nodes", [(_refusing("retrieval_floor"), _grounded_b)], indirect=True
)
def test_delegation_crosses_the_wire(two_nodes):
    url_a, a_cert, b_cert, b_key = two_nodes
    r = _ask(url_a, a_cert, b_cert, b_key)
    assert r.status_code == 200
    resp = FederatedResponse.model_validate(r.json())
    assert resp.answer.node_id == "node_b"
    assert resp.answer.claims[0].cited_chunk_ids == ["n2_doc_001:4"]
    assert resp.federation.answered_by == "node_b"
    assert resp.federation.hop_count == 1
    assert resp.refusal is None


@pytest.mark.parametrize(
    "two_nodes", [(_grounded_a, _grounded_b)], indirect=True
)
def test_local_answer_does_not_route(two_nodes):
    url_a, a_cert, b_cert, b_key = two_nodes
    resp = FederatedResponse.model_validate(_ask(url_a, a_cert, b_cert, b_key).json())
    assert resp.answer.node_id == "node_a"
    assert resp.federation is None
    assert resp.refusal is None


@pytest.mark.parametrize(
    "two_nodes",
    [(_refusing("retrieval_floor"), _refusing("zero_cursors"))],
    indirect=True,
)
def test_both_refuse_aggregated_over_wire(two_nodes):
    url_a, a_cert, b_cert, b_key = two_nodes
    resp = FederatedResponse.model_validate(_ask(url_a, a_cert, b_cert, b_key).json())
    assert resp.refusal.node_id == "node_a"
    assert resp.refusal.reason == "retrieval_floor"
    assert resp.refusal.peer_reasons == {"node_b": "zero_cursors"}
    assert resp.answer is None


@pytest.mark.parametrize(
    "two_nodes", [(_refusing("retrieval_floor"), _grounded_b)], indirect=True
)
def test_hop_one_at_a_never_reaches_b(two_nodes):
    """Loop prevention over the real wire: hop-1 into refusing A must refuse,
    not bounce to grounded B."""
    url_a, a_cert, b_cert, b_key = two_nodes
    resp = FederatedResponse.model_validate(_ask(url_a, a_cert, b_cert, b_key, hop=1).json())
    assert resp.refusal is not None
    assert resp.refusal.reason == "retrieval_floor"
    assert resp.answer is None


@pytest.mark.parametrize(
    "two_nodes", [(_grounded_a, _grounded_b)], indirect=True
)
def test_wrong_cert_rejected(two_nodes, tmp_path):
    """The mTLS replacement for the old wrong-secret-401 test: a caller
    presenting a cert node A never pinned never reaches routing at all —
    rejection happens at the TLS layer (httpx.RemoteProtocolError, a
    subclass of httpx.HTTPError), not as an HTTP status code."""
    url_a, a_cert, _b_cert, _b_key = two_nodes
    stranger_cert, stranger_key = generate_self_signed_cert("stranger", tmp_path)
    with pytest.raises(httpx.HTTPError):
        _ask(url_a, a_cert, stranger_cert, stranger_key)
```

- [ ] **Step 2: Run to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_loop.py -v`
Expected: 6 passed (slower than before — real TLS handshakes; budget up to ~30s for the whole file)

- [ ] **Step 3: Commit**

```bash
git add tests/test_federation_loop.py
git commit -m "test_federation_loop: real mTLS wiring, wrong-cert rejection test replaces wrong-secret (sub-project 5, task 5a)."
```

- [ ] **Step 4: Rewrite `tests/test_peer_selection_loop.py`** in full:

```python
# tests/test_peer_selection_loop.py
"""Three uvicorn nodes over real sockets, mutual TLS, no model/DB: node A
ranks B vs. C from injected summaries and delegates to the right one on the
first try. The external "driver" caller authenticates as node_b (one of the
two identities node A's CA bundle trusts) and trusts a_cert to validate
node A as the server."""
import socket
import ssl
import threading
import time

import httpx
import pytest
import uvicorn

from gin.eval.arms import ArmOutput
from gin.eval.claims import RawClaim
from gin.federation.certs import build_ca_bundle, generate_self_signed_cert
from gin.federation.client import HttpPeerClient
from gin.federation.config import NodeConfig, PeerConfig
from gin.federation.peer_summary_store import InMemoryPeerSummaryStore
from gin.federation.schema import FederatedQuery, FederatedResponse, PeerSummaryResponse
from gin.federation.server import create_app


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(app, port, cert_path, key_path, ca_bundle_path):
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="error",
        ssl_certfile=str(cert_path), ssl_keyfile=str(key_path),
        ssl_ca_certs=str(ca_bundle_path), ssl_cert_reqs=ssl.CERT_REQUIRED,
    ))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.05)
    return server


def _refuse(q):
    return ArmOutput(raw_text="[REFUSAL]", claims=[], retrieval_manifest_hash="",
                     refused=True, refusal_reason="retrieval_floor")


def _grounded(node_id):
    def fn(q):
        return ArmOutput(
            raw_text=f"{node_id} answer",
            claims=[RawClaim(text=f"{node_id} answer", span_type="EXACT",
                             cited_chunk_ids=[f"{node_id}:0"])],
            retrieval_manifest_hash="h", synthesis_mode="convergent",
        )
    return fn


def _cfg(node_id, port, peers, cert_path, key_path):
    return NodeConfig(
        node_id=node_id, host="127.0.0.1", port=port,
        database_url=f"postgresql://x/{node_id}", cold_path=f"data/cold_{node_id}",
        model_path="", n_gpu_layers=0, n_ctx=4096,
        cert_path=str(cert_path), key_path=str(key_path),
        peer_timeout_s=10.0, peers=peers,
    )


@pytest.fixture
def three_nodes(tmp_path):
    pa, pb, pc = _free_port(), _free_port(), _free_port()
    a_cert, a_key = generate_self_signed_cert("node_a", tmp_path)
    b_cert, b_key = generate_self_signed_cert("node_b", tmp_path)
    c_cert, c_key = generate_self_signed_cert("node_c", tmp_path)

    peer_client_a = HttpPeerClient(str(a_cert), str(a_key), timeout_s=10.0)
    peer_client_b = HttpPeerClient(str(b_cert), str(b_key), timeout_s=10.0)
    peer_client_c = HttpPeerClient(str(c_cert), str(c_key), timeout_s=10.0)

    cfg_a = _cfg("node_a", pa, (
        PeerConfig("node_b", f"http://127.0.0.1:{pb}", str(b_cert)),
        PeerConfig("node_c", f"http://127.0.0.1:{pc}", str(c_cert)),
    ), a_cert, a_key)
    cfg_b = _cfg("node_b", pb, (PeerConfig("node_a", f"http://127.0.0.1:{pa}", str(a_cert)),), b_cert, b_key)
    cfg_c = _cfg("node_c", pc, (PeerConfig("node_a", f"http://127.0.0.1:{pa}", str(a_cert)),), c_cert, c_key)

    summary_store = InMemoryPeerSummaryStore()
    summary_store.set("node_b", PeerSummaryResponse(
        node_id="node_b", embedding_centroid=[0.0, 1.0], distinctive_terms={"justice": 3.0}))
    summary_store.set("node_c", PeerSummaryResponse(
        node_id="node_c", embedding_centroid=[1.0, 0.0], distinctive_terms={"inflation": 3.0}))

    def embed(q):
        return [1.0, 0.0] if "inflation" in q else [0.0, 1.0]

    app_a = create_app(cfg_a, answer_fn=_refuse, peer_client=peer_client_a,
                       peer_summary_store=summary_store, embed_query_fn=embed)
    app_b = create_app(cfg_b, answer_fn=_grounded("node_b"), peer_client=peer_client_b)
    app_c = create_app(cfg_c, answer_fn=_grounded("node_c"), peer_client=peer_client_c)

    a_bundle = build_ca_bundle([b_cert, c_cert], tmp_path / "a_ca_bundle.pem")
    b_bundle = build_ca_bundle([a_cert], tmp_path / "b_ca_bundle.pem")
    c_bundle = build_ca_bundle([a_cert], tmp_path / "c_ca_bundle.pem")

    sa = _serve(app_a, pa, a_cert, a_key, a_bundle)
    sb = _serve(app_b, pb, b_cert, b_key, b_bundle)
    sc = _serve(app_c, pc, c_cert, c_key, c_bundle)
    yield f"https://127.0.0.1:{pa}", a_cert, b_cert, b_key
    sa.should_exit = sb.should_exit = sc.should_exit = True
    time.sleep(0.2)


def _ask(url, query, trust_cert_path, cert_path, key_path):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.load_verify_locations(cafile=str(trust_cert_path))
    ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
    fq = FederatedQuery(query=query, origin_node="driver", hop_count=0)
    with httpx.Client(verify=ctx) as client:
        r = client.post(f"{url}/v1/federated/query", json=fq.model_dump(), timeout=15.0)
    return FederatedResponse.model_validate(r.json())


def test_selects_c_for_inflation_query_first_try(three_nodes):
    url_a, a_cert, driver_cert, driver_key = three_nodes
    resp = _ask(url_a, "what drives inflation", a_cert, driver_cert, driver_key)
    assert resp.answer.node_id == "node_c"
    assert resp.federation.peers_attempted == ["node_c"]


def test_selects_b_for_justice_query_first_try(three_nodes):
    url_a, a_cert, driver_cert, driver_key = three_nodes
    resp = _ask(url_a, "environmental justice movements", a_cert, driver_cert, driver_key)
    assert resp.answer.node_id == "node_b"
    assert resp.federation.peers_attempted == ["node_b"]
```

- [ ] **Step 5: Run to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/test_peer_selection_loop.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add tests/test_peer_selection_loop.py
git commit -m "test_peer_selection_loop: real mTLS wiring, three real uvicorn nodes (sub-project 5, task 5b)."
```

- [ ] **Step 7: Rewrite `tests/test_anchor_sync_loop.py`** in full:

```python
# tests/test_anchor_sync_loop.py
"""Real-socket anchor sync loop, mutual TLS: two uvicorn nodes, in-memory
stores, no DB, no model — the background task actually runs and converges
the cache."""
import socket
import ssl
import threading
import time

import httpx
import pytest
import uvicorn

from gin.eval.arms import ArmOutput
from gin.federation.anchor_store import InMemoryPeerAnchorStore
from gin.federation.certs import build_ca_bundle, generate_self_signed_cert
from gin.federation.client import HttpPeerClient
from gin.federation.config import NodeConfig, PeerConfig
from gin.federation.schema import AnchorLeaf
from gin.federation.server import create_app


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _leaf(chunk_id: str, content_hash: str = "h") -> AnchorLeaf:
    return AnchorLeaf(chunk_id=chunk_id, content_hash=content_hash, outlet="o", title="t")


def _config(node_id, port, peer, cert_path, key_path, interval_s=0.05):
    return NodeConfig(
        node_id=node_id, host="127.0.0.1", port=port,
        database_url=f"postgresql://x/{node_id}", cold_path=f"data/cold_{node_id}",
        model_path="", n_gpu_layers=0, n_ctx=4096,
        cert_path=str(cert_path), key_path=str(key_path),
        peer_timeout_s=10.0, peers=(peer,),
        anchor_sync_interval_s=interval_s,
    )


def _serve(app, port, cert_path, key_path, ca_bundle_path) -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="error",
        ssl_certfile=str(cert_path), ssl_keyfile=str(key_path),
        ssl_ca_certs=str(ca_bundle_path), ssl_cert_reqs=ssl.CERT_REQUIRED,
    ))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.05)
    return server


def _grounded(q: str) -> ArmOutput:
    return ArmOutput(raw_text="x", claims=[], retrieval_manifest_hash="")


@pytest.fixture
def two_nodes(tmp_path):
    """Yields (store_a, b_rows, url_a, a_cert, b_cert, b_key): the driver in
    test_sync_stats_endpoint_reflects_cycles authenticates as node_b (the
    one identity node A's CA bundle trusts) and trusts a_cert to validate
    node A as the server."""
    port_a, port_b = _free_port(), _free_port()
    a_cert, a_key = generate_self_signed_cert("node_a", tmp_path)
    b_cert, b_key = generate_self_signed_cert("node_b", tmp_path)
    peer_client_a = HttpPeerClient(str(a_cert), str(a_key), timeout_s=10.0)
    peer_client_b = HttpPeerClient(str(b_cert), str(b_key), timeout_s=10.0)

    b_rows = [_leaf(f"doc_{i}:0", content_hash=f"h{i}") for i in range(20)]
    cfg_a = _config("node_a", port_a, PeerConfig("node_b", f"http://127.0.0.1:{port_b}", str(b_cert)), a_cert, a_key)
    cfg_b = _config("node_b", port_b, PeerConfig("node_a", f"http://127.0.0.1:{port_a}", str(a_cert)), b_cert, b_key)
    store_a = InMemoryPeerAnchorStore()
    app_a = create_app(
        cfg_a, answer_fn=_grounded, peer_client=peer_client_a,
        local_anchor_rows=lambda: [], peer_anchor_store=store_a,
    )
    app_b = create_app(
        cfg_b, answer_fn=_grounded, peer_client=peer_client_b,
        local_anchor_rows=lambda: b_rows,
    )
    a_bundle = build_ca_bundle([b_cert], tmp_path / "a_ca_bundle.pem")
    b_bundle = build_ca_bundle([a_cert], tmp_path / "b_ca_bundle.pem")
    server_a = _serve(app_a, port_a, a_cert, a_key, a_bundle)
    server_b = _serve(app_b, port_b, b_cert, b_key, b_bundle)
    yield store_a, b_rows, f"https://127.0.0.1:{port_a}", a_cert, b_cert, b_key
    server_a.should_exit = True
    server_b.should_exit = True
    time.sleep(0.2)


def test_background_loop_converges_cache_to_peer_ground_truth(two_nodes):
    store_a, b_rows, _, _, _, _ = two_nodes
    # HttpPeerClient opens a fresh httpx.Client (fresh TCP connection) per
    # request; on this machine each localhost round trip runs ~300-350ms, so
    # a first sync cycle across up to 16 mismatched buckets can take several
    # seconds. Generous deadline to absorb that without touching production
    # timing.
    deadline = time.monotonic() + 20
    expected = {r.chunk_id for r in b_rows}
    while time.monotonic() < deadline:
        if {r.chunk_id for r in store_a.all_rows("node_b")} == expected:
            break
        time.sleep(0.05)
    assert {r.chunk_id for r in store_a.all_rows("node_b")} == expected


def test_sync_stats_endpoint_reflects_cycles(two_nodes):
    _, _, url_a, a_cert, driver_cert, driver_key = two_nodes
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.load_verify_locations(cafile=str(a_cert))
    ctx.load_cert_chain(certfile=str(driver_cert), keyfile=str(driver_key))

    # Poll rather than a fixed sleep: the first cycle alone can take several
    # seconds on this machine (see note above), so a short fixed sleep is
    # unreliable — poll for the real signal (cycles_run >= 1) instead.
    deadline = time.monotonic() + 20
    cycles_run = 0
    while time.monotonic() < deadline:
        with httpx.Client(verify=ctx) as client:
            r = client.get(f"{url_a}/v1/federated/anchors/sync_stats", timeout=5.0)
        cycles_run = r.json()["cycles_run"]
        if cycles_run >= 1:
            break
        time.sleep(0.1)
    assert cycles_run >= 1
```

- [ ] **Step 8: Run to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/test_anchor_sync_loop.py -v`
Expected: 2 passed

- [ ] **Step 9: Run the full suite to confirm the whole ripple is green**

Run: `./venv/Scripts/python.exe -m pytest tests/ -v`
Expected: all pass, zero references to `shared_secret`/`GIN_FED_SECRET` remain (spot check: `grep -r "shared_secret\|GIN_FED_SECRET" tests/ gin/ scripts/` returns nothing)

- [ ] **Step 10: Commit**

```bash
git add tests/test_anchor_sync_loop.py
git commit -m "test_anchor_sync_loop: real mTLS wiring (sub-project 5, task 5c). Full suite green — shared_secret/GIN_FED_SECRET fully removed."
```

---

## Task 6: `node_serve.py` wiring

**Files:**
- Modify: `scripts/node_serve.py`

**Interfaces:**
- Consumes: `gin.federation.certs.build_ca_bundle` (Task 1); `NodeConfig.cert_path/key_path`, `PeerConfig.pinned_cert_path` (Task 3); `HttpPeerClient(cert_path, key_path, ...)` (Task 4).

No new automated test — this is the production entrypoint, verified live in Task 7's eval run. Manual smoke-check step included below instead.

- [ ] **Step 1: Rewrite `node_serve.py`**

```python
# scripts/node_serve.py
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
        print(f"[*] node {config.node_id}: no peers configured — server "
              f"accepts TLS connections but cannot authenticate any client cert")

    uvicorn.run(app, host=config.host, port=config.port, log_level="info", **ssl_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Manual smoke check** (no model/DB required — confirms the wiring compiles and config loads correctly)

```bash
./venv/Scripts/python.exe -c "
from gin.federation.certs import generate_self_signed_cert, build_ca_bundle
import tempfile, yaml
from pathlib import Path
tmp = Path(tempfile.mkdtemp())
a_cert, a_key = generate_self_signed_cert('node_a', tmp)
cfg = {
    'node_id': 'node_a', 'host': '127.0.0.1', 'port': 18471,
    'database_url': 'postgresql://x/x', 'cold_path': 'x',
    'cert_path': str(a_cert), 'key_path': str(a_key), 'peers': [],
}
(tmp / 'node_a.yaml').write_text(yaml.safe_dump(cfg))
from gin.federation.config import load_node_config
c = load_node_config(tmp / 'node_a.yaml')
print('config loads OK:', c.node_id, c.cert_path)
"
```

Expected: prints `config loads OK: node_a <path>` with no traceback.

- [ ] **Step 3: Commit**

```bash
git add scripts/node_serve.py
git commit -m "node_serve.py: mTLS wiring — CA bundle from pinned peers, HttpPeerClient by cert (sub-project 5, task 6)."
```

---

## Task 7: Live-eval certs + YAML regeneration + .gitignore

**Files:**
- Modify: `config/node_a.yaml`, `config/node_b.yaml`, `config/node_c.yaml`, `config/node_a_trust_gated.yaml`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `scripts/node_keygen.py` (Task 2).

- [ ] **Step 1: Add `certs/` to `.gitignore`**

Edit `.gitignore`, add a line:
```
certs/
```

- [ ] **Step 2: Generate real certs for the three live-eval nodes**

```bash
./venv/Scripts/python.exe scripts/node_keygen.py --node-id node_a --out-dir certs
./venv/Scripts/python.exe scripts/node_keygen.py --node-id node_b --out-dir certs
./venv/Scripts/python.exe scripts/node_keygen.py --node-id node_c --out-dir certs
```

Expected: prints three fingerprints; `certs/node_a/`, `certs/node_b/`, `certs/node_c/` each contain `cert.pem`/`key.pem`.

- [ ] **Step 3: Update `config/node_a.yaml`**

```yaml
# GIN federation node A (institutional corpus: corpus_node1.json).
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
    url: https://127.0.0.1:8472
    pinned_cert_path: certs/node_b/cert.pem
  - node_id: node_c
    url: https://127.0.0.1:8473
    pinned_cert_path: certs/node_c/cert.pem
anchor_sync_interval_s: 10
```

- [ ] **Step 4: Update `config/node_b.yaml`**

```yaml
# GIN federation node B (grassroots corpus: corpus_node2.json).
# If both models on GPU exceed VRAM, set n_gpu_layers: 0 here (CPU decode).
node_id: node_b
host: 127.0.0.1
port: 8472
database_url: postgresql://gin:gin@localhost:5432/gin_node_b
cold_path: data/cold_node_b
model_path: data/models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf
n_gpu_layers: -1
n_ctx: 4096
cert_path: certs/node_b/cert.pem
key_path: certs/node_b/key.pem
peer_timeout_s: 300
peers:
  - node_id: node_a
    url: https://127.0.0.1:8471
    pinned_cert_path: certs/node_a/cert.pem
  - node_id: node_c
    url: https://127.0.0.1:8473
    pinned_cert_path: certs/node_c/cert.pem
anchor_sync_interval_s: 10
```

- [ ] **Step 5: Update `config/node_c.yaml`**

```yaml
# GIN federation node C (monetary-policy corpus: corpus_node3.json).
# Runs CPU-only (n_gpu_layers: 0) so all three 7B models fit alongside
# node A and node B on a single 12GB GPU.
node_id: node_c
host: 127.0.0.1
port: 8473
database_url: postgresql://gin:gin@localhost:5432/gin_node_c
cold_path: data/cold_node_c
model_path: data/models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf
n_gpu_layers: 0
n_ctx: 4096
cert_path: certs/node_c/cert.pem
key_path: certs/node_c/key.pem
peer_timeout_s: 300
anchor_sync_interval_s: 10
peers:
  - node_id: node_a
    url: https://127.0.0.1:8471
    pinned_cert_path: certs/node_a/cert.pem
  - node_id: node_b
    url: https://127.0.0.1:8472
    pinned_cert_path: certs/node_b/cert.pem
```

- [ ] **Step 6: Update `config/node_a_trust_gated.yaml`**

```yaml
# GIN federation node A (institutional corpus: corpus_node1.json).
# This variant additionally trust-gates node_c out of consideration for its
# monetary_policy domain, for the trust-weights live eval's gated run.
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
    url: https://127.0.0.1:8472
    pinned_cert_path: certs/node_b/cert.pem
  - node_id: node_c
    url: https://127.0.0.1:8473
    pinned_cert_path: certs/node_c/cert.pem
anchor_sync_interval_s: 10
trust_weights:
  node_c:
    monetary_policy: 0.1
trust_gate_threshold: 0.5
```

- [ ] **Step 7: Live-eval verification (manual — requires the GPU/model/DB setup this repo already documents; not automatable in this plan)**

Run the existing 3-node eval exactly as sub-project 3/4 did (four terminals: `federation_db_setup.sh` once, then `node_serve.py --config config/node_b.yaml` / `node_c.yaml` / `node_a.yaml` each in its own terminal, then the eval driver):

```bash
./venv/Scripts/python.exe scripts/eval_peer_selection.py
```

Confirm against the spec's falsifiable-claim table:
- precision@1 1.0, avg peers tried 1.0, fabrication 0.0, attribution 1.0, honest refusal 1.0 (reproduces sub-project 3 exactly)

Then re-run with `config/node_a_trust_gated.yaml` in place of `config/node_a.yaml` for node A's terminal:
- `gated_peer_contacted` 0, honest refusal rate 1.0 for `c_only` queries (reproduces sub-project 4 exactly)

If either regresses, the mTLS wiring is wrong, not the eval — stop and debug before proceeding to Task 8.

- [ ] **Step 8: Commit**

```bash
git add config/node_a.yaml config/node_b.yaml config/node_c.yaml config/node_a_trust_gated.yaml .gitignore
git commit -m "Live-eval configs: https URLs, cert_path/key_path, pinned peer certs (sub-project 5, task 7). certs/ generated locally, gitignored — regenerate with node_keygen.py."
```

---

## Task 8: Documentation updates

**Files:**
- Modify: `architecture.md`
- Modify: `README.md`
- Modify: `docs/GIN_Node_Architecture_v1.md`

- [ ] **Step 1: Find and update the `architecture.md` Phase 3 checklist**

Locate the existing combined line (per spec research: `architecture.md:420` `🔲 gRPC/QUIC wire, PKI/mTLS`, and the diagram edge at `architecture.md:343`). Split it:
```
✅ mTLS: self-signed pinned peer certificates — measured on the live 3-node deployment (sub-project 5).
🔲 gRPC/QUIC wire
```

- [ ] **Step 2: Add a "peer authentication" subsection to `README.md`**

Alongside the existing peer-selection/trust-weights subsections, document:
- `python scripts/node_keygen.py --node-id <id> --out-dir certs` generates identity
- Exchange `certs/<id>/cert.pem` with peer operators out-of-band; confirm the printed fingerprint
- Add the peer's cert path to `peers[].pinned_cert_path` in your node's YAML
- Example config snippet (reuse the `config/node_a.yaml` shape from Task 7)

- [ ] **Step 3: Add a v1 implementation note to `docs/GIN_Node_Architecture_v1.md`**

Matching the existing peer-selection/trust-weights notes' pattern: mTLS is self-signed-and-pinned (no CA), certs are long-lived (10 years) with manual re-pinning for rotation, `shared_secret`/`GIN_FED_SECRET` no longer exist, and CA-based issuance / automated rotation / revocation remain future work (per the spec's explicit out-of-scope list).

- [ ] **Step 4: Commit**

```bash
git add architecture.md README.md docs/GIN_Node_Architecture_v1.md
git commit -m "Docs: federation mTLS shipped and measured (sub-project 5, task 8)."
```

---

## Self-Review Notes

**Spec coverage:** Every falsifiable-claim bar has a corresponding test (Task 5's real-socket tests for pinned-success/unpinned-rejection; Task 7's live-eval step for the two regression bars). Every architecture-table row from the spec maps to a task.

**Deviation from spec narrative (flagged, not hidden):** the spec's Data-flow step 6 described resolving the caller's `node_id` inside the endpoint handler after mTLS verification. This plan does not do that. Verified empirically during planning (installed uvicorn 0.51.0 source inspection) that uvicorn never exposes the verified client certificate into the ASGI scope, and checked against the actual code that no endpoint handler reads caller identity today (`_check_auth` was a pure allow/deny gate; `FederatedQuery.origin_node` already carries caller identity for loop-prevention, unrelated to auth). TLS-layer rejection alone — an unpinned cert never completes the handshake, so it never reaches the ASGI app — satisfies every falsifiable-claim bar. `_check_auth` is deleted, not replaced with a cert-extraction mechanism.

**Type/interface consistency:** `HttpPeerClient(cert_path, key_path, timeout_s=300.0, transport=None)` used identically across Task 4's tests, Task 5's real-socket tests, and Task 6's `node_serve.py`. `PeerConfig.pinned_cert_path` and `NodeConfig.cert_path/key_path` used identically everywhere they appear. `build_ca_bundle` returning `None` for an empty peer list is handled by its one real caller (`node_serve.py`, Task 6). The "driver authenticates as an already-pinned peer" pattern is applied consistently across all three Task 5 files.

**Placeholder scan:** No TBD/TODO; every step shows complete, runnable code.
