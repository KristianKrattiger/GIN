# Federation v1 — Sovereign Delegation Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two sovereign GIN node processes; when Node A cannot ground a query it delegates over HTTP to Node B and relays B's answer with attribution intact and explicitly marked as B's — measured against the spec's falsifiable bar.

**Architecture:** New `gin/federation/` package: Pydantic wire schema (the protocol contract), YAML node config, an injectable `PeerClient` (HTTP behind a Protocol), a router that delegates only on pre-commitment grounding failures, and a FastAPI app factory. The local answer path reuses `NoContinuationArm` unchanged except two new `ArmOutput` fields. Loop prevention is structural (hop count). A live eval driver measures the bar against two running node processes.

**Tech Stack:** Python 3.12 (repo venv), FastAPI + uvicorn + httpx, Pydantic v2, existing Postgres/pgvector corpus tier, llama-cpp-python for live decode, pytest.

**Spec:** `docs/superpowers/specs/2026-07-13-federation-v1-sovereign-delegation-design.md` — read it before starting.

## Global Constraints

- Python is ALWAYS the repo venv: `./venv/Scripts/python.exe` (Windows). Run pytest as `./venv/Scripts/python.exe -m pytest`.
- `PROTOCOL_VERSION = 1`. Refusal reason enum values exactly: `retrieval_floor`, `zero_cursors`, `hop_limit`, `version_mismatch`.
- Max hop count is 1. A node NEVER re-delegates an incoming federated request (`hop_count >= 1`).
- HTTP semantics: `401` bad bearer, `422` malformed body (FastAPI default); every other outcome is `200` with `FederatedResponse` — a refusal is an epistemic outcome, not a transport error.
- Per-node isolation: node A = DB `gin_node_a` + `data/cold_node_a`; node B = `gin_node_b` + `data/cold_node_b`. The corpus tier reads `GIN_DATABASE_URL` / `GIN_COLD_PATH` from env at call time; `apply_env()` must run before any DB touch.
- Never commit `.env`. The dev shared secret `dev-federation-secret` in `config/*.yaml` is intentionally non-secret; `GIN_FED_SECRET` env var overrides it.
- Commit messages: plain English, why-first, ending with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Model file for live runs: `data/models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf` (exists; gitignored).

---

### Task 1: Dependencies + wire schema

**Files:**
- Modify: `requirements.txt`
- Create: `gin/federation/__init__.py`
- Create: `gin/federation/schema.py`
- Test: `tests/test_federation_schema.py`

**Interfaces:**
- Consumes: nothing (leaf task).
- Produces: `PROTOCOL_VERSION: int`, `new_request_id() -> str`, and Pydantic models `WireClaim(text, span_type, cited_chunk_ids)`, `FederationLayer(answered_by, hop_count, transport, peer_url, request_id)`, `FederatedQuery(protocol_version, request_id, query, origin_node, hop_count)`, `FederatedAnswer(protocol_version, request_id, node_id, answer_text, claims, corpus_fingerprint, synthesis_mode, timing_s)`, `NodeRefusal(protocol_version, request_id, node_id, reason, detail, peer_reasons)`, `FederatedResponse(answer, refusal, federation)` with an exactly-one-of-answer/refusal validator. Every later task imports from `gin.federation.schema`.

- [ ] **Step 1: Add dependencies and install**

Append to `requirements.txt` after the `# Corpus tier` block:

```
# Federation (Phase 3, v1)
fastapi>=0.110
uvicorn>=0.29
httpx>=0.27
pydantic>=2.5
```

Run: `./venv/Scripts/pip install -r requirements.txt`
Expected: fastapi, uvicorn install cleanly (httpx already present).

- [ ] **Step 2: Write the failing test**

Create `tests/test_federation_schema.py`:

```python
"""Wire schema round-trips and envelope invariants."""
import pytest
from pydantic import ValidationError

from gin.federation.schema import (
    PROTOCOL_VERSION,
    FederatedAnswer,
    FederatedQuery,
    FederatedResponse,
    FederationLayer,
    NodeRefusal,
    WireClaim,
    new_request_id,
)


def test_query_round_trip():
    fq = FederatedQuery(query="q", origin_node="node_a", hop_count=1)
    assert fq.protocol_version == PROTOCOL_VERSION
    assert fq.request_id  # auto-generated
    again = FederatedQuery.model_validate(fq.model_dump())
    assert again == fq


def test_answer_round_trip_with_claims():
    ans = FederatedAnswer(
        request_id=new_request_id(),
        node_id="node_b",
        answer_text="Indigenous-led resistance efforts",
        claims=[
            WireClaim(
                text="Indigenous-led resistance efforts",
                span_type="EXACT",
                cited_chunk_ids=["n2_doc_001:4"],
            )
        ],
        corpus_fingerprint={"n_chunks": 46},
        synthesis_mode="convergent",
        timing_s=1.5,
    )
    again = FederatedAnswer.model_validate(ans.model_dump())
    assert again.claims[0].cited_chunk_ids == ["n2_doc_001:4"]


def test_refusal_reason_enum_enforced():
    with pytest.raises(ValidationError):
        NodeRefusal(
            request_id="r", node_id="node_a", reason="not_a_reason"
        )


def test_refusal_carries_peer_reasons():
    ref = NodeRefusal(
        request_id="r",
        node_id="node_a",
        reason="retrieval_floor",
        peer_reasons={"node_b": "zero_cursors"},
    )
    assert ref.peer_reasons["node_b"] == "zero_cursors"


def test_response_exactly_one_of_answer_refusal():
    ref = NodeRefusal(request_id="r", node_id="node_a", reason="hop_limit")
    ok = FederatedResponse(refusal=ref)
    assert ok.answer is None
    with pytest.raises(ValidationError):
        FederatedResponse()  # neither
    ans = FederatedAnswer(
        request_id="r", node_id="node_b", answer_text="x",
        claims=[], corpus_fingerprint={},
    )
    with pytest.raises(ValidationError):
        FederatedResponse(answer=ans, refusal=ref)  # both


def test_federation_layer_defaults():
    layer = FederationLayer(answered_by="node_b", hop_count=1, request_id="r")
    assert layer.transport == "http"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.federation'`

- [ ] **Step 4: Write the implementation**

Create `gin/federation/__init__.py`:

```python
"""Federation tier — Phase 3 v1: sovereign delegation between nodes.

Spec: docs/superpowers/specs/2026-07-13-federation-v1-sovereign-delegation-design.md
"""
```

Create `gin/federation/schema.py`:

```python
"""Wire protocol for Federation v1.

These Pydantic models ARE the protocol contract; HTTP is incidental transport
behind the PeerClient seam (client.py). Version every change through
PROTOCOL_VERSION — a node that receives a different version refuses with
``version_mismatch`` rather than best-effort parsing.
"""
from __future__ import annotations

from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

PROTOCOL_VERSION = 1

# A node's own failure reason. peer_reasons values are free-form strings
# (they include transport outcomes like "unreachable").
RefusalReason = Literal[
    "retrieval_floor", "zero_cursors", "hop_limit", "version_mismatch"
]


def new_request_id() -> str:
    return str(uuid4())


class WireClaim(BaseModel):
    """One extracted claim, mirroring gin.eval.claims.RawClaim on the wire."""

    text: str
    span_type: str
    cited_chunk_ids: list[str] = Field(default_factory=list)


class FederationLayer(BaseModel):
    """Provenance extension: how a delegated answer reached the caller."""

    answered_by: str
    hop_count: int
    transport: str = "http"
    peer_url: str = ""
    request_id: str


class FederatedQuery(BaseModel):
    protocol_version: int = PROTOCOL_VERSION
    request_id: str = Field(default_factory=new_request_id)
    query: str
    origin_node: str
    hop_count: int = 0


class FederatedAnswer(BaseModel):
    protocol_version: int = PROTOCOL_VERSION
    request_id: str
    node_id: str
    answer_text: str
    claims: list[WireClaim] = Field(default_factory=list)
    corpus_fingerprint: dict = Field(default_factory=dict)
    synthesis_mode: str = "unknown"
    timing_s: float = 0.0


class NodeRefusal(BaseModel):
    protocol_version: int = PROTOCOL_VERSION
    request_id: str
    node_id: str
    reason: RefusalReason
    detail: str = ""
    # On an aggregated (hop-0) refusal: what each consulted peer said.
    peer_reasons: dict[str, str] = Field(default_factory=dict)


class FederatedResponse(BaseModel):
    """Endpoint envelope: exactly one of answer/refusal, plus optional
    federation provenance (present only on hop-0 delegated answers)."""

    answer: Optional[FederatedAnswer] = None
    refusal: Optional[NodeRefusal] = None
    federation: Optional[FederationLayer] = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "FederatedResponse":
        if (self.answer is None) == (self.refusal is None):
            raise ValueError("exactly one of answer/refusal must be set")
        return self
```

- [ ] **Step 5: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_schema.py -v`
Expected: 6 PASS

- [ ] **Step 6: Commit**

```bash
git add requirements.txt gin/federation/__init__.py gin/federation/schema.py tests/test_federation_schema.py
git commit -m "Open the federation tier: versioned wire schema is the protocol contract.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Node configuration

**Files:**
- Create: `gin/federation/config.py`
- Create: `config/node_a.yaml`
- Create: `config/node_b.yaml`
- Test: `tests/test_federation_config.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `PeerConfig(node_id: str, url: str)` (frozen dataclass), `NodeConfig(node_id, host, port, database_url, cold_path, model_path, n_gpu_layers, n_ctx, shared_secret, peer_timeout_s, peers: tuple[PeerConfig, ...], chat_template)` (frozen dataclass), `load_node_config(path) -> NodeConfig`, `apply_env(config) -> None`. Tasks 6, 7, 9 consume `NodeConfig`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_federation_config.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_config.py -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError` on `gin.federation.config`

- [ ] **Step 3: Write the implementation**

Create `gin/federation/config.py`:

```python
"""Per-node configuration.

Each node process is sovereign: its own Postgres database, cold store, model,
port, and peer list. The corpus tier reads GIN_DATABASE_URL / GIN_COLD_PATH
from the environment at call time (gin/corpus/db.py), so ``apply_env`` must be
called before the process touches the database.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PeerConfig:
    node_id: str
    url: str  # base URL, no trailing slash


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
    shared_secret: str
    peer_timeout_s: float
    peers: tuple[PeerConfig, ...]
    chat_template: str = "mistral"


def load_node_config(path: str | Path) -> NodeConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    peers = tuple(
        PeerConfig(node_id=p["node_id"], url=str(p["url"]).rstrip("/"))
        for p in raw.get("peers", [])
    )
    # Committed config files carry a non-secret dev default; a real secret is
    # injected via env so it never lands in git.
    secret = os.environ.get("GIN_FED_SECRET") or raw["shared_secret"]
    return NodeConfig(
        node_id=raw["node_id"],
        host=raw.get("host", "127.0.0.1"),
        port=int(raw["port"]),
        database_url=raw["database_url"],
        cold_path=raw["cold_path"],
        model_path=raw.get("model_path", ""),
        n_gpu_layers=int(raw.get("n_gpu_layers", -1)),
        n_ctx=int(raw.get("n_ctx", 4096)),
        shared_secret=secret,
        peer_timeout_s=float(raw.get("peer_timeout_s", 300.0)),
        peers=peers,
        chat_template=raw.get("chat_template", "mistral"),
    )


def apply_env(config: NodeConfig) -> None:
    """Point the corpus tier at this node's database and cold store.

    Must run before the first DB connection; db.py reads env at call time.
    """
    os.environ["GIN_DATABASE_URL"] = config.database_url
    os.environ["GIN_COLD_PATH"] = config.cold_path
```

Create `config/node_a.yaml`:

```yaml
# GIN federation node A (institutional corpus: corpus_node1.json).
# shared_secret here is a NON-secret dev default for localhost;
# set GIN_FED_SECRET in the environment to override off-localhost.
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
    url: http://127.0.0.1:8472
```

Create `config/node_b.yaml`:

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
shared_secret: dev-federation-secret
peer_timeout_s: 300
peers:
  - node_id: node_a
    url: http://127.0.0.1:8471
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_config.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add gin/federation/config.py config/node_a.yaml config/node_b.yaml tests/test_federation_config.py
git commit -m "Per-node federation config: sovereign DB/cold/model/peer settings per process.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Grounding-failure reasons on ArmOutput

**Files:**
- Modify: `gin/eval/arms.py` (dataclass `ArmOutput` ~line 51; `_refusal_output` ~line 119; `NoContinuationArm.run` ~line 218)
- Test: `tests/test_federation_arm_reasons.py`

**Interfaces:**
- Consumes: existing `ArmOutput`, `_refusal_output`, `NoContinuationArm` in `gin/eval/arms.py`.
- Produces: `ArmOutput.refusal_reason: str` (`""` when not refused; `"retrieval_floor"` or `"zero_cursors"` when refused) and `ArmOutput.synthesis_mode: str` (bundle mode on success, `""` on refusal). Tasks 4, 6, 7 consume these fields.

- [ ] **Step 1: Write the failing test**

Create `tests/test_federation_arm_reasons.py`:

```python
"""ArmOutput carries the structured grounding-failure signal federation routes on."""
from gin.eval.arms import ArmOutput, _refusal_output


def test_refusal_output_default_reason():
    out = _refusal_output()
    assert out.refused is True
    assert out.refusal_reason == "zero_cursors"
    assert out.synthesis_mode == ""


def test_refusal_output_explicit_reason():
    out = _refusal_output(reason="retrieval_floor")
    assert out.refusal_reason == "retrieval_floor"


def test_success_output_defaults():
    out = ArmOutput(raw_text="x", claims=[], retrieval_manifest_hash="h")
    assert out.refusal_reason == ""
    assert out.synthesis_mode == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_arm_reasons.py -v`
Expected: FAIL — `AttributeError` / `TypeError` (no `refusal_reason` field, no `reason` kwarg)

- [ ] **Step 3: Implement the changes in `gin/eval/arms.py`**

Add two fields to `ArmOutput` (after the existing `chunks` field):

```python
    # Structured grounding-failure signal: "" when not refused, else
    # "retrieval_floor" (pre-decode confidence floor) or "zero_cursors"
    # (decode produced no grounded, query-relevant claims). Federation v1
    # delegates on exactly these pre-commitment reasons.
    refusal_reason: str = ""
    # Bundle mode on success ("convergent"/"divergent"), "" on refusal.
    synthesis_mode: str = ""
```

Change `_refusal_output` to accept a reason (keyword-only, defaulted so the
RAG/Flagged arms are untouched):

```python
def _refusal_output(
    manifest_hash: str = "",
    bundle: Optional[SynthesisBundle] = None,
    *,
    reason: str = "zero_cursors",
) -> ArmOutput:
    return ArmOutput(
        raw_text=REFUSAL_SENTINEL,
        claims=[],
        retrieval_manifest_hash=manifest_hash,
        refused=True,
        node_of=_node_map(bundle) if bundle else {},
        chunks=_chunk_texts(bundle) if bundle else [],
        refusal_reason=reason,
    )
```

In `NoContinuationArm.run`, tag the reasons:
- the `except RetrievalConfidenceError:` branch becomes
  `return _refusal_output(reason="retrieval_floor")`
- the three post-decode refusal branches (`not _retrieval_relevant(...)`,
  `not _claims_query_relevant(...)`, gold-coverage) keep the default
  `reason="zero_cursors"` (no change needed beyond the new default).
- the final success `return ArmOutput(...)` gains
  `synthesis_mode=result.bundle.mode,`

- [ ] **Step 4: Run new test AND the full suite (regression gate)**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_arm_reasons.py tests/ -x -q`
Expected: new tests PASS; the whole existing suite stays green (the fields are additive with defaults).

- [ ] **Step 5: Commit**

```bash
git add gin/eval/arms.py tests/test_federation_arm_reasons.py
git commit -m "Tag arm refusals with their grounding-failure reason; federation routes on it.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Local-answer service seam + wire conversion

**Files:**
- Create: `gin/federation/service.py`
- Test: `tests/test_federation_service.py`

**Interfaces:**
- Consumes: `NoContinuationArm`, `ArmConfig`, `ArmOutput` from `gin.eval.arms`; `RawClaim` from `gin.eval.claims`; `WireClaim` from Task 1.
- Produces: `answer_query(query: str, llm: Any, arm_config: Optional[ArmConfig] = None) -> ArmOutput` and `claims_to_wire(output: ArmOutput) -> list[WireClaim]`. Tasks 6, 7, 9 consume both.

- [ ] **Step 1: Write the failing test**

Create `tests/test_federation_service.py`:

```python
"""The service seam converts arm output to wire claims losslessly."""
from gin.eval.arms import ArmOutput
from gin.eval.claims import RawClaim
from gin.federation.service import claims_to_wire


def test_claims_to_wire_preserves_fields():
    out = ArmOutput(
        raw_text="a | b",
        claims=[
            RawClaim(text="a", span_type="EXACT", cited_chunk_ids=["n2_doc_001:4"]),
            RawClaim(text="b", span_type="AMBIGUOUS", cited_chunk_ids=["x:0", "y:1"]),
        ],
        retrieval_manifest_hash="h",
    )
    wire = claims_to_wire(out)
    assert [w.text for w in wire] == ["a", "b"]
    assert wire[0].span_type == "EXACT"
    assert wire[1].cited_chunk_ids == ["x:0", "y:1"]


def test_claims_to_wire_empty():
    out = ArmOutput(raw_text="", claims=[], retrieval_manifest_hash="")
    assert claims_to_wire(out) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_service.py -v`
Expected: FAIL — no module `gin.federation.service`

- [ ] **Step 3: Write the implementation**

Create `gin/federation/service.py`:

```python
"""Local answer seam for federation.

One function both the node server and the router call. It wraps
NoContinuationArm — the measured Phase 1/2 answer path — unchanged: retrieval
floor, materialize, constrained decode, claim extraction, and the refusal
semantics federation delegates on. The spec's ``answer_query`` service
function is this thin wrapper; the CLI keeps its existing direct path.
"""
from __future__ import annotations

from typing import Any, Optional

from gin.eval.arms import ArmConfig, ArmOutput, NoContinuationArm

from .schema import WireClaim


def answer_query(
    query: str, llm: Any, arm_config: Optional[ArmConfig] = None
) -> ArmOutput:
    """Run this node's full local answer path for one query."""
    return NoContinuationArm(arm_config).run(query, llm)


def claims_to_wire(output: ArmOutput) -> list[WireClaim]:
    """Serialize extracted claims for the wire, field-for-field."""
    return [
        WireClaim(
            text=c.text,
            span_type=c.span_type,
            cited_chunk_ids=list(c.cited_chunk_ids),
        )
        for c in output.claims
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_service.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add gin/federation/service.py tests/test_federation_service.py
git commit -m "Service seam: federation calls the measured NoContinuation path through one function.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: PeerClient — HTTP behind a Protocol

**Files:**
- Create: `gin/federation/client.py`
- Test: `tests/test_federation_client.py`

**Interfaces:**
- Consumes: `PeerConfig` (Task 2); `FederatedQuery`, `FederatedAnswer`, `NodeRefusal`, `FederatedResponse` (Task 1).
- Produces: `PeerUnreachable(Exception)` with `.peer` attribute; `PeerClient` Protocol with `query(peer: PeerConfig, fq: FederatedQuery) -> FederatedAnswer | NodeRefusal`; `HttpPeerClient(shared_secret: str, timeout_s: float = 300.0, transport: Optional[httpx.BaseTransport] = None)`. Tasks 6, 7, 9 consume; the `transport` kwarg is the test seam (httpx.MockTransport) and the future gRPC/QUIC replacement point.

- [ ] **Step 1: Write the failing test**

Create `tests/test_federation_client.py`:

```python
"""HttpPeerClient: parsing, auth header, and failure mapping via MockTransport."""
import httpx
import pytest

from gin.federation.client import HttpPeerClient, PeerUnreachable
from gin.federation.config import PeerConfig
from gin.federation.schema import (
    FederatedAnswer,
    FederatedQuery,
    FederatedResponse,
    NodeRefusal,
)

PEER = PeerConfig(node_id="node_b", url="http://peer-b")


def _fq() -> FederatedQuery:
    return FederatedQuery(query="q", origin_node="node_a", hop_count=1)


def test_returns_parsed_answer_and_sends_bearer():
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

    client = HttpPeerClient("s3cret", transport=httpx.MockTransport(handler))
    out = client.query(PEER, _fq())
    assert isinstance(out, FederatedAnswer)
    assert out.node_id == "node_b"
    assert seen["auth"] == "Bearer s3cret"
    assert seen["url"] == "http://peer-b/v1/federated/query"


def test_returns_parsed_refusal():
    def handler(request: httpx.Request) -> httpx.Response:
        body = FederatedResponse(
            refusal=NodeRefusal(
                request_id="r", node_id="node_b", reason="zero_cursors"
            )
        )
        return httpx.Response(200, json=body.model_dump())

    client = HttpPeerClient("s", transport=httpx.MockTransport(handler))
    out = client.query(PEER, _fq())
    assert isinstance(out, NodeRefusal)
    assert out.reason == "zero_cursors"


def test_http_error_maps_to_peer_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    client = HttpPeerClient("s", transport=httpx.MockTransport(handler))
    with pytest.raises(PeerUnreachable) as exc:
        client.query(PEER, _fq())
    assert exc.value.peer.node_id == "node_b"


def test_connect_error_maps_to_peer_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = HttpPeerClient("s", transport=httpx.MockTransport(handler))
    with pytest.raises(PeerUnreachable):
        client.query(PEER, _fq())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_client.py -v`
Expected: FAIL — no module `gin.federation.client`

- [ ] **Step 3: Write the implementation**

Create `gin/federation/client.py`:

```python
"""PeerClient: how one node talks to another.

The Protocol is the seam — the router depends on it, tests inject fakes, and
a gRPC/QUIC implementation (the documented institutional target) can replace
HttpPeerClient without touching routing logic. HTTP failures of any kind
surface as PeerUnreachable; the caller decides what an unreachable peer means.
"""
from __future__ import annotations

from typing import Optional, Protocol, Union, runtime_checkable

import httpx

from .config import PeerConfig
from .schema import FederatedAnswer, FederatedQuery, FederatedResponse, NodeRefusal


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


class HttpPeerClient:
    """HTTP/JSON implementation of PeerClient.

    ``transport`` is injectable for tests (httpx.MockTransport); production
    uses the default network transport.
    """

    def __init__(
        self,
        shared_secret: str,
        timeout_s: float = 300.0,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self._headers = {"Authorization": f"Bearer {shared_secret}"}
        self._timeout = timeout_s
        self._transport = transport

    def query(
        self, peer: PeerConfig, fq: FederatedQuery
    ) -> Union[FederatedAnswer, NodeRefusal]:
        try:
            with httpx.Client(
                transport=self._transport, timeout=self._timeout
            ) as client:
                r = client.post(
                    f"{peer.url}/v1/federated/query",
                    headers=self._headers,
                    json=fq.model_dump(),
                )
                r.raise_for_status()
        except httpx.HTTPError as exc:
            raise PeerUnreachable(peer, exc) from exc
        resp = FederatedResponse.model_validate(r.json())
        return resp.answer if resp.answer is not None else resp.refusal
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_client.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add gin/federation/client.py tests/test_federation_client.py
git commit -m "PeerClient protocol + HTTP implementation; the wire is a swappable seam.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Router — delegate only on pre-commitment failure

**Files:**
- Create: `gin/federation/router.py`
- Test: `tests/test_federation_router.py`

**Interfaces:**
- Consumes: `NodeConfig`, `PeerConfig` (Task 2); `PeerClient`, `PeerUnreachable` (Task 5); `ArmOutput` with `refusal_reason`/`synthesis_mode` (Task 3); `claims_to_wire` (Task 4); schema models (Task 1).
- Produces: `AnswerFn = Callable[[str], ArmOutput]`; `RoutedResult(refused, source_node, answer_text, claims: list[WireClaim], synthesis_mode, corpus_fingerprint: dict, federation: Optional[FederationLayer], refusal_reasons: dict[str, str], request_id)` dataclass; `answer_or_delegate(query, *, config: NodeConfig, answer_fn: AnswerFn, peer_client: PeerClient, request_id: Optional[str] = None) -> RoutedResult`. Task 7 consumes.

- [ ] **Step 1: Write the failing test**

Create `tests/test_federation_router.py`:

```python
"""Router: delegates exactly when the local path refuses, and only then."""
from gin.eval.arms import ArmOutput
from gin.eval.claims import RawClaim
from gin.federation.client import PeerUnreachable
from gin.federation.config import NodeConfig, PeerConfig
from gin.federation.router import answer_or_delegate
from gin.federation.schema import FederatedAnswer, NodeRefusal, WireClaim

PEER = PeerConfig(node_id="node_b", url="http://peer-b")
CFG = NodeConfig(
    node_id="node_a", host="127.0.0.1", port=8471,
    database_url="postgresql://x/gin_node_a", cold_path="data/cold_node_a",
    model_path="", n_gpu_layers=0, n_ctx=4096,
    shared_secret="s", peer_timeout_s=5.0, peers=(PEER,),
)


def _grounded() -> ArmOutput:
    return ArmOutput(
        raw_text="local answer",
        claims=[RawClaim(text="local answer", span_type="EXACT",
                         cited_chunk_ids=["n1_doc_002:1"])],
        retrieval_manifest_hash="h",
        synthesis_mode="convergent",
    )


def _refusing(reason: str) -> ArmOutput:
    return ArmOutput(
        raw_text="[REFUSAL]", claims=[], retrieval_manifest_hash="",
        refused=True, refusal_reason=reason,
    )


class SpyPeer:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def query(self, peer, fq):
        self.calls.append(fq)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def test_local_success_never_calls_peer():
    peer = SpyPeer(None)
    out = answer_or_delegate(
        "q", config=CFG, answer_fn=lambda q: _grounded(), peer_client=peer
    )
    assert out.refused is False
    assert out.source_node == "node_a"
    assert out.federation is None
    assert peer.calls == []
    assert out.claims[0].cited_chunk_ids == ["n1_doc_002:1"]


def test_local_refusal_delegates_with_hop_one():
    answer = FederatedAnswer(
        request_id="r", node_id="node_b", answer_text="peer answer",
        claims=[WireClaim(text="peer answer", span_type="EXACT",
                          cited_chunk_ids=["n2_doc_002:3"])],
        corpus_fingerprint={"n_chunks": 46}, synthesis_mode="convergent",
    )
    peer = SpyPeer(answer)
    out = answer_or_delegate(
        "q", config=CFG, answer_fn=lambda q: _refusing("retrieval_floor"),
        peer_client=peer,
    )
    assert out.refused is False
    assert out.source_node == "node_b"
    assert out.federation is not None
    assert out.federation.answered_by == "node_b"
    assert out.federation.hop_count == 1
    assert out.corpus_fingerprint == {"n_chunks": 46}
    assert len(peer.calls) == 1
    assert peer.calls[0].hop_count == 1
    assert peer.calls[0].origin_node == "node_a"


def test_both_refuse_aggregates_reasons():
    refusal = NodeRefusal(request_id="r", node_id="node_b", reason="zero_cursors")
    peer = SpyPeer(refusal)
    out = answer_or_delegate(
        "q", config=CFG, answer_fn=lambda q: _refusing("retrieval_floor"),
        peer_client=peer,
    )
    assert out.refused is True
    assert out.refusal_reasons == {
        "node_a": "retrieval_floor", "node_b": "zero_cursors"
    }


def test_peer_unreachable_is_honest_refusal():
    peer = SpyPeer(PeerUnreachable(PEER, ConnectionError("down")))
    out = answer_or_delegate(
        "q", config=CFG, answer_fn=lambda q: _refusing("zero_cursors"),
        peer_client=peer,
    )
    assert out.refused is True
    assert out.refusal_reasons == {
        "node_a": "zero_cursors", "node_b": "unreachable"
    }


def test_no_peers_refuses_locally():
    cfg = NodeConfig(**{**CFG.__dict__, "peers": ()})
    out = answer_or_delegate(
        "q", config=cfg, answer_fn=lambda q: _refusing("retrieval_floor"),
        peer_client=SpyPeer(None),
    )
    assert out.refused is True
    assert out.refusal_reasons == {"node_a": "retrieval_floor"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_router.py -v`
Expected: FAIL — no module `gin.federation.router`

- [ ] **Step 3: Write the implementation**

Create `gin/federation/router.py`:

```python
"""Delegation logic: local answer first; on pre-commitment grounding failure,
ask the configured peer.

Loop prevention is structural: this router runs only for hop-0
(caller-facing) requests. Incoming federated requests (hop_count >= 1) are
answered locally by the server and never re-enter the router, so a request
can cross at most one node boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from gin.eval.arms import ArmOutput

from .client import PeerClient, PeerUnreachable
from .config import NodeConfig
from .schema import (
    FederatedQuery,
    FederationLayer,
    NodeRefusal,
    WireClaim,
    new_request_id,
)
from .service import claims_to_wire

AnswerFn = Callable[[str], ArmOutput]


@dataclass
class RoutedResult:
    """Outcome of answer-or-delegate, ready for provenance assembly."""

    refused: bool
    source_node: str
    answer_text: str = ""
    claims: list[WireClaim] = field(default_factory=list)
    synthesis_mode: str = "unknown"
    # The answering node's fingerprint; empty when answered locally (the
    # server fills its own fingerprint in that case).
    corpus_fingerprint: dict = field(default_factory=dict)
    federation: Optional[FederationLayer] = None
    refusal_reasons: dict[str, str] = field(default_factory=dict)
    request_id: str = ""


def answer_or_delegate(
    query: str,
    *,
    config: NodeConfig,
    answer_fn: AnswerFn,
    peer_client: PeerClient,
    request_id: Optional[str] = None,
) -> RoutedResult:
    rid = request_id or new_request_id()
    local = answer_fn(query)
    if not local.refused:
        return RoutedResult(
            refused=False,
            source_node=config.node_id,
            answer_text=local.raw_text,
            claims=claims_to_wire(local),
            synthesis_mode=local.synthesis_mode or "unknown",
            request_id=rid,
        )

    reasons = {config.node_id: local.refusal_reason or "zero_cursors"}
    if not config.peers:
        return RoutedResult(
            refused=True, source_node=config.node_id,
            refusal_reasons=reasons, request_id=rid,
        )

    peer = config.peers[0]
    fq = FederatedQuery(
        request_id=rid, query=query, origin_node=config.node_id, hop_count=1
    )
    try:
        outcome = peer_client.query(peer, fq)
    except PeerUnreachable:
        reasons[peer.node_id] = "unreachable"
        return RoutedResult(
            refused=True, source_node=config.node_id,
            refusal_reasons=reasons, request_id=rid,
        )
    if isinstance(outcome, NodeRefusal):
        reasons[outcome.node_id] = outcome.reason
        return RoutedResult(
            refused=True, source_node=config.node_id,
            refusal_reasons=reasons, request_id=rid,
        )
    return RoutedResult(
        refused=False,
        source_node=outcome.node_id,
        answer_text=outcome.answer_text,
        claims=list(outcome.claims),
        synthesis_mode=outcome.synthesis_mode,
        corpus_fingerprint=outcome.corpus_fingerprint,
        federation=FederationLayer(
            answered_by=outcome.node_id,
            hop_count=1,
            transport="http",
            peer_url=peer.url,
            request_id=rid,
        ),
        request_id=rid,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_router.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add gin/federation/router.py tests/test_federation_router.py
git commit -m "Router: delegate on pre-commitment grounding failure only; refusals aggregate honestly.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Node server — FastAPI app factory

**Files:**
- Create: `gin/federation/server.py`
- Test: `tests/test_federation_server.py`

**Interfaces:**
- Consumes: everything from Tasks 1–6.
- Produces: `create_app(config: NodeConfig, *, answer_fn: AnswerFn, peer_client: Optional[PeerClient] = None, corpus_fingerprint: Optional[dict] = None) -> FastAPI` exposing `POST /v1/federated/query`. Tasks 8 and 9 consume.

- [ ] **Step 1: Write the failing test**

Create `tests/test_federation_server.py`:

```python
"""Server guards: auth, version, hop limit; local-only for hop>=1."""
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
    shared_secret="s3cret", peer_timeout_s=5.0,
    peers=(PeerConfig(node_id="node_a", url="http://peer-a"),),
)
AUTH = {"Authorization": "Bearer s3cret"}


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


def _post(client, payload, headers=AUTH):
    return client.post("/v1/federated/query", json=payload, headers=headers)


def _fq(hop: int) -> dict:
    return FederatedQuery(
        query="q", origin_node="node_a", hop_count=hop
    ).model_dump()


def test_missing_bearer_is_401():
    app = create_app(CFG, answer_fn=_grounded, peer_client=ExplodingPeer())
    client = TestClient(app)
    r = client.post("/v1/federated/query", json=_fq(1))
    assert r.status_code == 401


def test_wrong_bearer_is_401():
    app = create_app(CFG, answer_fn=_grounded, peer_client=ExplodingPeer())
    client = TestClient(app)
    r = _post(client, _fq(1), headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


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
    # ExplodingPeer proves the peer is not consulted on local success.
    resp = FederatedResponse.model_validate(r.json())
    assert resp.answer.node_id == "node_b"
    assert resp.federation is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_server.py -v`
Expected: FAIL — no module `gin.federation.server`

- [ ] **Step 3: Write the implementation**

Create `gin/federation/server.py`:

```python
"""FastAPI app factory for one federation node.

Guards run in order: bearer auth (401) -> protocol version (typed refusal) ->
hop limit (typed refusal). hop_count >= 1 requests are answered locally and
NEVER re-delegated — that, plus the router only running at hop 0, is the
entire loop-prevention story. answer_fn / peer_client / corpus_fingerprint
are injected so tests run without a model, a database, or a network.
"""
from __future__ import annotations

import time
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException

from .client import PeerClient
from .config import NodeConfig
from .router import AnswerFn, answer_or_delegate
from .schema import (
    PROTOCOL_VERSION,
    FederatedAnswer,
    FederatedQuery,
    FederatedResponse,
    NodeRefusal,
)
from .service import claims_to_wire


def create_app(
    config: NodeConfig,
    *,
    answer_fn: AnswerFn,
    peer_client: Optional[PeerClient] = None,
    corpus_fingerprint: Optional[dict] = None,
) -> FastAPI:
    app = FastAPI(title=f"GIN federation node {config.node_id}")
    fingerprint = corpus_fingerprint or {}

    def _check_auth(authorization: str = Header(default="")) -> None:
        if authorization != f"Bearer {config.shared_secret}":
            raise HTTPException(status_code=401, detail="bad or missing bearer token")

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
    def federated_query(
        fq: FederatedQuery, _: None = Depends(_check_auth)
    ) -> FederatedResponse:
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
            # Incoming federated request (or no peer configured): local only.
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

        # hop 0: caller-facing — may delegate.
        routed = answer_or_delegate(
            fq.query,
            config=config,
            answer_fn=answer_fn,
            peer_client=peer_client,
            request_id=fq.request_id,
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
                corpus_fingerprint=routed.corpus_fingerprint or fingerprint,
                synthesis_mode=routed.synthesis_mode,
                timing_s=time.monotonic() - started,
            ),
            federation=routed.federation,
        )

    return app
```

Note: `_refusal(fq, local.refusal_reason or "zero_cursors")` — `refusal_reason`
is one of the schema's `RefusalReason` values by construction (Task 3 maps
exactly `retrieval_floor`/`zero_cursors`); Pydantic will reject anything else,
which is the correct failure mode for an unexpected reason.

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_server.py -v`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add gin/federation/server.py tests/test_federation_server.py
git commit -m "Node server: auth/version/hop guards; hop>=1 requests never re-delegate.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Real-socket loop test (two uvicorn nodes, no model, no DB)

**Files:**
- Test: `tests/test_federation_loop.py`

**Interfaces:**
- Consumes: `create_app` (Task 7), `HttpPeerClient` (Task 5), `NodeConfig`/`PeerConfig` (Task 2), schema (Task 1), stub `ArmOutput`s.
- Produces: nothing new — this is the CI-safe end-to-end proof of the spec's "real network boundary": two uvicorn servers on localhost ephemeral ports, real HTTP both driver→A and A→B.

- [ ] **Step 1: Write the test (it should pass immediately if Tasks 1–7 are correct — that is the point of an integration gate)**

Create `tests/test_federation_loop.py`:

```python
"""End-to-end sovereign delegation over real localhost sockets.

Two uvicorn servers (node A and node B) with stubbed answer paths — no model,
no database — exercising the full wire: driver -> A (hop 0) -> B (hop 1).
"""
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from gin.eval.arms import ArmOutput
from gin.eval.claims import RawClaim
from gin.federation.client import HttpPeerClient
from gin.federation.config import NodeConfig, PeerConfig
from gin.federation.schema import FederatedQuery, FederatedResponse
from gin.federation.server import create_app

SECRET = "loop-test-secret"


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


def _config(node_id: str, port: int, peer: PeerConfig) -> NodeConfig:
    return NodeConfig(
        node_id=node_id, host="127.0.0.1", port=port,
        database_url=f"postgresql://x/{node_id}", cold_path=f"data/cold_{node_id}",
        model_path="", n_gpu_layers=0, n_ctx=4096,
        shared_secret=SECRET, peer_timeout_s=10.0, peers=(peer,),
    )


def _serve(app, port: int) -> uvicorn.Server:
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
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
def two_nodes(request):
    """Start node B (grounded) and node A (answer_fn per-test via param)."""
    a_fn, b_fn = request.param
    port_a, port_b = _free_port(), _free_port()
    cfg_a = _config("node_a", port_a, PeerConfig("node_b", f"http://127.0.0.1:{port_b}"))
    cfg_b = _config("node_b", port_b, PeerConfig("node_a", f"http://127.0.0.1:{port_a}"))
    peer_client = HttpPeerClient(SECRET, timeout_s=10.0)
    app_a = create_app(cfg_a, answer_fn=a_fn, peer_client=peer_client)
    app_b = create_app(cfg_b, answer_fn=b_fn, peer_client=peer_client)
    server_a = _serve(app_a, port_a)
    server_b = _serve(app_b, port_b)
    yield f"http://127.0.0.1:{port_a}", f"http://127.0.0.1:{port_b}"
    server_a.should_exit = True
    server_b.should_exit = True
    time.sleep(0.2)


def _ask(url: str, hop: int = 0, secret: str = SECRET) -> httpx.Response:
    fq = FederatedQuery(query="q", origin_node="driver", hop_count=hop)
    return httpx.post(
        f"{url}/v1/federated/query",
        headers={"Authorization": f"Bearer {secret}"},
        json=fq.model_dump(),
        timeout=15.0,
    )


@pytest.mark.parametrize(
    "two_nodes", [(_refusing("retrieval_floor"), _grounded_b)], indirect=True
)
def test_delegation_crosses_the_wire(two_nodes):
    url_a, _ = two_nodes
    r = _ask(url_a)
    assert r.status_code == 200
    resp = FederatedResponse.model_validate(r.json())
    assert resp.answer.node_id == "node_b"
    assert resp.answer.claims[0].cited_chunk_ids == ["n2_doc_001:4"]
    assert resp.federation.answered_by == "node_b"
    assert resp.federation.hop_count == 1


@pytest.mark.parametrize(
    "two_nodes", [(_grounded_a, _grounded_b)], indirect=True
)
def test_local_answer_does_not_route(two_nodes):
    url_a, _ = two_nodes
    resp = FederatedResponse.model_validate(_ask(url_a).json())
    assert resp.answer.node_id == "node_a"
    assert resp.federation is None


@pytest.mark.parametrize(
    "two_nodes",
    [(_refusing("retrieval_floor"), _refusing("zero_cursors"))],
    indirect=True,
)
def test_both_refuse_aggregated_over_wire(two_nodes):
    url_a, _ = two_nodes
    resp = FederatedResponse.model_validate(_ask(url_a).json())
    assert resp.refusal.node_id == "node_a"
    assert resp.refusal.reason == "retrieval_floor"
    assert resp.refusal.peer_reasons == {"node_b": "zero_cursors"}


@pytest.mark.parametrize(
    "two_nodes", [(_refusing("retrieval_floor"), _grounded_b)], indirect=True
)
def test_hop_one_at_a_never_reaches_b(two_nodes):
    """Loop prevention over the real wire: hop-1 into refusing A must refuse,
    not bounce to grounded B."""
    url_a, _ = two_nodes
    resp = FederatedResponse.model_validate(_ask(url_a, hop=1).json())
    assert resp.refusal is not None
    assert resp.refusal.reason == "retrieval_floor"


@pytest.mark.parametrize(
    "two_nodes", [(_grounded_a, _grounded_b)], indirect=True
)
def test_wrong_secret_rejected(two_nodes):
    url_a, _ = two_nodes
    r = _ask(url_a, secret="wrong")
    assert r.status_code == 401
```

- [ ] **Step 2: Run the loop test**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_loop.py -v`
Expected: 5 PASS. If a test hangs >30 s, a server failed to start — check that ports are free and `server.started` is being polled.

- [ ] **Step 3: Run the full suite**

Run: `./venv/Scripts/python.exe -m pytest tests/ -q`
Expected: everything green.

- [ ] **Step 4: Commit**

```bash
git add tests/test_federation_loop.py
git commit -m "Prove the delegation loop over real localhost sockets, model-free.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Node entrypoint + per-node database setup

**Files:**
- Create: `scripts/node_serve.py`
- Create: `scripts/federation_db_setup.sh`
- Test: smoke only (`--help`), covered in steps

**Interfaces:**
- Consumes: `load_node_config`/`apply_env` (Task 2), `create_app` (Task 7), `HttpPeerClient` (Task 5), `answer_query` (Task 4), `corpus_fingerprint` from `gin.corpus.fingerprint`, `ArmConfig` from `gin.eval.arms`, `Llama` from `llama_cpp`.
- Produces: `python scripts/node_serve.py --config config/node_a.yaml` runs one node; `bash scripts/federation_db_setup.sh` creates + populates both per-node databases.

- [ ] **Step 1: Write `scripts/node_serve.py`**

```python
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
```

- [ ] **Step 2: Smoke-test the entrypoint**

Run: `./venv/Scripts/python.exe scripts/node_serve.py --help`
Expected: usage text, exit 0 (no model/DB touched).

- [ ] **Step 3: Write `scripts/federation_db_setup.sh`**

```bash
#!/usr/bin/env bash
# Create and populate the two per-node federation databases.
# Prereqs: gin-postgres container running; venv installed.
# Idempotent-ish: CREATE DATABASE fails harmlessly if it already exists.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=./venv/Scripts/python.exe

echo "[1/3] creating databases"
docker exec gin-postgres psql -U gin -d gin -c "CREATE DATABASE gin_node_a OWNER gin;" || true
docker exec gin-postgres psql -U gin -d gin -c "CREATE DATABASE gin_node_b OWNER gin;" || true

echo "[2/3] applying schema"
docker exec -i gin-postgres psql -U gin -d gin_node_a < docker/init-db.sql
docker exec -i gin-postgres psql -U gin -d gin_node_b < docker/init-db.sql

echo "[3/3] ingesting split corpora (node A <- corpus_node1.json, node B <- corpus_node2.json)"
GIN_DATABASE_URL="postgresql://gin:gin@localhost:5432/gin_node_a" \
GIN_COLD_PATH="data/cold_node_a" \
  "$PY" scripts/corpus_ingest.py --source corpus_node1.json --no-edges

GIN_DATABASE_URL="postgresql://gin:gin@localhost:5432/gin_node_b" \
GIN_COLD_PATH="data/cold_node_b" \
  "$PY" scripts/corpus_ingest.py --source corpus_node2.json --no-edges

echo "done. verify:"
docker exec gin-postgres psql -U gin -d gin_node_a -c "SELECT COUNT(*) AS node_a_chunks FROM chunks;"
docker exec gin-postgres psql -U gin -d gin_node_b -c "SELECT COUNT(*) AS node_b_chunks FROM chunks;"
```

Note: `--no-edges` is correct for both — `data/corpus_edges.yaml` contains
cross-node edges (`n1_* ↔ n2_*`) that cannot exist when the corpora live in
separate databases. Cross-node divergent synthesis is explicitly out of scope
(spec, Out of scope #6).

- [ ] **Step 4: Run the setup and verify both databases**

Run: `bash scripts/federation_db_setup.sh`
Expected: both CREATE DATABASE succeed (or already-exists), schema applies, both ingests report chunk counts, final SELECTs show nonzero chunk counts in each DB (node A ~50s, node B ~50 — exact counts depend on the corpora).

- [ ] **Step 5: Commit**

```bash
git add scripts/node_serve.py scripts/federation_db_setup.sh
git commit -m "Node entrypoint + split-corpus DB setup: the two corpora finally live apart.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: Federation eval module + queryset

**Files:**
- Create: `gin/federation/eval.py`
- Create: `data/eval/queryset_federation.yaml`
- Test: `tests/test_federation_eval.py`

**Interfaces:**
- Consumes: `WireClaim` (Task 1).
- Produces: `FederationQuery(id, query, federation_class, gold_chunk_ids)`, `load_federation_queryset(path) -> list[FederationQuery]`, `claims_verify(claims, fetch_text: Callable[[str], Optional[str]]) -> bool`, `verify_claims_in_db(claims, database_url) -> bool`, `QueryOutcome(id, federation_class, refused, routed, source_node, attribution_verified, refusal_reasons)`, `compute_metrics(outcomes) -> dict` with keys `n_queries, routing_false_positives, routing_recall, routed_answer_attribution_verified, routed_fabrication_rate, honest_refusal_rate, a_answered_locally, per_query`. Task 11 consumes.

- [ ] **Step 1: Write the failing test**

Create `tests/test_federation_eval.py`:

```python
"""Federation bar metrics and claim verification, no DB required."""
import pytest

from gin.federation.eval import (
    QueryOutcome,
    claims_verify,
    compute_metrics,
    load_federation_queryset,
)
from gin.federation.schema import WireClaim


def test_load_queryset_validates_class(tmp_path):
    p = tmp_path / "qs.yaml"
    p.write_text(
        "queries:\n"
        "  - id: ok\n    query: q\n    federation_class: b_only\n"
        "    gold_chunk_ids:\n      - n2_doc_002:3\n",
        encoding="utf-8",
    )
    qs = load_federation_queryset(p)
    assert qs[0].federation_class == "b_only"
    assert qs[0].gold_chunk_ids == ("n2_doc_002:3",)

    p.write_text(
        "queries:\n  - id: bad\n    query: q\n    federation_class: nope\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_federation_queryset(p)


def test_claims_verify_substring_normalized():
    chunk = {"n2_doc_002:3": "LANDBACK is described as an organizing  and\nnarrative framework."}
    fetch = chunk.get
    good = [WireClaim(text="LANDBACK is described as an organizing and narrative framework.",
                      span_type="EXACT", cited_chunk_ids=["n2_doc_002:3"])]
    assert claims_verify(good, fetch) is True
    fabricated = [WireClaim(text="LANDBACK was invented in 2031",
                            span_type="EXACT", cited_chunk_ids=["n2_doc_002:3"])]
    assert claims_verify(fabricated, fetch) is False
    missing_chunk = [WireClaim(text="anything", span_type="EXACT",
                               cited_chunk_ids=["nope:0"])]
    assert claims_verify(missing_chunk, fetch) is False
    assert claims_verify([], fetch) is False  # no claims = nothing verified
    uncited = [WireClaim(text="anything", span_type="EXACT", cited_chunk_ids=[])]
    assert claims_verify(uncited, fetch) is False


def test_compute_metrics_perfect_run():
    outcomes = [
        QueryOutcome(id="a1", federation_class="a_answerable", refused=False,
                     routed=False, source_node="node_a"),
        QueryOutcome(id="b1", federation_class="b_only", refused=False,
                     routed=True, source_node="node_b", attribution_verified=True),
        QueryOutcome(id="n1", federation_class="neither", refused=True,
                     routed=True, refusal_reasons={"node_a": "zero_cursors",
                                                   "node_b": "zero_cursors"}),
    ]
    m = compute_metrics(outcomes)
    assert m["routing_false_positives"] == 0
    assert m["routing_recall"] == 1.0
    assert m["routed_answer_attribution_verified"] == 1.0
    assert m["routed_fabrication_rate"] == 0.0
    assert m["honest_refusal_rate"] == 1.0
    assert m["a_answered_locally"] == 1


def test_compute_metrics_failures_visible():
    outcomes = [
        QueryOutcome(id="a1", federation_class="a_answerable", refused=False,
                     routed=True, source_node="node_b"),   # false positive
        QueryOutcome(id="b1", federation_class="b_only", refused=False,
                     routed=True, source_node="node_b",
                     attribution_verified=False),           # fabrication
        QueryOutcome(id="b2", federation_class="b_only", refused=True,
                     routed=False),                          # missed routing
        QueryOutcome(id="n1", federation_class="neither", refused=False,
                     routed=False, source_node="node_a"),    # dishonest answer
    ]
    m = compute_metrics(outcomes)
    assert m["routing_false_positives"] == 1
    assert m["routing_recall"] == 0.5
    assert m["routed_fabrication_rate"] == 1.0
    assert m["honest_refusal_rate"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_eval.py -v`
Expected: FAIL — no module `gin.federation.eval`

- [ ] **Step 3: Write the implementation**

Create `gin/federation/eval.py`:

```python
"""Federation v1 bar: class-labeled queryset, outcomes, metrics.

The eval driver — unlike Node A — legitimately holds credentials for BOTH
node databases, so it performs the attribution verification A architecturally
cannot: every routed claim's text must appear verbatim (whitespace-normalized)
in every chunk it cites, fetched from the answering node's own database.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import psycopg
import yaml

from .schema import WireClaim

VALID_CLASSES = {"a_answerable", "b_only", "neither"}


@dataclass(frozen=True)
class FederationQuery:
    id: str
    query: str
    federation_class: str
    gold_chunk_ids: tuple[str, ...] = ()


def load_federation_queryset(path: str | Path) -> list[FederationQuery]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out: list[FederationQuery] = []
    for q in raw["queries"]:
        cls = q["federation_class"]
        if cls not in VALID_CLASSES:
            raise ValueError(
                f"query {q['id']!r}: federation_class {cls!r} not in "
                f"{sorted(VALID_CLASSES)}"
            )
        out.append(
            FederationQuery(
                id=str(q["id"]),
                query=str(q["query"]),
                federation_class=cls,
                gold_chunk_ids=tuple(q.get("gold_chunk_ids", []) or []),
            )
        )
    return out


def _normalize(s: str) -> str:
    return " ".join(s.split()).lower()


def claims_verify(
    claims: list[WireClaim], fetch_text: Callable[[str], Optional[str]]
) -> bool:
    """True iff every claim cites >=1 chunk and its text appears
    (whitespace-normalized) in EVERY chunk it cites. No claims = False —
    an answer with nothing to verify is not a verified answer."""
    if not claims:
        return False
    for claim in claims:
        if not claim.cited_chunk_ids:
            return False
        for chunk_id in claim.cited_chunk_ids:
            text = fetch_text(chunk_id)
            if text is None or _normalize(claim.text) not in _normalize(text):
                return False
    return True


def verify_claims_in_db(claims: list[WireClaim], database_url: str) -> bool:
    """DB-backed claims_verify against the answering node's chunks table."""
    with psycopg.connect(database_url) as conn:

        def fetch(chunk_id: str) -> Optional[str]:
            row = conn.execute(
                "SELECT text FROM chunks WHERE chunk_id = %s", (chunk_id,)
            ).fetchone()
            return row[0] if row else None

        return claims_verify(claims, fetch)


@dataclass
class QueryOutcome:
    id: str
    federation_class: str
    refused: bool
    routed: bool
    source_node: str = ""
    attribution_verified: Optional[bool] = None
    refusal_reasons: dict = field(default_factory=dict)


def compute_metrics(outcomes: list[QueryOutcome]) -> dict:
    a = [o for o in outcomes if o.federation_class == "a_answerable"]
    b = [o for o in outcomes if o.federation_class == "b_only"]
    n = [o for o in outcomes if o.federation_class == "neither"]
    routed_answers_b = [o for o in b if o.routed and not o.refused]
    verified = [o for o in routed_answers_b if o.attribution_verified]
    return {
        "n_queries": len(outcomes),
        # Bar: 0 — an A-answerable query must never consult the peer.
        "routing_false_positives": sum(1 for o in a if o.routed),
        # Bar: 1.0 — every B-only query must reach the peer.
        "routing_recall": (sum(1 for o in b if o.routed) / len(b)) if b else None,
        # Bar: 1.0 / 0.0 — routed answers verify against B's corpus.
        "routed_answer_attribution_verified": (
            len(verified) / len(routed_answers_b)
        ) if routed_answers_b else None,
        "routed_fabrication_rate": (
            1.0 - len(verified) / len(routed_answers_b)
        ) if routed_answers_b else None,
        # Bar: 1.0 — neither-class queries end in refusal, never an answer.
        "honest_refusal_rate": (
            sum(1 for o in n if o.refused) / len(n)
        ) if n else None,
        "a_answered_locally": sum(
            1 for o in a if not o.refused and not o.routed
        ),
        "per_query": [o.__dict__ for o in outcomes],
    }
```

Create `data/eval/queryset_federation.yaml`:

```yaml
# Federation v1 bar queryset.
# Spec: docs/superpowers/specs/2026-07-13-federation-v1-sovereign-delegation-design.md
# Node A holds corpus_node1.json (institutional); Node B holds corpus_node2.json
# (grassroots). Classes:
#   a_answerable -> A grounds locally; routing is a false positive
#   b_only       -> A cannot ground; must route; B grounds
#   neither      -> both refuse; honest aggregated refusal is the pass
# The two a_answerable queries are the proven single-node controls from
# queryset_twonode.yaml. If a b_only query turns out to ground on A at the
# retrieval floor, replace it with a query deeper in B's corpus — the class
# labels are part of the falsifiable setup, not sacred text.
queries:
  - id: fed_a_2023_anomaly
    query: How much warmer was 2023 than the twentieth-century average?
    federation_class: a_answerable
    gold_chunk_ids:
      - n1_doc_002:1

  - id: fed_a_ocean_acidification
    query: What is ocean acidification and what causes it?
    federation_class: a_answerable
    gold_chunk_ids:
      - n1_doc_004:0

  - id: fed_b_landback
    query: What is LANDBACK and what does it seek to reclaim?
    federation_class: b_only
    gold_chunk_ids:
      - n2_doc_002:3

  - id: fed_b_weact_founding
    query: How was WE ACT for Environmental Justice founded?
    federation_class: b_only
    gold_chunk_ids:
      - n2_doc_007:0
      - n2_doc_007:1

  - id: fed_neither_sports
    query: Who won the national football championship final this year?
    federation_class: neither

  - id: fed_neither_sourdough
    query: What temperature should sourdough bread be baked at?
    federation_class: neither
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_eval.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add gin/federation/eval.py data/eval/queryset_federation.yaml tests/test_federation_eval.py
git commit -m "Federation bar metrics + class-labeled queryset; the driver verifies what A cannot.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 11: Eval driver + live two-node run

**Files:**
- Create: `scripts/eval_federation.py`

**Interfaces:**
- Consumes: everything from Task 10; `FederatedQuery`/`FederatedResponse` (Task 1); httpx.
- Produces: `data/eval_runs/<ts>/federation_metrics.json` artifact; console summary against the bar.

- [ ] **Step 1: Write `scripts/eval_federation.py`**

```python
"""Measure the Federation v1 bar against two live node processes.

Prereqs (three terminals):
    bash scripts/federation_db_setup.sh                      # once
    python scripts/node_serve.py --config config/node_b.yaml # terminal 1
    python scripts/node_serve.py --config config/node_a.yaml # terminal 2
    python scripts/eval_federation.py                        # terminal 3

Bar (spec): routing_false_positives 0; routing_recall 1.0;
routed_fabrication_rate 0.0; honest_refusal_rate 1.0.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx

from gin.federation.eval import (
    QueryOutcome,
    compute_metrics,
    load_federation_queryset,
    verify_claims_in_db,
)
from gin.federation.schema import FederatedQuery, FederatedResponse

DEFAULT_OUT = ROOT / "data" / "eval_runs"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-a-url", default="http://127.0.0.1:8471")
    parser.add_argument(
        "--node-b-db",
        default="postgresql://gin:gin@localhost:5432/gin_node_b",
        help="verification DB for routed answers (the answering node's)",
    )
    parser.add_argument(
        "--queryset", default=str(ROOT / "data" / "eval" / "queryset_federation.yaml")
    )
    parser.add_argument("--secret", default="dev-federation-secret")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    queries = load_federation_queryset(args.queryset)
    headers = {"Authorization": f"Bearer {args.secret}"}
    outcomes: list[QueryOutcome] = []

    with httpx.Client(timeout=args.timeout) as client:
        for q in queries:
            fq = FederatedQuery(query=q.query, origin_node="eval_driver", hop_count=0)
            r = client.post(
                f"{args.node_a_url}/v1/federated/query",
                headers=headers,
                json=fq.model_dump(),
            )
            r.raise_for_status()
            resp = FederatedResponse.model_validate(r.json())

            if resp.refusal is not None:
                # Delegation happened iff the peer contributed a reason.
                routed = bool(resp.refusal.peer_reasons)
                outcome = QueryOutcome(
                    id=q.id,
                    federation_class=q.federation_class,
                    refused=True,
                    routed=routed,
                    refusal_reasons={
                        resp.refusal.node_id: resp.refusal.reason,
                        **resp.refusal.peer_reasons,
                    },
                )
            else:
                routed = resp.federation is not None
                verified = (
                    verify_claims_in_db(resp.answer.claims, args.node_b_db)
                    if routed
                    else None
                )
                outcome = QueryOutcome(
                    id=q.id,
                    federation_class=q.federation_class,
                    refused=False,
                    routed=routed,
                    source_node=resp.answer.node_id,
                    attribution_verified=verified,
                )
            print(
                f"[{q.id}] class={q.federation_class} refused={outcome.refused} "
                f"routed={outcome.routed} source={outcome.source_node!r} "
                f"verified={outcome.attribution_verified}"
            )
            outcomes.append(outcome)

    metrics = compute_metrics(outcomes)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact = run_dir / "federation_metrics.json"
    artifact.write_text(
        json.dumps(
            {
                "run_id": ts,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "node_a_url": args.node_a_url,
                "queryset": str(args.queryset),
                **metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = {k: v for k, v in metrics.items() if k != "per_query"}
    print(json.dumps(summary, indent=2))
    print(f"artifact: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-test**

Run: `./venv/Scripts/python.exe scripts/eval_federation.py --help`
Expected: usage text, exit 0.

- [ ] **Step 3: Commit the driver**

```bash
git add scripts/eval_federation.py
git commit -m "Federation eval driver: measure the bar against two live nodes.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 4: LIVE RUN (needs GPU/model; not CI)**

Start both nodes (separate terminals or background processes), then run the driver:

```bash
# terminal 1
./venv/Scripts/python.exe scripts/node_serve.py --config config/node_b.yaml
# terminal 2
./venv/Scripts/python.exe scripts/node_serve.py --config config/node_a.yaml
# terminal 3
./venv/Scripts/python.exe scripts/eval_federation.py
```

Expected against the bar: `routing_false_positives: 0`, `routing_recall: 1.0`,
`routed_fabrication_rate: 0.0`, `routed_answer_attribution_verified: 1.0`,
`honest_refusal_rate: 1.0`, `a_answered_locally: 2`.

If both models on GPU exceed the 12 GB card, set `n_gpu_layers: 0` in
`config/node_b.yaml` and restart node B (CPU decode is slower; the driver
timeout of 600 s covers it).

**If a `b_only` query fails its class** (A answers it locally, or B refuses
it): the queryset is part of the falsifiable setup — inspect the per-query
artifact, replace the query with one that genuinely separates the corpora
(deeper in B's corpus, e.g. another `n2_doc_*` chunk), re-run, and record the
substitution in the commit message. Do NOT loosen the metrics.

- [ ] **Step 5: Commit the run artifact**

```bash
git add data/eval_runs/<ts>/federation_metrics.json
git commit -m "Federation v1 bar measured live: <one-line summary of the metrics>.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 12: Documentation + final validation + push

**Files:**
- Modify: `README.md` (status table row + new federation quick-start subsection after the scan-first section)
- Modify: `architecture.md` (Phase 3 checklist, ~line 382)
- Modify: `docs/GIN_Node_Architecture_v1.md` (transport note near line 119)

**Interfaces:**
- Consumes: the measured artifact from Task 11.
- Produces: docs consistent with what v1 does and defers.

- [ ] **Step 1: Update `architecture.md` Phase 3 checklist**

Replace the two unchecked items under `### Phase 3 — Federation`:

```markdown
- ✅ Sovereign delegation loop (zero-cursor routing v1) — two node processes,
  HTTP+JSON schema-first transport behind the `PeerClient` seam
  (`gin/federation/`); pre-commitment grounding failures delegate to the
  configured peer; B's answer relays with attribution intact and explicitly
  marked as B's. Measured: `data/eval_runs/<ts>/federation_metrics.json`.
  Spec: docs/superpowers/specs/2026-07-13-federation-v1-sovereign-delegation-design.md
- 🔲 Merkle diff sync of anchor metadata (spec #2 — load-bearing at N>2)
- 🔲 gRPC/QUIC wire (swap inside `PeerClient`; institutional target)
- 🔲 Trust weights, PKI/mTLS, peer selection
```

(Substitute the real run timestamp for `<ts>`.)

- [ ] **Step 2: Add README federation quick-start + status row**

Add a subsection after the scan-first section (before `## Manifest version handoff…`):

```markdown
### Federation v1 — sovereign delegation (two nodes, one machine)

```bash
# one-time: create + populate per-node databases (A: corpus_node1, B: corpus_node2)
bash scripts/federation_db_setup.sh

# serve both nodes (two terminals)
python scripts/node_serve.py --config config/node_b.yaml
python scripts/node_serve.py --config config/node_a.yaml

# measure the bar
python scripts/eval_federation.py
```

When Node A cannot ground a query (retrieval floor or zero cursors before any
content), it delegates to Node B over HTTP; B runs its own SEAR decode and
returns the answer + attribution, relayed with a federation provenance layer
(`answered_by`, `hop_count`). B's corpus never leaves B except as the spans it
chose to emit. Requests at `hop_count >= 1` are never re-delegated. Bar and
scope: docs/superpowers/specs/2026-07-13-federation-v1-sovereign-delegation-design.md.
```

And update the status table's federation row from `🔲` to:

```markdown
| Federation routing with sync metadata (Phase 3) | ✅ v1 sovereign delegation loop measured (run `<ts>`: routing FP 0, recall 1.0, routed fabrication 0.0, honest refusal 1.0); Merkle sync + trust weights deferred to spec #2+ |
```

- [ ] **Step 3: Add transport note to `docs/GIN_Node_Architecture_v1.md`**

Immediately after the `**Protocol**: gRPC over QUIC (primary)…` line, add:

```markdown
> **v1 implementation note (2026-07):** the shipped two-node loop speaks
> HTTP/1.1 + JSON behind the `PeerClient` seam (`gin/federation/client.py`);
> the Pydantic schema is the protocol contract. gRPC/QUIC remains the
> institutional-deployment target and replaces the transport without touching
> routing logic.
```

- [ ] **Step 4: Full suite + final validation**

Run: `./venv/Scripts/python.exe -m pytest tests/ -q`
Expected: all green.

Run: `git status --short`
Expected: only the intended doc modifications; `.env`, `data/models/`, `data/cold_node_*` untracked/ignored.

- [ ] **Step 5: Commit and push**

```bash
git add README.md architecture.md docs/GIN_Node_Architecture_v1.md
git commit -m "Document Federation v1: the delegation loop is measured; sync and trust are next.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push origin main
```

---

## Self-review notes (already applied)

- **Spec coverage:** falsifiable bar → Tasks 10/11; architecture modules table → Tasks 1–7 (one module each); wire protocol → Task 1; data flow + loop prevention → Tasks 6/7/8; sovereignty semantics → router relays without re-verification, driver verifies (Tasks 6/10); auth + failure handling → Tasks 5/7 (401, unreachable→honest refusal, version refusal); three test tiers → unit (Tasks 1–7), real-socket integration (Task 8), live bar (Task 11); DB split → Task 9; doc updates → Task 12; out-of-scope list → nothing in this plan builds any of it.
- **Deviation from spec, intentional:** the spec's "integration tier via GreedyMaskDecoder" is replaced by stubbed `answer_fn`s over real sockets — strictly more faithful to the "real network boundary" claim and simpler (the deterministic decoder would still need hand-built bundles; the decode path itself is already covered by existing tests and the live run).
- **Type consistency check:** `WireClaim` fields (`text/span_type/cited_chunk_ids`) used identically in Tasks 1, 4, 5, 6, 10; `RoutedResult.corpus_fingerprint` consumed in Task 7 (`routed.corpus_fingerprint or fingerprint`); `refusal_reason` values produced in Task 3 match the `RefusalReason` literal in Task 1; `NodeConfig` field order in test constructors matches Task 2's dataclass.
