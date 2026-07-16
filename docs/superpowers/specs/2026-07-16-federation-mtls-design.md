# Federation mTLS: Self-Signed Pinned Peer Authentication (Design)

**Date:** 2026-07-16
**Status:** approved design, pre-implementation
**Phase:** GIN Phase 3 (federation), fifth sub-project

---

## Falsifiable claim

Given the existing 3-node deployment (node_a/b/c, plus the trust-gated
config variant from sub-project 4), replacing plaintext HTTP + shared-secret
bearer auth with mutual TLS over pinned self-signed certificates must not
change any routing or ranking behavior — only how a connection is
authenticated. Every prior live-eval result must reproduce exactly over the
new transport, and the new mechanism must demonstrably reject callers it
doesn't recognize.

| Metric | Bar |
|---|---|
| Peer presenting its correctly pinned cert | request succeeds, identical response to today |
| Peer presenting an unpinned/wrong cert | TLS handshake fails; request never reaches FastAPI routing |
| Peer presenting no client cert | rejected (`ssl_cert_reqs=CERT_REQUIRED`) |
| Regression: sub-project 3 live eval (`eval_peer_selection.py`) over mTLS | reproduces exact prior result set (precision@1 1.0, avg peers tried 1.0, fabrication 0.0, attribution 1.0, honest refusal 1.0) |
| Regression: sub-project 4 gated/ungated live eval over mTLS | reproduces exact prior result set (gated 0-contact, honest refusal 1.0; ungated regression exact) |
| Bearer-token code path | fully removed; no request is accepted without a valid mTLS handshake |

If any bar fails, the design is wrong, not the eval.

## Scope decisions (made 2026-07-16, with rationale)

1. **mTLS only this phase — no transport-framework swap.** The driver is
   preparing for real multi-machine deployment (nodes leaving localhost for
   the first time), which is a security-of-connection problem, not a
   performance problem. FastAPI/uvicorn (server) and httpx (client) already
   support TLS/mTLS natively (`ssl_certfile`/`ssl_keyfile`/`ssl_ca_certs`,
   `verify=`/`cert=`), so gRPC/QUIC isn't required to get authenticated
   transport. Bundling the transport-framework swap in would touch the same
   code twice and delay the actually-urgent piece. gRPC/QUIC remains a
   separate future sub-project, unblocked by this one — `HttpPeerClient`
   still implements the existing `PeerClient` Protocol, so a later swap is
   still a change *inside* the client, not to its interface.
2. **Self-signed + pinned certificate, not a CA, not a Tailscale
   dependency.** The trigger is independently-operated nodes with no shared
   operator — a CA introduces a governance dependency (who runs it, who's
   trusted to admit nodes) that cuts against GIN's node-sovereignty stance,
   and a VPN mesh like Tailscale requires every operator to join one shared
   tailnet, which doesn't hold once federation crosses trust boundaries GIN
   doesn't control. Each node generates its own keypair; trust is
   established the way SSH `known_hosts` trust is — out-of-band exchange,
   pinned locally, no third party vouches for anyone.
3. **Pin the full certificate file, not just a fingerprint hash.**
   Standard TLS libraries (Python `ssl`, `httpx`) validate a presented
   certificate against a CA bundle, not an arbitrary hash comparison.
   Storing the peer's actual self-signed certificate and using it as that
   connection's sole trusted root lets ordinary chain validation do the
   work — no custom handshake hooks, no hand-rolled fingerprint
   verification inside the TLS layer. The SHA-256 fingerprint is still
   printed at keygen time and is the value operators actually compare
   out-of-band; the full cert is what's mechanically enforced.
4. **Remove the shared-secret bearer token entirely.** mTLS fully subsumes
   its job — the peer's cert *is* the credential, cryptographically bound
   to the connection rather than a copyable string in a header. Keeping
   both would add provisioning/rotation surface (a secret and a cert per
   peer relationship) for a threat mTLS already covers, with no bug class
   the shared secret catches that mTLS doesn't. `GIN_FED_SECRET` env
   override is removed along with the config field.
5. **Long-lived certs (10 years), manual re-pinning for rotation.**
   Rotation and automated revocation are institutional-tier concerns, same
   posture the prior specs already used to defer PKI/mTLS itself. Solving
   them now would be building infrastructure for a governance model
   (Council-operated CA, scheduled rotation) that doesn't exist yet. If a
   cert needs to change, the operator regenerates and re-exchanges it —
   the same manual step as initial pinning.
6. **Breaking config change, no back-compat shim.** Every existing
   `config/node_*.yaml` needs regenerated certs and updated peer blocks;
   there is no dual-mode (bearer-or-cert) auth path. Consistent with the
   project's general practice of changing the code directly rather than
   supporting two auth mechanisms indefinitely.

## Architecture

Extends `gin/federation/` (built across the four prior sub-projects).

| Module | Change |
|---|---|
| `gin/federation/config.py` | `NodeConfig` gains `cert_path: str`, `key_path: str`; drops `shared_secret`. `PeerConfig` gains `pinned_cert_path: str`; its implicit-trust-via-shared-secret model is gone. `load_node_config` resolves the new paths the same way it already resolves `cold_path`/`model_path`; `GIN_FED_SECRET` env override is removed. |
| `scripts/node_keygen.py` (new) | Generates an ECDSA P-256 self-signed keypair (`CN=<node_id>`, 10-year validity) via the `cryptography` library. Writes `certs/<node_id>/{cert.pem, key.pem}`. Prints the SHA-256 fingerprint for out-of-band pinning confirmation. Cert-generation logic is a plain importable function, not just a CLI entry point, so tests can call it directly to produce throwaway cert pairs. |
| `gin/federation/server.py` | `_check_auth` (bearer-token dependency, `server.py:125-128`) is replaced by a dependency that reads the verified client certificate off the connection and resolves it to a `node_id` by matching against the configured `peers` list. Endpoint signatures (`server.py:146-244`) are unchanged — handlers still receive the caller's `node_id`, now cryptographically backed instead of string-backed. |
| `scripts/node_serve.py` | `uvicorn.run` (`node_serve.py:69`) gains `ssl_certfile=config.cert_path`, `ssl_keyfile=config.key_path`, `ssl_ca_certs=<bundle built from all peers' pinned_cert_path>`, `ssl_cert_reqs=ssl.CERT_REQUIRED`. The CA bundle is built once at startup by concatenating every configured peer's pinned cert into a temp file. |
| `gin/federation/client.py` | `HttpPeerClient` (`client.py:47-104`) constructs one `httpx.Client` per peer relationship: `verify=peer.pinned_cert_path` (trust only that peer's pinned cert as CA), `cert=(config.cert_path, config.key_path)` (present own identity). The `Authorization: Bearer` header (`client.py:60`) is removed. The `PeerClient` Protocol (`client.py:37-44`) is unchanged. |
| `config/node_a.yaml`, `node_b.yaml`, `node_c.yaml`, `node_a_trust_gated.yaml` | `url` scheme becomes `https://`; each node adds top-level `cert_path`/`key_path`; each peer entry gains `pinned_cert_path`; `shared_secret` line removed. |
| `certs/<node_id>/{cert.pem, key.pem}` (new, generated, gitignored) | Per-node identity artifacts, output of `node_keygen.py`. Not committed — same treatment as any other locally-generated credential. |

### Open implementation question (pre-plan spike)

Exactly how to retrieve the verified client certificate from a FastAPI
request depends on what the ASGI server exposes — uvicorn's TLS-extension
support for this varies and needs a quick spike before the implementation
plan can size that task precisely. This is called out explicitly rather
than hand-waved; the plan will carry it as its own early task with a
concrete fallback (e.g., a thin layer that pulls the peer cert directly
from the raw socket if the ASGI extension path doesn't expose it cleanly).

## Data flow

1. **Bootstrap (once per node):** operator runs `node_keygen.py` for their
   node — `cert.pem`/`key.pem` written to `certs/<node_id>/`, SHA-256
   fingerprint printed to the terminal.
2. **Pinning (once per peer relationship, out-of-band):** operators
   exchange `cert.pem` files however they'd exchange any credential today.
   Each operator adds the peer's `node_id`, `url`, and `pinned_cert_path`
   to their own config, confirming the printed fingerprint matches what
   was communicated out-of-band.
3. **Server startup:** `node_serve.py` builds a CA bundle from all
   configured peers' pinned certs and starts uvicorn requiring and
   verifying client certificates against that bundle.
4. **Client construction:** `HttpPeerClient` builds a distinct
   `httpx.Client` per peer, trusting only that peer's pinned cert as CA
   and presenting this node's own cert/key.
5. **Per-request handshake:** an outbound request succeeds only if the
   responding server's cert matches the pinned one *and* the server
   independently validates the caller's client cert against its own CA
   bundle (built from its own peers list) — authentication is mutual and
   symmetric, not just client-verifies-server.
6. **Authorization in the handler:** the server resolves the verified
   client cert's fingerprint to a `node_id` via the peers list and passes
   it to the endpoint handler — exactly where `_check_auth` previously
   granted or denied access, now backed by the handshake instead of a
   header string.

## Error handling

- **Peer presents an unpinned/wrong cert:** TLS handshake fails at the
  transport layer; the request never reaches FastAPI routing — a stronger
  guarantee than today's bearer check, which only rejected inside the
  application after the connection was already established.
- **Peer presents no client cert:** rejected by `ssl_cert_reqs=CERT_REQUIRED`
  before the handshake completes.
- **A configured peer's pinned cert file is missing/unreadable at server
  startup:** fail fast — the CA bundle can't be built, so the node refuses
  to start rather than silently serving with a stale or partial bundle.
- **This node's own cert/key is missing:** fail fast at startup, the same
  treatment as today's missing `model_path`/`database_url`.
- **Expired cert (after the 10-year validity window):** out of scope for
  this phase; manifests as a handshake failure. Operator regenerates and
  re-pins — documented, not automated (decision 5).

## Testing — three tiers

1. **Unit (no I/O beyond temp files, no real socket):** `node_keygen.py`'s
   cert-generation function produces a valid self-signed cert with the
   expected `CN`; CA-bundle-building (concatenating peer certs) is pure
   file logic; the "verified cert → `node_id`" resolution function is
   tested directly against a fabricated fingerprint/peers-list pair,
   without driving it through an actual handshake.
2. **Integration (real-socket, extends `tests/test_federation_loop.py:24-90`,
   which already spins up real uvicorn servers on background threads):**
   generate two throwaway cert pairs via the keygen library function, wire
   mTLS into two real servers, and confirm: a request with the correctly
   pinned cert succeeds; a request with an unpinned/wrong cert is rejected
   at the handshake; a request with no client cert is rejected. This
   directly replaces what the old bearer-token tests checked, now proving
   cryptographic identity instead of a shared string.
3. **Live eval:** regenerate certs and config for node_a/b/c and
   `node_a_trust_gated`, then rerun `eval_federation.py`,
   `eval_peer_selection.py`, and the sub-project 4 gated/ungated eval
   **unmodified** — must reproduce the exact prior numbers over the new
   transport (per the falsifiable-claim table), proving this is a
   transport-layer change with zero behavioral delta to routing or
   ranking.

## Out of scope (later, in likely order)

1. Certificate rotation / automated renewal
2. Automated revocation (CRL/OCSP) — manual re-pinning is the only
   mechanism this phase provides
3. CA-based issuance / centralized trust
4. gRPC/QUIC transport swap
5. Blending trust into the RRF-fused score, query-time domain
   classification, runtime/API-driven weight updates, Epistemic Council
   automation — unchanged carryover from sub-project 4, untouched by this
   phase

## Documentation updates shipped with implementation

- `architecture.md` Phase 3 checklist: split the existing combined
  "gRPC/QUIC wire, PKI/mTLS" line — mark mTLS done/measured with live-eval
  numbers, leave gRPC/QUIC as the remaining open item.
- `README.md`: a "peer authentication" subsection covering the keygen +
  out-of-band pinning workflow and the updated config shape.
- `docs/GIN_Node_Architecture_v1.md`: a v1 implementation note (matching
  the existing peer-selection/trust-weights notes) describing the
  self-signed-and-pinned trust model, explicitly noting rotation/CA/
  revocation remain future work.

## New dependencies

`cryptography` (Python, for self-signed keypair/cert generation in
`node_keygen.py`) — not currently in `requirements.txt`. Chosen over
shelling out to the `openssl` CLI for portability (this environment is
Windows; requiring `openssl` on `PATH` is an extra operational dependency
the project doesn't otherwise have). TLS itself needs no new dependency —
uvicorn's `ssl_*` options and httpx's `verify=`/`cert=` are built on
Python's stdlib `ssl` module, already exercised transitively by both
existing dependencies.
