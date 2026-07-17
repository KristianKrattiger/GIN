# Streaming Reasoning Trace: Incremental Claim Events (Design)

**Date:** 2026-07-16
**Status:** approved design, pre-implementation
**Phase:** GIN Phase 3 (federation), sixth sub-project — reframes the deferred "gRPC/QUIC wire" architecture-checklist line

---

## Falsifiable claim

A new streaming endpoint delivers incremental visibility into local synthesis
— which claims get admitted, as they're admitted — without changing the
existing non-streaming endpoint's behavior at all, and without requiring a
real model to test the mechanism end-to-end.

| Metric | Bar |
|---|---|
| Streamed claim count/content vs. final answer's claim list | identical, same order |
| Query that refuses instantly (zero cursors) | zero `claim_admitted` events before the terminal event |
| Non-streaming `/v1/federated/query` behavior | byte-for-byte unchanged — same response shape, same auth, same routing, existing test suite fully green with no modifications |
| Real end-to-end run (manual, real model) | first `claim_admitted` event observably arrives before total request completion for a query taking more than a few seconds — proves genuine incremental delivery, not a fast final dump |
| New dependencies | none |

If any bar fails, the design is wrong, not the eval.

## Scope decisions (made 2026-07-16, with rationale)

1. **Reframes "gRPC/QUIC," doesn't build it.** The original architecture
   doc's rationale for gRPC/QUIC was "low latency... stream support for long
   reasoning traces" — investigation found real evidence for the second half
   (median synthesis time ~50s/query, up to 416s measured in
   `data/eval_runs/*/meta.json`) but none yet for the first (no measured
   bottleneck in HTTP/1.1+JSON+mTLS). gRPC's actual mature transport is
   HTTP/2, not QUIC — QUIC/HTTP3 support in gRPC (especially Python's
   implementation) is still experimental, so "gRPC over QUIC" as literally
   stated isn't an off-the-shelf combination today. Building gRPC (new
   protobuf schema, a parallel server process since gRPC doesn't share
   ASGI's request lifecycle, the mTLS trust model re-implemented against
   grpc's own credential API) to solve a trace-visibility need that NDJSON
   streaming over the existing stack solves just as well is disproportionate.
   gRPC/QUIC remains available as a later transport-framework swap if a
   concrete need for cross-language clients or connection migration
   (roaming/cellular) shows up — this sub-project doesn't block that, it
   just doesn't build it to solve a problem it doesn't have.
2. **Claim-close granularity, not token-level.** SEAR's decode-time
   constraint (`sear/processor.py`) already has a per-token state machine —
   span-begin, span-close, citation-complete — but only claim-close
   (`_close_span`, `sear/processor.py:506-534`) becomes a streamed event.
   Token-level streaming would require switching the blocking
   `llm.create_completion(...)` call to `stream=True` and restructuring the
   decode loop in `gin/corpus/generate.py` — a materially bigger, riskier
   change to the synthesis pipeline itself for a capability nobody asked
   for here (the explicit ask was claim-trace visibility, not token
   streaming).
3. **New endpoint (`POST /v1/federated/query/stream`), not content
   negotiation on the existing one.** Zero risk to the existing endpoint's
   contract; explicit opt-in for callers that want trace visibility.
4. **Terminal-node-only — no multi-hop relay, and no special-casing to
   enforce it.** `HttpPeerClient` (peer-to-peer delegation) is untouched;
   inter-node calls stay request/response. Because the event sink is only
   ever populated during *local* synthesis, a query that ends up delegating
   naturally streams its own (possibly empty) local-attempt trace, goes
   quiet while `peer_client.query()` blocks on the remote peer, then emits
   the terminal event once the delegate answers. No code needs to detect
   "this query delegated, suppress streaming" — the queue is simply empty
   during that window because nothing local is producing events then.
5. **`contextvars.ContextVar` carries the event sink, not a threaded
   parameter.** Propagating an optional sink callback through every function
   in the `answer_or_delegate` → `answer_fn` → `generate.py` → SEAR call
   chain would touch signatures across ~4-5 files for a feature only one
   caller (the new streaming endpoint) uses. A `ContextVar`, set once at the
   top of the streaming endpoint and read only where an event actually
   fires, keeps every existing signature untouched.
   `asyncio.to_thread` propagates the current context into its worker
   thread automatically (stdlib guarantee), so this works cleanly with the
   thread-per-request pattern below.
6. **SEAR gets exactly one new optional constructor parameter and stays
   decoupled from text-decoding/chunk-id-resolution/HTTP concerns.**
   `ExtractiveCopyConstraint.__init__` gains
   `on_segment_closed: Optional[Callable[[Segment], None]] = None`, called
   from `_close_span` with the raw `Segment` (token IDs, not decoded text)
   right after it's appended to `self.segments`
   (`sear/processor.py:516-523`). Text decoding and chunk-ID resolution
   already happen exactly this way, just later — `segments_to_raw_claims`
   (`gin/eval/claims.py:86-114`) takes a `detok` callable
   (already a stdlib-cheap, synchronous operation — no model forward
   pass — established pattern at `gin/corpus/generate.py:185`) and a
   `doc_index_to_chunk_id` map. The event-shaping closure that wires
   `on_segment_closed` to the `ContextVar` sink lives in `gin/corpus/generate.py`,
   where `detok` and the chunk-ID map are already in scope — SEAR itself
   never imports anything from `gin/federation/` or `gin/eval/`.
7. **A background thread runs the unchanged synthesis call; an async
   generator drains a queue.** The streaming endpoint doesn't reimplement
   `answer_fn`/`answer_or_delegate` — it wraps the exact same call in
   `asyncio.to_thread(...)`, while an async generator polls a
   `queue.Queue()` the event sink pushes into, yielding NDJSON lines as
   they arrive, until the background call resolves and the terminal event
   fires.
8. **Client disconnect doesn't cancel in-flight synthesis.** llama.cpp's
   blocking generation call can't be interrupted mid-token without deeper
   changes to the decode loop itself. If nobody's listening when synthesis
   finishes, the result is just discarded. Documented limitation, not
   solved here — matches the "don't touch the LLM call" scope decision
   above.
9. **NDJSON, not SSE.** Every current and near-term caller is a Python
   driver/agent, not a browser — SSE's `EventSource`-oriented framing
   (event/id/retry semantics, reconnection) buys nothing here. Plain
   newline-delimited JSON is simpler to emit (`StreamingResponse`) and
   simpler to consume (`httpx`'s `.iter_lines()`). Revisiting this is cheap
   later since the event schema itself doesn't depend on the framing.
10. **No new dependencies.** Starlette's `StreamingResponse` (already a
    transitive FastAPI dependency), stdlib `contextvars`/`queue`/`threading`,
    and the existing Pydantic schema patterns cover everything.

## Architecture

Extends `sear/` (decode-time constraint), `gin/corpus/generate.py`
(synthesis orchestration), and `gin/federation/` (the wire layer built
across sub-projects 1-5).

| Module | Change |
|---|---|
| `sear/processor.py` | `ExtractiveCopyConstraint.__init__` gains `on_segment_closed: Optional[Callable[[Segment], None]] = None`. `_close_span` (`:506-534`) calls it, if set, immediately after `self.segments.append(...)` (`:516-523`). No other change — SEAR remains unaware of HTTP, text decoding, or chunk-ID resolution. |
| `gin/federation/trace_events.py` (new) | Three Pydantic event models — `RetrievalSettledEvent` (`synthesis_mode`, `manifest_hash`, `chunk_count`), `ClaimAdmittedEvent` (`claim: WireClaim`, reusing the existing schema exactly), `SynthesisCompleteEvent` (`response: FederatedResponse`, the exact terminal shape the non-streaming endpoint already returns) — a `TraceEvent` union, and `current_event_sink: ContextVar[Optional[Callable[[TraceEvent], None]]]` (default `None`). |
| `gin/corpus/generate.py` | Wherever `ExtractiveCopyConstraint` is constructed (`:235`), wire `on_segment_closed` to a closure that: decodes the segment via the already-available `detok` callable (`:185`) and the doc-index→chunk-ID map (same logic `segments_to_raw_claims` already uses), builds a `ClaimAdmittedEvent`, and pushes it to `current_event_sink.get()` if set. Also pushes a `RetrievalSettledEvent` once retrieval/manifest-build completes, before decode starts — reuses data already computed for `retrieval_manifest_hash`. |
| `gin/federation/server.py` | New endpoint `POST /v1/federated/query/stream`. Builds a `queue.Queue()`, sets `current_event_sink` to a closure pushing onto it, runs the existing `answer_fn`/`answer_or_delegate` call (unmodified) via `asyncio.to_thread(...)`, and returns a `StreamingResponse` backed by an async generator that drains the queue and yields NDJSON lines until the background call resolves, then yields a final `SynthesisCompleteEvent` line and closes. The existing `/v1/federated/query` endpoint is completely untouched. |
| `gin/federation/client.py` (`HttpPeerClient`) | **Unchanged.** Peer-to-peer delegation stays request/response; this sub-project only adds a new origin-facing endpoint external callers use directly. |
| `tests/test_streaming_endpoint.py` (new) | `TestClient`-based. A fake `answer_fn` manually pushes events through `current_event_sink` (the same mechanism the real chain uses) to prove endpoint mechanics — ordering, terminal event, content-type, empty-trace-on-instant-refusal — decoupled from real SEAR internals, matching how every other federation endpoint test in this codebase already injects fakes rather than running a real model. |
| `tests/test_processor.py` | New test(s): `on_segment_closed` fires with the correct `Segment` at span-close, using the file's existing scripted-token test harness. |

## Data flow

1. An external caller (driver, agent — not a peer node) POSTs a
   `FederatedQuery` to `/v1/federated/query/stream`.
2. The endpoint sets `current_event_sink` to a closure pushing onto a
   fresh `queue.Queue()`, then starts the existing `answer_fn` (or
   `answer_or_delegate`, at hop 0) via `asyncio.to_thread(...)` — the exact
   same call path `/v1/federated/query` already uses, unmodified.
3. `asyncio.to_thread` propagates the current `ContextVar` state into the
   worker thread, so the deep call chain (unchanged code) transparently has
   access to the sink without any signature changes.
4. As retrieval settles and, during decode, as each claim span closes,
   `generate.py`'s wiring pushes `RetrievalSettledEvent`/`ClaimAdmittedEvent`
   instances onto the queue via the sink.
5. Back on the request's event loop, an async generator polls the queue
   and yields each event as an NDJSON line as soon as it arrives —
   genuinely incremental, not buffered until the end.
6. When the background call resolves (answer or refusal, local or
   delegated), the generator yields a final `SynthesisCompleteEvent`
   carrying the exact `FederatedResponse` the non-streaming endpoint would
   have returned, and the stream closes.
7. If local synthesis refuses and the router delegates (hop 0 only,
   existing logic untouched), no new events fire during the peer-to-peer
   HTTP call (`HttpPeerClient` doesn't touch the sink) — the caller sees a
   quiet gap, then the terminal event once the delegate responds.

## Error handling

- **Background synthesis raises:** caught in the async generator, emitted
  as a terminal event shaped like a refusal with reason `internal_error`,
  not a raw traceback or a silently dead stream.
- **Client disconnects mid-stream:** the background `asyncio.to_thread`
  call runs to completion regardless; its result is discarded if nobody's
  reading. No cancellation — documented limitation (scope decision 8).
- **Instant refusal (zero cursors, no retrieval):** no spans ever open, so
  no `claim_admitted` events queue; the terminal event fires as soon as
  `answer_fn` returns — falls out of the design with no special-casing.
- **Delegated query:** covered under scope decision 4 — a quiet gap during
  the peer wait, not an error, not specially detected.

## Testing — three tiers

1. **Unit (`tests/test_processor.py`):** `on_segment_closed` fires with the
   correct `Segment` content when a span closes, using the file's existing
   scripted-token/fake-logits test harness — no HTTP, no real model.
2. **Endpoint (`tests/test_streaming_endpoint.py`, `TestClient`):** a fake
   `answer_fn` manually drives `current_event_sink` to prove the endpoint's
   queue-draining, event ordering, NDJSON content-type, and terminal-event
   mechanics — decoupled from real SEAR internals, matching every other
   federation endpoint test's use of injected fakes over a real model.
   Cases: multiple claims stream in order matching the final answer;
   instant refusal streams zero claim events; a synthesis-raising fake
   yields an `internal_error` terminal event instead of a broken stream.
3. **Manual live verification:** one real end-to-end run against a real
   model (same posture as the mTLS work's live-eval gate — this codebase
   has never run a real model inside the automated suite) confirming
   `claim_admitted` events for a real query arrive measurably before the
   final response, for a query long enough to matter (multi-second decode).

## Out of scope (later, in likely order)

1. Token-level streaming (would require `stream=True` + decode-loop
   restructuring in `gin/corpus/generate.py`)
2. Multi-hop trace relay (a driver watching node A's delegate-to-B trace
   live, relayed through A)
3. Cancelling in-flight synthesis on client disconnect
4. SSE framing (if a browser-facing consumer ever needs it — the event
   schema doesn't change, only the wire framing would)
5. gRPC over HTTP/2 (if cross-language client interop becomes a concrete
   need)
6. gRPC over QUIC (blocked on Python ecosystem maturity, not a GIN-side
   decision)
7. Finer SEAR-internal events (citation-sub-step, connective-phrase,
   divergent-mode requirement-satisfied) — the `on_segment_closed` hook
   point doesn't preclude adding more later, but claim-close is the
   granularity that was actually asked for

## Documentation updates shipped with implementation

- `architecture.md` Phase 3 checklist: replace the `🔲 gRPC/QUIC wire` line
  with the measured streaming-trace result once shipped, and explicitly
  note gRPC/QUIC-as-transport-swap is a distinct, still-open, still-later
  item (same pattern as how the mTLS spec split "PKI/mTLS" out of the
  combined checklist line).
- `README.md`: a "streaming reasoning trace" subsection alongside the
  peer-authentication one, showing the NDJSON event shapes and an example
  client loop consuming `/v1/federated/query/stream`.
- `docs/GIN_Node_Architecture_v1.md`: a v1 implementation note (matching
  the existing peer-selection/trust-weights/mTLS notes' pattern) at the
  "Protocol: gRPC over QUIC" bullet (`:119`), clarifying that trace
  streaming shipped over NDJSON on the existing HTTP/mTLS stack, and that
  gRPC/QUIC remains the deferred institutional-deployment target for the
  transport itself, not yet built.

## New dependencies

None — reuses Starlette's `StreamingResponse` (already a transitive
FastAPI dependency), stdlib `contextvars`/`queue`/`threading`, and the
existing Pydantic schema/`WireClaim`/`FederatedResponse` types.
