# Streaming Reasoning Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new streaming endpoint (`POST /v1/federated/query/stream`) that delivers incremental visibility into local synthesis — retrieval settling, then each admitted claim as it closes — as NDJSON, without changing the existing `/v1/federated/query` endpoint's behavior at all.

**Architecture:** SEAR's decode-time constraint (`sear/processor.py`) gains one optional constructor hook fired on every claim-close. A `contextvars.ContextVar`, read only at the `gin.corpus` layer, carries an optional per-request sink so the hook can reach the HTTP layer without any existing function signature between them changing. `gin.federation` (which already depends on `gin.corpus`) translates the corpus-tier primitive events into its own wire types and streams them; `gin.corpus` itself never imports anything from `gin.eval` or `gin.federation`, preserving the codebase's existing one-directional layering (`sear` → `gin.corpus` → `gin.eval` → `gin.federation`).

**Tech Stack:** Python 3.10+, stdlib `contextvars`/`queue`/`asyncio` only, Starlette's `StreamingResponse` (already a transitive FastAPI dependency), existing Pydantic schema patterns. No new dependencies.

## Global Constraints

- No new dependencies — stdlib + Starlette's existing `StreamingResponse` only.
- `gin/corpus/` must not import anything from `gin/eval/` or `gin/federation/` — this is an existing, unbroken invariant in the codebase (confirmed: `gin/corpus/generate.py` currently has zero imports from either); the corpus-tier trace event types must stay dependency-free primitives, with wire-type translation happening only in `gin/federation`.
- The existing `/v1/federated/query` endpoint's behavior must be byte-for-byte unchanged — same response shape, same routing, existing test suite green with zero modifications to any test file it currently has.
- `_close_span` (`sear/processor.py:506-534`) is the only hook point — one callback fired per closed "extract" segment, no token-level hook, no other segment kind (connective/cite) triggers it.
- Streamed claim content/order must exactly match what the final (non-streaming) response would contain for the same query — same text-decode and chunk-ID-resolution logic as `gin.eval.claims.segments_to_raw_claims`, deliberately reimplemented at the `gin.corpus` layer (not imported) to preserve the layering constraint above; `SpanType.EXACT`/`SpanType.AMBIGUOUS` string values (`"EXACT"`/`"AMBIGUOUS"`, confirmed at `gin/eval/claims.py:36-37`) are hardcoded to match, not imported.

---

## Task 1: SEAR gets the `on_segment_closed` hook

**Files:**
- Modify: `sear/processor.py:29-116` (constructor), `sear/processor.py:506-534` (`_close_span`)
- Test: `tests/test_processor.py`

**Interfaces:**
- Produces: `ExtractiveCopyConstraint.__init__(..., on_segment_closed: Optional[Callable[[Segment], None]] = None)` — called with the just-appended `Segment` immediately after every span close. `Segment` (already defined at `sear/processor.py:20-25`) is unchanged: `token_ids: list[int]`, `sources: list[tuple[int,int,int]]`, `kind: str`, `guidance: str`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_processor.py` (uses the file's existing `_make_corpus`/`_VOCAB` fixtures, and mirrors the exact token-feeding pattern of `test_connective_between_spans_not_extract`, which already proves this `seq` closes a span on the trailing full-sequence call):

```python
def test_on_segment_closed_fires_on_span_close():
    corpus = _make_corpus()
    closed: list = []
    c = _make_constraint(corpus, on_segment_closed=closed.append)
    seq = [_VOCAB["the"], _VOCAB["fox"], _VOCAB["ran"], _VOCAB["|"]]
    flat = np.zeros(_V, dtype=np.float32)
    for i in range(len(seq)):
        c(np.array(seq[:i], dtype=np.intc), flat.copy())
    c(np.array(seq, dtype=np.intc), flat.copy())  # consumes "|", closing the span
    assert len(closed) == 1
    assert closed[0].kind == "extract"
    assert closed[0] is c.segments[0]


def test_on_segment_closed_not_called_when_unset():
    corpus = _make_corpus()
    c = _make_constraint(corpus)  # on_segment_closed defaults to None
    seq = [_VOCAB["the"], _VOCAB["fox"], _VOCAB["ran"], _VOCAB["|"]]
    flat = np.zeros(_V, dtype=np.float32)
    for i in range(len(seq)):
        c(np.array(seq[:i], dtype=np.intc), flat.copy())
    c(np.array(seq, dtype=np.intc), flat.copy())  # must not raise with no callback set
    assert len(c.segments) == 1
```

Also extend the file's `_make_constraint` helper (currently `tests/test_processor.py:30-55`) to accept and forward the new parameter — add `on_segment_closed=None` to its keyword-only parameters and pass it through to `ExtractiveCopyConstraint(...)`:

```python
def _make_constraint(
    corpus: Corpus,
    *,
    connective_starts: frozenset[int] | None = None,
    connective_phrases: dict[int, list[int]] | None = None,
    cite_ids: dict[int, int] | None = None,
    close_on_doc_divergence: bool = False,
    required_doc_groups: list[frozenset[int]] | None = None,
    on_segment_closed=None,
) -> ExtractiveCopyConstraint:
    starts = connective_starts if connective_starts is not None else frozenset({_VOCAB["but"]})
    phrases = connective_phrases if connective_phrases is not None else {_VOCAB["but"]: [_VOCAB["but"]]}
    cite = cite_ids or {}
    cite_sequences = {doc: [tok] for tok, doc in cite.items()}
    return ExtractiveCopyConstraint(
        corpus,
        prompt_len=0,
        eos_id=_VOCAB["<eos>"],
        delim_id=_VOCAB["|"],
        min_span_len=3,
        connective_starts=starts,
        connective_phrases=phrases,
        cite_ids=cite,
        cite_sequences_by_doc=cite_sequences,
        close_on_doc_divergence=close_on_doc_divergence,
        required_doc_groups=required_doc_groups,
        on_segment_closed=on_segment_closed,
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/test_processor.py::test_on_segment_closed_fires_on_span_close -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'on_segment_closed'`

- [ ] **Step 3: Add the parameter to the constructor**

Edit `sear/processor.py`. In the constructor signature (`:60-62`), change:

```python
        divergence_sentence_ends: Optional[dict[int, dict[int, int]]] = None,
        ranked_sentence_starts: Optional[list[tuple[int, int, float]]] = None,
    ):
```

to:

```python
        divergence_sentence_ends: Optional[dict[int, dict[int, int]]] = None,
        ranked_sentence_starts: Optional[list[tuple[int, int, float]]] = None,
        on_segment_closed: Optional[Callable[["Segment"], None]] = None,
    ):
```

In the constructor body (`:92`), change:

```python
        self.ranked_sentence_starts = ranked_sentence_starts or []
```

to:

```python
        self.ranked_sentence_starts = ranked_sentence_starts or []
        self.on_segment_closed = on_segment_closed
```

- [ ] **Step 4: Fire the callback in `_close_span`**

Edit `sear/processor.py`. In `_close_span` (`:506-534`), change:

```python
        self.segments.append(
            Segment(
                list(self._cur_tokens),
                sources,
                "extract",
                guidance=self._current_span_guidance,
            )
        )
        self._has_closed_extract = True
```

to:

```python
        self.segments.append(
            Segment(
                list(self._cur_tokens),
                sources,
                "extract",
                guidance=self._current_span_guidance,
            )
        )
        if self.on_segment_closed is not None:
            self.on_segment_closed(self.segments[-1])
        self._has_closed_extract = True
```

- [ ] **Step 5: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/test_processor.py -v`
Expected: all pass, including the two new tests

- [ ] **Step 6: Commit**

```bash
git add sear/processor.py tests/test_processor.py
git commit -m "SEAR: on_segment_closed hook fires per admitted claim (streaming trace, task 1)."
```

---

## Task 2: Corpus-tier trace primitives

**Files:**
- Create: `gin/corpus/trace_events.py`
- Modify: `gin/corpus/models.py` (add `doc_index_to_chunk_id` helper)
- Test: `tests/test_corpus_trace_events.py`

**Interfaces:**
- Produces: `RetrievalSettledTrace(synthesis_mode: str, manifest_hash: str, chunk_count: int)`, `ClaimClosedTrace(text: str, span_type: str, cited_chunk_ids: list[str])`, `current_trace_sink: ContextVar[Optional[Callable[[TraceEvent], None]]]` (default `None`); `doc_index_to_chunk_id(ctx: SynthesisContext) -> dict[int, str]`.
- Consumes: nothing new — `SynthesisContext` already exists at `gin/corpus/models.py:107`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_corpus_trace_events.py
"""Corpus-tier trace primitives: dependency-free event types + ContextVar sink."""
from gin.corpus.models import ChunkHit, SynthesisContext, doc_index_to_chunk_id
from gin.corpus.trace_events import (
    ClaimClosedTrace,
    RetrievalSettledTrace,
    current_trace_sink,
)
from uuid import uuid4

DOC = uuid4()


def _hit(chunk_id: str) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id, doc_id=DOC, text="text", head_sentence="head",
        eval_layer="counterfactual", eval_tag=None, content_hash="x",
        outlet="o", title="t", rrf_score=0.5,
    )


def test_doc_index_to_chunk_id_maps_from_context():
    ctx = SynthesisContext(
        doc_index_to_hit={0: _hit("a:0"), 1: _hit("b:0")},
        cite_index_to_doc={1: 0, 2: 1},
        mode="convergent",
    )
    assert doc_index_to_chunk_id(ctx) == {0: "a:0", 1: "b:0"}


def test_current_trace_sink_defaults_to_none():
    assert current_trace_sink.get() is None


def test_current_trace_sink_set_and_reset():
    received = []
    token = current_trace_sink.set(received.append)
    try:
        sink = current_trace_sink.get()
        assert sink is not None
        sink(RetrievalSettledTrace(synthesis_mode="convergent", manifest_hash="h", chunk_count=3))
        sink(ClaimClosedTrace(text="claim text", span_type="EXACT", cited_chunk_ids=["a:0"]))
    finally:
        current_trace_sink.reset(token)
    assert current_trace_sink.get() is None
    assert len(received) == 2
    assert isinstance(received[0], RetrievalSettledTrace)
    assert isinstance(received[1], ClaimClosedTrace)
    assert received[1].cited_chunk_ids == ["a:0"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_corpus_trace_events.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'gin.corpus.trace_events'` and `ImportError: cannot import name 'doc_index_to_chunk_id'`

- [ ] **Step 3: Write `gin/corpus/trace_events.py`**

```python
"""Ambient, request-scoped hook for streaming synthesis progress.

A ContextVar rather than a parameter threaded through decode_bundle's
callers, so instrumenting it for one streaming caller doesn't touch every
function signature between the HTTP layer and here. Deliberately
dependency-free — no gin.eval, no gin.federation imports — this module
sits at the gin.corpus layer; gin.federation (which already depends on
gin.corpus) translates these primitives into its own wire event types.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable, Optional, Union


@dataclass(frozen=True)
class RetrievalSettledTrace:
    synthesis_mode: str
    manifest_hash: str
    chunk_count: int


@dataclass(frozen=True)
class ClaimClosedTrace:
    text: str
    span_type: str
    cited_chunk_ids: list[str] = field(default_factory=list)


TraceEvent = Union[RetrievalSettledTrace, ClaimClosedTrace]

current_trace_sink: ContextVar[Optional[Callable[[TraceEvent], None]]] = ContextVar(
    "current_trace_sink", default=None
)
```

- [ ] **Step 4: Add `doc_index_to_chunk_id` to `gin/corpus/models.py`**

Read `gin/corpus/models.py` first to place this correctly — add it as a module-level function right after the `SynthesisContext` class definition (which holds `doc_index_to_hit: dict[int, ChunkHit]` at `:109`):

```python
def doc_index_to_chunk_id(ctx: SynthesisContext) -> dict[int, str]:
    return {i: hit.chunk_id for i, hit in ctx.doc_index_to_hit.items()}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_corpus_trace_events.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add gin/corpus/trace_events.py gin/corpus/models.py tests/test_corpus_trace_events.py
git commit -m "Corpus-tier trace primitives: dependency-free event types + ContextVar sink (streaming trace, task 2)."
```

---

## Task 3: Wire trace emission into `decode_bundle`

**Files:**
- Modify: `gin/corpus/generate.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `on_segment_closed` (Task 1); `current_trace_sink`, `RetrievalSettledTrace`, `ClaimClosedTrace`, `doc_index_to_chunk_id` (Task 2).
- Produces: no new public interface — `decode_bundle`'s signature is unchanged; this task only wires it internally to push events onto `current_trace_sink.get()` when one is set.

This task is the one place `"EXACT"`/`"AMBIGUOUS"` are hardcoded (matching `gin.eval.claims.SpanType.EXACT.value`/`.AMBIGUOUS.value` exactly, per Global Constraints — `gin.corpus` cannot import that enum without inverting the layering).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_generate.py`, reusing the file's existing `GreedyMaskDecoder`-based real-decode setup (the exact same corpus/hit/bundle/ctx as `test_convergent_numeric_sentence_closes_at_sentence_end`, which already proves this input produces real extracted text):

```python
def test_decode_bundle_emits_trace_events():
    from gin.corpus.trace_events import (
        ClaimClosedTrace,
        RetrievalSettledTrace,
        current_trace_sink,
    )

    llm = GreedyMaskDecoder()
    corpus = Corpus.from_texts({"anomaly": ANOMALY_TEXT}, tokenize=llm.tokenize)
    hit = ChunkHit(
        chunk_id="n1_doc_002:1", doc_id=DOC, text=ANOMALY_TEXT,
        head_sentence=ANOMALY_TEXT.split(",")[0] + ".",
        eval_layer="realism", eval_tag=None, content_hash="x",
        outlet="NOAA", title="2023 anomaly", rrf_score=0.9,
    )
    bundle = SynthesisBundle(hits=[hit], edges=[], mode="convergent", pairs=[])
    spans = corpus.sentence_starts
    preferred = {(0, pos) for (doc, pos) in spans if doc == 0}
    ctx = SynthesisContext(
        doc_index_to_hit={0: hit}, cite_index_to_doc={1: 0}, mode="convergent",
        preferred_starts=preferred,
        ranked_sentence_starts=[(0, pos, 1.0) for (doc, pos) in spans if doc == 0],
        top_doc_idx=0,
    )

    received: list = []
    token = current_trace_sink.set(received.append)
    try:
        result = decode_bundle(
            "2023 global surface temperature anomaly",
            corpus, ctx, bundle, llm,
            chat_template="plain", query_steered=True,
        )
    finally:
        current_trace_sink.reset(token)

    retrieval_events = [e for e in received if isinstance(e, RetrievalSettledTrace)]
    claim_events = [e for e in received if isinstance(e, ClaimClosedTrace)]
    assert len(retrieval_events) == 1
    assert retrieval_events[0].synthesis_mode == "convergent"
    assert retrieval_events[0].chunk_count == 1
    assert received.index(retrieval_events[0]) == 0  # retrieval event fires before any claim event
    assert len(claim_events) >= 1
    assert claim_events[0].cited_chunk_ids == ["n1_doc_002:1"]
    assert "2.12 degrees" in claim_events[0].text or any(
        "2.12 degrees" in e.text for e in claim_events
    )
    assert result.raw_text  # unchanged existing behavior still holds
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/test_generate.py::test_decode_bundle_emits_trace_events -v`
Expected: FAIL — zero events received (no wiring yet)

- [ ] **Step 3: Wire trace emission into `decode_bundle`**

Edit `gin/corpus/generate.py`. Add the import (alongside the existing `from sear.processor import ExtractiveCopyConstraint, Segment` line at `:16`):

```python
from sear.processor import ExtractiveCopyConstraint, Segment

from .models import SynthesisBundle, SynthesisContext, doc_index_to_chunk_id
from .trace_events import ClaimClosedTrace, RetrievalSettledTrace, current_trace_sink
```

(Note: `.models` already appears in the existing `from .models import SynthesisBundle, SynthesisContext` line at `:19` — merge `doc_index_to_chunk_id` into that same import rather than duplicating the line.)

In `decode_bundle` (`:158-294`), right before the `constraint = ExtractiveCopyConstraint(...)` call (`:235`), insert the retrieval-settled emission:

```python
    sink = current_trace_sink.get()
    if sink is not None:
        sink(RetrievalSettledTrace(
            synthesis_mode=bundle.mode,
            manifest_hash=retrieval_manifest.manifest_hash if retrieval_manifest else "",
            chunk_count=len(bundle.hits),
        ))

    def _on_segment_closed(seg: Segment) -> None:
        sink = current_trace_sink.get()
        if sink is None or seg.kind != "extract":
            return
        text = detok(seg.token_ids).strip()
        if not text:
            return
        chunk_map = doc_index_to_chunk_id(ctx)
        source_ids = [chunk_map[d] for (d, _s, _e) in seg.sources if d in chunk_map]
        # "EXACT"/"AMBIGUOUS" mirror gin.eval.claims.SpanType's values exactly
        # (gin/eval/claims.py:36-37) — hardcoded, not imported, per the
        # gin.corpus/gin.eval/gin.federation layering constraint.
        span_type = "AMBIGUOUS" if len(seg.sources) > 1 else "EXACT"
        sink(ClaimClosedTrace(text=text, span_type=span_type, cited_chunk_ids=source_ids))

    constraint = ExtractiveCopyConstraint(
        corpus=corpus,
        prompt_len=len(prompt_ids),
        eos_id=llm.token_eos(),
        delim_id=tokenize(b"|")[-1],
        min_span_len=params["min_span_len"],
        connective_starts=ctx.connective_starts,
        connective_continuations=ctx.connective_continuations,
        connective_phrases=ctx.connective_phrases,
        cite_ids=cite_ids,
        cite_sequences_by_doc=cite_sequences,
        cite_continuations=cite_cont,
        close_on_doc_divergence=divergent,
        required_doc_groups=ctx.required_doc_groups,
        focus_doc_indices=focus_docs,
        reject_ambiguous_spans=divergent,
        allow_shared_prefix=divergent or competing or not steered,
        span_must_start_at_sentence=divergent or competing,
        span_must_close_at_sentence_end=True,
        require_cite_after_extract=params["require_cites"],
        stop_when_groups_satisfied=params["stop_when_satisfied"],
        stop_after_first_extract=params["stop_after_first_extract"],
        block_eos_until_groups_satisfied=params["block_eos"],
        force_connective_ids=ctx.force_connective_ids,
        preferred_starts=preferred_starts,
        forbidden_starts=ctx.forbidden_starts if divergent else None,
        divergence_starts=ctx.divergence_starts if divergent else None,
        divergence_sentence_ends=ctx.divergence_sentence_ends if divergent else None,
        ranked_sentence_starts=ctx.ranked_sentence_starts if steered else None,
        require_divergence_after_first=divergent,
        on_segment_closed=_on_segment_closed,
    )
```

`_on_segment_closed` is defined unconditionally (not just when a sink is set) and is always passed to the constraint — it's a no-op read (`current_trace_sink.get()` returning `None`) when nothing is listening, so the non-streaming path (today's `/v1/federated/query`, and every existing test) pays only the cost of one `ContextVar.get()` call per closed claim, with no other behavior change.

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/test_generate.py -v`
Expected: all pass, including the new `test_decode_bundle_emits_trace_events`

- [ ] **Step 5: Run the full suite to confirm zero regression**

Run: `./venv/Scripts/python.exe -m pytest tests/ -q`
Expected: same pass count as before this task, plus the new tests — no existing test's behavior changes (the sink is `None` everywhere except the one new test that explicitly sets it)

- [ ] **Step 6: Commit**

```bash
git add gin/corpus/generate.py tests/test_generate.py
git commit -m "decode_bundle emits retrieval-settled and claim-closed trace events when a sink is set (streaming trace, task 3)."
```

---

## Task 4: Federation wire-layer trace events

**Files:**
- Create: `gin/federation/trace_events.py`
- Modify: `gin/federation/schema.py:19-21` (add `internal_error` to `RefusalReason`)
- Test: `tests/test_federation_trace_events.py`

**Interfaces:**
- Consumes: `WireClaim`, `FederatedResponse` (existing, `gin/federation/schema.py`).
- Produces: `RetrievalSettledEvent(event: Literal["retrieval_settled"], synthesis_mode: str, manifest_hash: str, chunk_count: int)`, `ClaimAdmittedEvent(event: Literal["claim_admitted"], claim: WireClaim)`, `SynthesisCompleteEvent(event: Literal["synthesis_complete"], response: FederatedResponse)`, `TraceEvent` union.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_federation_trace_events.py
"""Wire-protocol event shapes for the streaming reasoning-trace endpoint."""
import json

from gin.federation.schema import FederatedAnswer, FederatedResponse, WireClaim
from gin.federation.trace_events import (
    ClaimAdmittedEvent,
    RetrievalSettledEvent,
    SynthesisCompleteEvent,
)


def test_retrieval_settled_event_shape():
    e = RetrievalSettledEvent(synthesis_mode="convergent", manifest_hash="h", chunk_count=3)
    assert e.event == "retrieval_settled"
    data = json.loads(e.model_dump_json())
    assert data == {
        "event": "retrieval_settled", "synthesis_mode": "convergent",
        "manifest_hash": "h", "chunk_count": 3,
    }


def test_claim_admitted_event_wraps_wire_claim():
    claim = WireClaim(text="grounded claim", span_type="EXACT", cited_chunk_ids=["a:0"])
    e = ClaimAdmittedEvent(claim=claim)
    assert e.event == "claim_admitted"
    assert e.claim == claim


def test_synthesis_complete_event_wraps_federated_response():
    resp = FederatedResponse(
        answer=FederatedAnswer(request_id="r", node_id="node_a", answer_text="text")
    )
    e = SynthesisCompleteEvent(response=resp)
    assert e.event == "synthesis_complete"
    assert e.response.answer.node_id == "node_a"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_trace_events.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gin.federation.trace_events'`

- [ ] **Step 3: Write `gin/federation/trace_events.py`**

```python
"""Wire-protocol events for the streaming reasoning-trace endpoint
(POST /v1/federated/query/stream). Translates gin.corpus.trace_events'
primitive, dependency-free trace types into this layer's wire vocabulary —
the same translation boundary gin.federation.service.claims_to_wire already
draws between gin.eval's RawClaim and this module's WireClaim.
"""
from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel

from .schema import FederatedResponse, WireClaim


class RetrievalSettledEvent(BaseModel):
    event: Literal["retrieval_settled"] = "retrieval_settled"
    synthesis_mode: str
    manifest_hash: str
    chunk_count: int


class ClaimAdmittedEvent(BaseModel):
    event: Literal["claim_admitted"] = "claim_admitted"
    claim: WireClaim


class SynthesisCompleteEvent(BaseModel):
    event: Literal["synthesis_complete"] = "synthesis_complete"
    response: FederatedResponse


TraceEvent = Union[RetrievalSettledEvent, ClaimAdmittedEvent, SynthesisCompleteEvent]
```

- [ ] **Step 4: Add `internal_error` to `RefusalReason`**

Edit `gin/federation/schema.py:19-21`. Change:

```python
RefusalReason = Literal[
    "retrieval_floor", "zero_cursors", "hop_limit", "version_mismatch"
]
```

to:

```python
RefusalReason = Literal[
    "retrieval_floor", "zero_cursors", "hop_limit", "version_mismatch",
    "internal_error",
]
```

This is additive — every existing `NodeRefusal` construction still uses one of the original four values; `internal_error` is only ever used by the new streaming endpoint's error path (Task 5).

- [ ] **Step 5: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_federation_trace_events.py tests/test_federation_config.py -v`
Expected: all pass (`test_federation_config.py` included as a quick regression check on `schema.py`'s neighbors — no behavior there should change)

- [ ] **Step 6: Commit**

```bash
git add gin/federation/trace_events.py gin/federation/schema.py tests/test_federation_trace_events.py
git commit -m "Federation wire-layer trace event types + internal_error refusal reason (streaming trace, task 4)."
```

---

## Task 5: Streaming endpoint

**Files:**
- Modify: `gin/federation/server.py`
- Test: `tests/test_streaming_endpoint.py`

**Interfaces:**
- Consumes: `current_trace_sink`, `RetrievalSettledTrace`, `ClaimClosedTrace` (Task 2/3, via `gin.corpus.trace_events`); `RetrievalSettledEvent`, `ClaimAdmittedEvent`, `SynthesisCompleteEvent` (Task 4).
- Produces: `POST /v1/federated/query/stream`, NDJSON (`application/x-ndjson`), each line one `TraceEvent`, terminated by a `SynthesisCompleteEvent` line.

This task also extracts the existing `federated_query` endpoint's body into a shared helper `_answer_federated_query(fq) -> FederatedResponse` so both endpoints call identical logic — the non-streaming endpoint's behavior does not change (Global Constraint), it's now one line calling the extracted helper.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_streaming_endpoint.py
"""POST /v1/federated/query/stream: NDJSON trace events, terminal response
event, and full backward-compatible byte-for-byte parity with the
non-streaming endpoint's response shape for the same query."""
import json

from fastapi.testclient import TestClient

from gin.corpus.trace_events import ClaimClosedTrace, RetrievalSettledTrace, current_trace_sink
from gin.eval.arms import ArmOutput
from gin.eval.claims import RawClaim
from gin.federation.config import NodeConfig, PeerConfig
from gin.federation.schema import FederatedQuery, FederatedResponse
from gin.federation.server import create_app

CFG = NodeConfig(
    node_id="node_a", host="127.0.0.1", port=8471,
    database_url="postgresql://x/gin_node_a", cold_path="data/cold_node_a",
    model_path="", n_gpu_layers=0, n_ctx=4096,
    cert_path="a_cert.pem", key_path="a_key.pem", peer_timeout_s=5.0, peers=(),
)


def _grounded_with_events(q: str) -> ArmOutput:
    """Simulates what the real decode_bundle -> generate_no_continuation ->
    NoContinuationArm chain does: push trace events through the ambient
    sink while producing the final ArmOutput — same mechanism the real
    chain uses, just driven manually since this test injects a fake
    answer_fn rather than running a real model."""
    sink = current_trace_sink.get()
    if sink is not None:
        sink(RetrievalSettledTrace(synthesis_mode="convergent", manifest_hash="h", chunk_count=1))
        sink(ClaimClosedTrace(text="grounded claim", span_type="EXACT", cited_chunk_ids=["c:0"]))
    return ArmOutput(
        raw_text="grounded claim",
        claims=[RawClaim(text="grounded claim", span_type="EXACT", cited_chunk_ids=["c:0"])],
        retrieval_manifest_hash="h",
        synthesis_mode="convergent",
    )


def _refusing(q: str) -> ArmOutput:
    return ArmOutput(raw_text="[REFUSAL]", claims=[], retrieval_manifest_hash="",
                     refused=True, refusal_reason="retrieval_floor")


def _raising(q: str) -> ArmOutput:
    raise RuntimeError("simulated synthesis failure")


def _lines(response) -> list[dict]:
    return [json.loads(line) for line in response.text.strip().split("\n") if line]


def test_stream_emits_retrieval_and_claim_events_before_terminal():
    app = create_app(CFG, answer_fn=_grounded_with_events)
    client = TestClient(app)
    fq = FederatedQuery(query="q", origin_node="driver", hop_count=0)
    r = client.post("/v1/federated/query/stream", json=fq.model_dump())
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    events = _lines(r)
    assert [e["event"] for e in events] == [
        "retrieval_settled", "claim_admitted", "synthesis_complete",
    ]
    assert events[1]["claim"]["text"] == "grounded claim"
    assert events[1]["claim"]["cited_chunk_ids"] == ["c:0"]
    assert events[2]["response"]["answer"]["node_id"] == "node_a"


def test_stream_matches_non_streaming_response_for_same_query():
    app = create_app(CFG, answer_fn=_grounded_with_events)
    client = TestClient(app)
    fq_payload = FederatedQuery(query="q", origin_node="driver", hop_count=0).model_dump()

    stream_events = _lines(client.post("/v1/federated/query/stream", json=fq_payload))
    plain_resp = FederatedResponse.model_validate(
        client.post("/v1/federated/query", json=fq_payload).json()
    )

    terminal = stream_events[-1]
    assert terminal["event"] == "synthesis_complete"
    assert terminal["response"]["answer"]["answer_text"] == plain_resp.answer.answer_text
    assert terminal["response"]["answer"]["claims"] == [
        c.model_dump() for c in plain_resp.answer.claims
    ]


def test_instant_refusal_streams_zero_claim_events():
    app = create_app(CFG, answer_fn=_refusing)
    client = TestClient(app)
    fq = FederatedQuery(query="q", origin_node="driver", hop_count=0)
    r = client.post("/v1/federated/query/stream", json=fq.model_dump())
    events = _lines(r)
    assert [e["event"] for e in events] == ["synthesis_complete"]
    assert events[0]["response"]["refusal"]["reason"] == "retrieval_floor"


def test_synthesis_exception_yields_internal_error_terminal_event():
    app = create_app(CFG, answer_fn=_raising)
    client = TestClient(app)
    fq = FederatedQuery(query="q", origin_node="driver", hop_count=0)
    r = client.post("/v1/federated/query/stream", json=fq.model_dump())
    assert r.status_code == 200
    events = _lines(r)
    assert len(events) == 1
    assert events[0]["event"] == "synthesis_complete"
    assert events[0]["response"]["refusal"]["reason"] == "internal_error"


def test_non_streaming_endpoint_unchanged():
    app = create_app(CFG, answer_fn=_grounded_with_events)
    client = TestClient(app)
    fq = FederatedQuery(query="q", origin_node="driver", hop_count=0)
    r = client.post("/v1/federated/query", json=fq.model_dump())
    resp = FederatedResponse.model_validate(r.json())
    assert resp.answer.node_id == "node_a"
    assert resp.answer.answer_text == "grounded claim"
    assert resp.federation is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_streaming_endpoint.py -v`
Expected: FAIL — `404 Not Found` for `/v1/federated/query/stream` (route doesn't exist yet)

- [ ] **Step 3: Extract `_answer_federated_query` and add the streaming endpoint**

Edit `gin/federation/server.py`. Replace the file's entire import section (everything from the first `import asyncio` line through the `from .trust_gate import filter_trusted` line) with the following — it folds in `WireClaim` (new, into the existing `.schema` import), `queue` and `StreamingResponse` (new), and the two new `gin.corpus.trace_events`/`.trace_events` imports:

```python
import asyncio
import contextlib
import queue
import time
from contextlib import asynccontextmanager
from typing import Callable, Optional

from fastapi import FastAPI
from starlette.responses import StreamingResponse

from gin.corpus.relevance import query_keywords
from gin.corpus.trace_events import ClaimClosedTrace, RetrievalSettledTrace
from gin.corpus.trace_events import current_trace_sink as corpus_trace_sink

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
    WireClaim,
)
from .service import claims_to_wire
from .trace_events import ClaimAdmittedEvent, RetrievalSettledEvent, SynthesisCompleteEvent
from .trust_gate import filter_trusted
```

(`asyncio`, `contextlib`, `time`, `contextmanager`, `Callable`, `Optional`, `FastAPI` already exist at the top of the file — this replaces the existing `:16-22` import block wholesale with the additions folded in, rather than appending duplicate imports.)

Replace the existing `federated_query` endpoint (`:142-204`):

```python
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
```

with the extracted-helper version plus the new streaming endpoint:

```python
    def _answer_federated_query(fq: FederatedQuery) -> FederatedResponse:
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

    @app.post(
        "/v1/federated/query",
        response_model=FederatedResponse,
        response_model_exclude_none=True,
    )
    def federated_query(fq: FederatedQuery) -> FederatedResponse:
        return _answer_federated_query(fq)

    @app.post("/v1/federated/query/stream")
    async def federated_query_stream(fq: FederatedQuery) -> StreamingResponse:
        async def event_lines():
            q: "queue.Queue" = queue.Queue()

            def sink(trace) -> None:
                if isinstance(trace, RetrievalSettledTrace):
                    q.put(RetrievalSettledEvent(
                        synthesis_mode=trace.synthesis_mode,
                        manifest_hash=trace.manifest_hash,
                        chunk_count=trace.chunk_count,
                    ))
                elif isinstance(trace, ClaimClosedTrace):
                    q.put(ClaimAdmittedEvent(claim=WireClaim(
                        text=trace.text,
                        span_type=trace.span_type,
                        cited_chunk_ids=trace.cited_chunk_ids,
                    )))

            def run() -> FederatedResponse:
                token = corpus_trace_sink.set(sink)
                try:
                    return _answer_federated_query(fq)
                finally:
                    corpus_trace_sink.reset(token)

            task = asyncio.ensure_future(asyncio.to_thread(run))
            while not task.done():
                try:
                    event = q.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.02)
                    continue
                yield (event.model_dump_json() + "\n").encode("utf-8")
            while True:
                try:
                    event = q.get_nowait()
                except queue.Empty:
                    break
                yield (event.model_dump_json() + "\n").encode("utf-8")

            try:
                response = task.result()
            except Exception as exc:
                response = _refusal(fq, "internal_error", detail=str(exc))
            yield (SynthesisCompleteEvent(response=response).model_dump_json() + "\n").encode("utf-8")

        return StreamingResponse(event_lines(), media_type="application/x-ndjson")
```

Note on the polling loop: `queue.Queue.get_nowait()` + `asyncio.sleep(0.02)` on empty, never a blocking `queue.get(timeout=...)` — a blocking call would stall the whole event loop, not just this request. A 20ms poll is a deliberately simple choice (Global Constraint: no new dependencies) — not a tight latency guarantee, adequate for a research node, not a production SLA.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_streaming_endpoint.py -v`
Expected: 5 passed

- [ ] **Step 5: Run the full suite to confirm the non-streaming endpoint truly didn't change**

Run: `./venv/Scripts/python.exe -m pytest tests/ -q`
Expected: every existing test still passes unmodified — `tests/test_federation_server.py`, `tests/test_trust_gate_wiring.py`, `tests/test_federation_loop.py`, `tests/test_peer_selection_loop.py`, and every other file that exercises `/v1/federated/query` in particular

- [ ] **Step 6: Commit**

```bash
git add gin/federation/server.py tests/test_streaming_endpoint.py
git commit -m "POST /v1/federated/query/stream: NDJSON trace events, non-streaming endpoint unchanged (streaming trace, task 5)."
```

---

## Task 6: Manual live verification

**Files:** none — verification only, no code changes.

No automated test — this codebase has never run a real model inside the pytest suite (confirmed across every federation test file); the mTLS work's live-eval gate (`docs/superpowers/plans/2026-07-16-federation-mtls.md` Task 7) set the precedent for a manual verification step instead.

- [ ] **Step 1: Start one node with a real model**

```bash
./venv/Scripts/python.exe scripts/node_serve.py --config config/node_a.yaml
```

(Or any single-node config already set up from the mTLS work's live-eval run — `config/node_a.yaml`, `node_b.yaml`, or `node_c.yaml`.)

- [ ] **Step 2: Call the streaming endpoint with a real query, timing event arrival**

```bash
./venv/Scripts/python.exe -c "
import ssl, time, httpx, json
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
ctx.check_hostname = False
ctx.load_verify_locations(cafile='certs/node_a/cert.pem')
ctx.load_cert_chain(certfile='certs/node_b/cert.pem', keyfile='certs/node_b/key.pem')
started = time.monotonic()
with httpx.Client(verify=ctx, timeout=120.0) as client:
    with client.stream('POST', 'https://127.0.0.1:8471/v1/federated/query/stream',
                        json={'query': 'what caused the 2023 anomaly', 'origin_node': 'driver'}) as r:
        for line in r.iter_lines():
            if not line:
                continue
            evt = json.loads(line)
            print(f'{time.monotonic()-started:6.2f}s  {evt[\"event\"]}')
"
```

Expected: a `retrieval_settled` line, then one or more `claim_admitted` lines, each printed with a measurably earlier timestamp than the final `synthesis_complete` line — proving events arrive incrementally during decode, not batched at the end, for a query whose total wall-clock time is long enough to matter (consistent with the ~50s median measured in `data/eval_runs/*/meta.json`).

- [ ] **Step 3: Record the result**

No commit for this task — note the observed timings in the PR description or session notes when this plan is executed, per the falsifiable claim's "first `claim_admitted` event observably arrives before total request completion" bar.

---

## Task 7: Documentation updates

**Files:**
- Modify: `architecture.md`
- Modify: `README.md`
- Modify: `docs/GIN_Node_Architecture_v1.md`

- [ ] **Step 1: Update `architecture.md`'s Phase 3 checklist**

Locate the `🔲 gRPC/QUIC wire` line (added during the mTLS sub-project, per `docs/superpowers/plans/2026-07-16-federation-mtls.md` Task 8). Replace it with:

```
✅ Streaming reasoning trace: NDJSON incremental claim events over the
  existing mTLS stack — reframes (does not build) the gRPC/QUIC line;
  see docs/superpowers/specs/2026-07-16-streaming-reasoning-trace-design.md
  for why gRPC/QUIC itself remains deferred.
```

- [ ] **Step 2: Add a "streaming reasoning trace" subsection to `README.md`**

Alongside the existing peer-authentication subsection, document: the three event shapes (`retrieval_settled`, `claim_admitted`, `synthesis_complete`), an example `httpx` streaming client loop (reuse the Task 6 verification script's shape), and that `/v1/federated/query` (non-streaming) is unchanged and remains the endpoint peer-to-peer delegation uses internally.

- [ ] **Step 3: Add a v1 implementation note to `docs/GIN_Node_Architecture_v1.md`**

At the "Protocol: gRPC over QUIC" bullet (`:119`), matching the existing peer-selection/trust-weights/mTLS notes' pattern: trace streaming shipped as NDJSON on the existing HTTP/mTLS stack; gRPC/QUIC remains the deferred institutional-deployment target for the transport itself; note gRPC's actual mature transport is HTTP/2, and QUIC support in gRPC's Python ecosystem is still immature, so "gRPC over QUIC" isn't yet an off-the-shelf combination.

- [ ] **Step 4: Commit**

```bash
git add architecture.md README.md docs/GIN_Node_Architecture_v1.md
git commit -m "Docs: streaming reasoning trace shipped (Phase 3, sub-project 6)."
```

---

## Self-Review Notes

**Spec coverage:** every scope decision in the spec maps to a task — SEAR hook (Task 1), dependency-free corpus-tier primitives (Task 2), wiring (Task 3), wire-layer translation (Task 4), the endpoint itself + non-streaming-endpoint-unchanged guarantee (Task 5), real-model verification (Task 6), docs (Task 7).

**Layering constraint, concretely enforced:** Task 2/3 never import from `gin.eval` or `gin.federation` — verified against the actual current state of `gin/corpus/generate.py`'s imports (confirmed empty of both before this plan). Task 4/5 do the wire-type translation, consistent with `gin.federation.service.claims_to_wire`'s existing `RawClaim` → `WireClaim` translation pattern at a different layer boundary.

**Type/interface consistency:** `RetrievalSettledTrace`/`ClaimClosedTrace` (Task 2, `gin.corpus`) and `RetrievalSettledEvent`/`ClaimAdmittedEvent` (Task 4, `gin.federation`) are deliberately distinct types at the layer boundary — Task 5's `sink` closure is the one place they're translated into each other, matching the design's stated translation boundary. `on_segment_closed`'s callback signature (`Callable[[Segment], None]`, Task 1) is never used directly outside `gin/corpus/generate.py` (Task 3) — `Segment` itself never crosses into `gin.federation`.

**Placeholder scan:** no TBD/TODO; every step shows complete, runnable code grounded in the actual current file contents (not approximated) — `gin/corpus/generate.py`'s constraint-construction block, `gin/federation/server.py`'s full current endpoint bodies, and `sear/processor.py`'s exact constructor/`_close_span` code were all read in full before this plan was written.
