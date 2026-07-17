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
            # exclude_none matches the non-streaming endpoint's
            # response_model_exclude_none=True (see the /v1/federated/query
            # route above) so the terminal event's `response` payload has
            # the same shape here as there for the same query.
            yield (
                SynthesisCompleteEvent(response=response).model_dump_json(exclude_none=True)
                + "\n"
            ).encode("utf-8")

        return StreamingResponse(event_lines(), media_type="application/x-ndjson")

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
