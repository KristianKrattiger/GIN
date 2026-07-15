"""FastAPI app factory for one federation node.

Guards run in order: bearer auth (401) -> protocol version (typed refusal) ->
hop limit (typed refusal). hop_count >= 1 requests are answered locally and
NEVER re-delegated — that, plus the router only running at hop 0, is the
entire loop-prevention story. answer_fn / peer_client / corpus_fingerprint
are injected so tests run without a model, a database, or a network.
"""
from __future__ import annotations

import asyncio
import contextlib
import hmac
import time
from contextlib import asynccontextmanager
from typing import Callable, Optional

from fastapi import Depends, FastAPI, Header, HTTPException

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
        by_id = {p.node_id: p for p in config.peers}
        return [by_id[nid] for nid in order]

    sync_stats = AnchorSyncStats(
        node_id=config.node_id,
        peer_node_id=config.peers[0].node_id if config.peers else "",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = None
        if peer_anchor_store is not None and peer_client is not None and config.peers:
            task = asyncio.create_task(
                run_forever(
                    config.peers[0], peer_client, peer_anchor_store,
                    config.anchor_sync_interval_s, sync_stats,
                    summary_store=peer_summary_store,
                )
            )
        yield
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title=f"GIN federation node {config.node_id}", lifespan=lifespan)

    def _check_auth(authorization: str = Header(default="")) -> None:
        expected = f"Bearer {config.shared_secret}"
        if not hmac.compare_digest(authorization, expected):
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
    def anchors_root(_: None = Depends(_check_auth)) -> AnchorRootResponse:
        rows = anchor_rows_fn()
        return AnchorRootResponse(
            node_id=config.node_id,
            root_hash=root_hash(all_bucket_hashes(rows)),
            leaf_count=len(rows),
        )

    @app.get("/v1/federated/anchors/buckets", response_model=AnchorBucketsResponse)
    def anchors_buckets(_: None = Depends(_check_auth)) -> AnchorBucketsResponse:
        rows = anchor_rows_fn()
        return AnchorBucketsResponse(node_id=config.node_id, bucket_hashes=all_bucket_hashes(rows))

    @app.get(
        "/v1/federated/anchors/bucket/{index}", response_model=AnchorLeavesResponse
    )
    def anchors_bucket(index: int, _: None = Depends(_check_auth)) -> AnchorLeavesResponse:
        rows = anchor_rows_fn()
        buckets = build_buckets(rows)
        return AnchorLeavesResponse(
            node_id=config.node_id, bucket_index=index, leaves=buckets.get(index, [])
        )

    @app.get("/v1/federated/anchors/sync_stats", response_model=AnchorSyncStats)
    def anchors_sync_stats(_: None = Depends(_check_auth)) -> AnchorSyncStats:
        return sync_stats

    @app.get("/v1/federated/summary", response_model=PeerSummaryResponse)
    def federated_summary(_: None = Depends(_check_auth)) -> PeerSummaryResponse:
        return summary_fn()

    return app
