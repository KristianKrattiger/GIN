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
