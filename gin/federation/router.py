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
from .config import NodeConfig, PeerConfig
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
    peers_attempted: list[str] = field(default_factory=list)
    request_id: str = ""


def answer_or_delegate(
    query: str,
    *,
    config: NodeConfig,
    answer_fn: AnswerFn,
    peer_client: PeerClient,
    request_id: Optional[str] = None,
    peer_ranker: Optional[Callable[[str], list[PeerConfig]]] = None,
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

    peers_to_try = peer_ranker(query) if peer_ranker is not None else list(config.peers)
    attempted: list[str] = []
    for peer in peers_to_try:
        attempted.append(peer.node_id)
        fq = FederatedQuery(
            request_id=rid, query=query, origin_node=config.node_id, hop_count=1
        )
        try:
            outcome = peer_client.query(peer, fq)
        except PeerUnreachable:
            reasons[peer.node_id] = "unreachable"
            continue
        if isinstance(outcome, NodeRefusal):
            reasons[outcome.node_id] = outcome.reason
            continue
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
                peers_attempted=list(attempted),
            ),
            peers_attempted=list(attempted),
            request_id=rid,
        )

    return RoutedResult(
        refused=True, source_node=config.node_id,
        refusal_reasons=reasons, peers_attempted=list(attempted), request_id=rid,
    )
