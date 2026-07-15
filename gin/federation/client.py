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
from .schema import (
    AnchorBucketsResponse,
    AnchorLeavesResponse,
    AnchorRootResponse,
    FederatedAnswer,
    FederatedQuery,
    FederatedResponse,
    NodeRefusal,
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

    def get_anchor_root(self, peer: PeerConfig) -> AnchorRootResponse:
        return self._get(peer, "/v1/federated/anchors/root", AnchorRootResponse)

    def get_anchor_buckets(self, peer: PeerConfig) -> AnchorBucketsResponse:
        return self._get(peer, "/v1/federated/anchors/buckets", AnchorBucketsResponse)

    def get_anchor_bucket(self, peer: PeerConfig, index: int) -> AnchorLeavesResponse:
        return self._get(peer, f"/v1/federated/anchors/bucket/{index}", AnchorLeavesResponse)

    def _get(self, peer: PeerConfig, path: str, model_cls):
        try:
            with httpx.Client(
                transport=self._transport, timeout=self._timeout
            ) as client:
                r = client.get(f"{peer.url}{path}", headers=self._headers)
                r.raise_for_status()
        except httpx.HTTPError as exc:
            raise PeerUnreachable(peer, exc) from exc
        return model_cls.model_validate(r.json())
