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
