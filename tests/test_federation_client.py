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
