"""HttpPeerClient: parsing, mTLS identity, and failure mapping via MockTransport.

httpx still constructs a real ssl.SSLContext (and loads the cert/key files
from disk) even when a MockTransport intercepts the connection, so these
tests need real — if throwaway — cert fixtures, not arbitrary path strings.
"""
import httpx
import pytest

from gin.federation.certs import generate_self_signed_cert
from gin.federation.client import HttpPeerClient, PeerUnreachable
from gin.federation.config import PeerConfig
from gin.federation.schema import (
    AnchorBucketsResponse,
    AnchorLeaf,
    AnchorLeavesResponse,
    AnchorRootResponse,
    FederatedAnswer,
    FederatedQuery,
    FederatedResponse,
    NodeRefusal,
    PeerSummaryResponse,
)


@pytest.fixture
def own_identity(tmp_path):
    cert_path, key_path = generate_self_signed_cert("node_a", tmp_path)
    return str(cert_path), str(key_path)


@pytest.fixture
def peer(tmp_path):
    cert_path, _ = generate_self_signed_cert("node_b", tmp_path)
    return PeerConfig(node_id="node_b", url="http://peer-b", pinned_cert_path=str(cert_path))


def _fq() -> FederatedQuery:
    return FederatedQuery(query="q", origin_node="node_a", hop_count=1)


def test_returns_parsed_answer_and_sends_no_bearer_header(own_identity, peer):
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

    client = HttpPeerClient(*own_identity, transport=httpx.MockTransport(handler))
    out = client.query(peer, _fq())
    assert isinstance(out, FederatedAnswer)
    assert out.node_id == "node_b"
    assert seen["auth"] is None  # no bearer header — mTLS is the identity now
    assert seen["url"] == "http://peer-b/v1/federated/query"


def test_returns_parsed_refusal(own_identity, peer):
    def handler(request: httpx.Request) -> httpx.Response:
        body = FederatedResponse(
            refusal=NodeRefusal(
                request_id="r", node_id="node_b", reason="zero_cursors"
            )
        )
        return httpx.Response(200, json=body.model_dump())

    client = HttpPeerClient(*own_identity, transport=httpx.MockTransport(handler))
    out = client.query(peer, _fq())
    assert isinstance(out, NodeRefusal)
    assert out.reason == "zero_cursors"


def test_http_error_maps_to_peer_unreachable(own_identity, peer):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    client = HttpPeerClient(*own_identity, transport=httpx.MockTransport(handler))
    with pytest.raises(PeerUnreachable) as exc:
        client.query(peer, _fq())
    assert exc.value.peer.node_id == "node_b"


def test_connect_error_maps_to_peer_unreachable(own_identity, peer):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = HttpPeerClient(*own_identity, transport=httpx.MockTransport(handler))
    with pytest.raises(PeerUnreachable):
        client.query(peer, _fq())


def test_remote_protocol_error_maps_to_peer_unreachable(own_identity, peer):
    """The real-world shape of a rejected mTLS handshake: httpx surfaces it
    as RemoteProtocolError (subclass of HTTPError), not a connect-time error
    — verified against a real uvicorn+httpx stack during design. The existing
    except httpx.HTTPError must already cover this with no new code."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("Server disconnected without sending a response.")

    client = HttpPeerClient(*own_identity, transport=httpx.MockTransport(handler))
    with pytest.raises(PeerUnreachable):
        client.query(peer, _fq())


def test_get_anchor_root_parses_and_sends_no_bearer(own_identity, peer):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        body = AnchorRootResponse(node_id="node_b", root_hash="abc", leaf_count=50)
        return httpx.Response(200, json=body.model_dump())

    client = HttpPeerClient(*own_identity, transport=httpx.MockTransport(handler))
    out = client.get_anchor_root(peer)
    assert out.root_hash == "abc"
    assert seen["url"] == "http://peer-b/v1/federated/anchors/root"
    assert seen["auth"] is None


def test_get_anchor_buckets_parses(own_identity, peer):
    def handler(request: httpx.Request) -> httpx.Response:
        body = AnchorBucketsResponse(node_id="node_b", bucket_hashes=["h"] * 16)
        return httpx.Response(200, json=body.model_dump())

    client = HttpPeerClient(*own_identity, transport=httpx.MockTransport(handler))
    out = client.get_anchor_buckets(peer)
    assert len(out.bucket_hashes) == 16


def test_get_anchor_bucket_hits_indexed_path(own_identity, peer):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        body = AnchorLeavesResponse(
            node_id="node_b", bucket_index=7,
            leaves=[AnchorLeaf(chunk_id="c", content_hash="h", outlet="o", title="t")],
        )
        return httpx.Response(200, json=body.model_dump())

    client = HttpPeerClient(*own_identity, transport=httpx.MockTransport(handler))
    out = client.get_anchor_bucket(peer, 7)
    assert seen["url"] == "http://peer-b/v1/federated/anchors/bucket/7"
    assert out.leaves[0].chunk_id == "c"


def test_anchor_endpoint_http_error_maps_to_peer_unreachable(own_identity, peer):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    client = HttpPeerClient(*own_identity, transport=httpx.MockTransport(handler))
    with pytest.raises(PeerUnreachable):
        client.get_anchor_root(peer)


def test_get_summary_parses_and_hits_summary_path(own_identity, peer):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        body = PeerSummaryResponse(
            node_id="node_c", embedding_centroid=[0.1, 0.2],
            distinctive_terms={"inflation": 2.0},
        )
        return httpx.Response(200, json=body.model_dump())

    client = HttpPeerClient(*own_identity, transport=httpx.MockTransport(handler))
    out = client.get_summary(peer)
    assert out.node_id == "node_c"
    assert out.distinctive_terms == {"inflation": 2.0}
    assert seen["url"] == "http://peer-b/v1/federated/summary"
    assert seen["auth"] is None


def test_get_summary_http_error_maps_to_peer_unreachable(own_identity, peer):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = HttpPeerClient(*own_identity, transport=httpx.MockTransport(handler))
    with pytest.raises(PeerUnreachable):
        client.get_summary(peer)
