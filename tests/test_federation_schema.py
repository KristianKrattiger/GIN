"""Wire schema round-trips and envelope invariants."""
import pytest
from pydantic import ValidationError

from gin.federation.schema import (
    PROTOCOL_VERSION,
    FederatedAnswer,
    FederatedQuery,
    FederatedResponse,
    FederationLayer,
    NodeRefusal,
    WireClaim,
    new_request_id,
)


def test_query_round_trip():
    fq = FederatedQuery(query="q", origin_node="node_a", hop_count=1)
    assert fq.protocol_version == PROTOCOL_VERSION
    assert fq.request_id  # auto-generated
    again = FederatedQuery.model_validate(fq.model_dump())
    assert again == fq


def test_answer_round_trip_with_claims():
    ans = FederatedAnswer(
        request_id=new_request_id(),
        node_id="node_b",
        answer_text="Indigenous-led resistance efforts",
        claims=[
            WireClaim(
                text="Indigenous-led resistance efforts",
                span_type="EXACT",
                cited_chunk_ids=["n2_doc_001:4"],
            )
        ],
        corpus_fingerprint={"n_chunks": 46},
        synthesis_mode="convergent",
        timing_s=1.5,
    )
    again = FederatedAnswer.model_validate(ans.model_dump())
    assert again.claims[0].cited_chunk_ids == ["n2_doc_001:4"]


def test_refusal_reason_enum_enforced():
    with pytest.raises(ValidationError):
        NodeRefusal(
            request_id="r", node_id="node_a", reason="not_a_reason"
        )


def test_refusal_carries_peer_reasons():
    ref = NodeRefusal(
        request_id="r",
        node_id="node_a",
        reason="retrieval_floor",
        peer_reasons={"node_b": "zero_cursors"},
    )
    assert ref.peer_reasons["node_b"] == "zero_cursors"


def test_response_exactly_one_of_answer_refusal():
    ref = NodeRefusal(request_id="r", node_id="node_a", reason="hop_limit")
    ok = FederatedResponse(refusal=ref)
    assert ok.answer is None
    with pytest.raises(ValidationError):
        FederatedResponse()  # neither
    ans = FederatedAnswer(
        request_id="r", node_id="node_b", answer_text="x",
        claims=[], corpus_fingerprint={},
    )
    with pytest.raises(ValidationError):
        FederatedResponse(answer=ans, refusal=ref)  # both


def test_federation_layer_defaults():
    layer = FederationLayer(answered_by="node_b", hop_count=1, request_id="r")
    assert layer.transport == "http"


from gin.federation.schema import (
    AnchorBucketsResponse,
    AnchorLeaf,
    AnchorLeavesResponse,
    AnchorRootResponse,
    AnchorSyncStats,
)


def test_anchor_root_response_round_trip():
    resp = AnchorRootResponse(node_id="node_a", root_hash="abc123", leaf_count=55)
    again = AnchorRootResponse.model_validate(resp.model_dump())
    assert again == resp
    assert again.protocol_version == PROTOCOL_VERSION


def test_anchor_buckets_response_round_trip():
    resp = AnchorBucketsResponse(node_id="node_a", bucket_hashes=["h"] * 16)
    again = AnchorBucketsResponse.model_validate(resp.model_dump())
    assert len(again.bucket_hashes) == 16


def test_anchor_leaves_response_round_trip():
    resp = AnchorLeavesResponse(
        node_id="node_b",
        bucket_index=3,
        leaves=[
            AnchorLeaf(
                chunk_id="n2_doc_001:0", content_hash="h1",
                outlet="node_2_grassroots", title="WE ACT",
            )
        ],
    )
    again = AnchorLeavesResponse.model_validate(resp.model_dump())
    assert again.leaves[0].chunk_id == "n2_doc_001:0"


def test_anchor_sync_stats_defaults():
    stats = AnchorSyncStats(node_id="node_a", peer_node_id="node_b")
    assert stats.cycles_run == 0
    assert stats.last_root_matched is False
    assert stats.last_cycle_buckets_synced == 0
    assert stats.last_cycle_bytes == 0


from gin.federation.schema import PeerSummaryResponse


def test_peer_summary_response_round_trip():
    resp = PeerSummaryResponse(
        node_id="node_c",
        embedding_centroid=[0.1, 0.2, 0.3],
        distinctive_terms={"inflation": 2.1, "reserve": 1.8},
        domains=["monetary_policy"],
    )
    again = PeerSummaryResponse.model_validate(resp.model_dump())
    assert again == resp
    assert again.protocol_version == PROTOCOL_VERSION


def test_peer_summary_defaults_empty_collections():
    resp = PeerSummaryResponse(node_id="node_c")
    assert resp.embedding_centroid == []
    assert resp.distinctive_terms == {}
    assert resp.domains == []


def test_peer_summary_domains_round_trips_multiple():
    resp = PeerSummaryResponse(
        node_id="node_a", domains=["environmental_measurement", "monetary_policy"],
    )
    again = PeerSummaryResponse.model_validate(resp.model_dump())
    assert again.domains == ["environmental_measurement", "monetary_policy"]


def test_federation_layer_peers_attempted_defaults_empty():
    layer = FederationLayer(answered_by="node_b", hop_count=1, request_id="r")
    assert layer.peers_attempted == []


def test_federation_layer_carries_peers_attempted():
    layer = FederationLayer(
        answered_by="node_c", hop_count=1, request_id="r",
        peers_attempted=["node_b", "node_c"],
    )
    again = FederationLayer.model_validate(layer.model_dump())
    assert again.peers_attempted == ["node_b", "node_c"]
