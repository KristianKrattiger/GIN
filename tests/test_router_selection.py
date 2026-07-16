"""Multi-peer ranked delegation: try peers in ranker order, fall back on
refusal, record the full attempt order — never exceeding hop_count=1."""
from gin.eval.arms import ArmOutput
from gin.federation.config import NodeConfig, PeerConfig
from gin.federation.router import answer_or_delegate
from gin.federation.schema import FederatedAnswer, NodeRefusal, WireClaim

PEER_B = PeerConfig(node_id="node_b", url="http://b")
PEER_C = PeerConfig(node_id="node_c", url="http://c")


def _cfg():
    return NodeConfig(
        node_id="node_a", host="127.0.0.1", port=8471,
        database_url="postgresql://x/a", cold_path="data/cold_a",
        model_path="", n_gpu_layers=0, n_ctx=4096,
        cert_path="a_cert.pem", key_path="a_key.pem",
        peer_timeout_s=5.0, peers=(PEER_B, PEER_C),
    )


def _refuse_local(q):
    return ArmOutput(raw_text="[REFUSAL]", claims=[], retrieval_manifest_hash="",
                     refused=True, refusal_reason="retrieval_floor")


class ScriptedPeer:
    """Answers for `answerer` node_id; refuses for everyone else. Records calls."""

    def __init__(self, answerer):
        self.answerer = answerer
        self.calls = []

    def query(self, peer, fq):
        self.calls.append(peer.node_id)
        if peer.node_id == self.answerer:
            return FederatedAnswer(
                request_id=fq.request_id, node_id=peer.node_id,
                answer_text="grounded", claims=[WireClaim(
                    text="grounded", span_type="EXACT", cited_chunk_ids=["c:0"])],
                corpus_fingerprint={"n": 1}, synthesis_mode="convergent",
            )
        return NodeRefusal(request_id=fq.request_id, node_id=peer.node_id,
                           reason="zero_cursors")


def test_ranker_order_tried_first_success_no_fallback():
    peer_client = ScriptedPeer(answerer="node_c")
    # ranker puts C first (correct); B never contacted
    result = answer_or_delegate(
        "q", config=_cfg(), answer_fn=_refuse_local, peer_client=peer_client,
        peer_ranker=lambda q: [PEER_C, PEER_B],
    )
    assert not result.refused
    assert result.source_node == "node_c"
    assert result.peers_attempted == ["node_c"]
    assert peer_client.calls == ["node_c"]
    assert result.federation.peers_attempted == ["node_c"]


def test_falls_back_to_next_peer_on_refusal():
    peer_client = ScriptedPeer(answerer="node_c")
    # ranker wrongly puts B first; B refuses, C answers
    result = answer_or_delegate(
        "q", config=_cfg(), answer_fn=_refuse_local, peer_client=peer_client,
        peer_ranker=lambda q: [PEER_B, PEER_C],
    )
    assert not result.refused
    assert result.source_node == "node_c"
    assert result.peers_attempted == ["node_b", "node_c"]


def test_all_peers_refuse_aggregates_reasons():
    peer_client = ScriptedPeer(answerer="none")
    result = answer_or_delegate(
        "q", config=_cfg(), answer_fn=_refuse_local, peer_client=peer_client,
        peer_ranker=lambda q: [PEER_B, PEER_C],
    )
    assert result.refused
    assert result.refusal_reasons["node_a"] == "retrieval_floor"
    assert result.refusal_reasons["node_b"] == "zero_cursors"
    assert result.refusal_reasons["node_c"] == "zero_cursors"
    assert result.peers_attempted == ["node_b", "node_c"]


def test_default_ranker_is_config_order():
    peer_client = ScriptedPeer(answerer="node_b")
    result = answer_or_delegate(
        "q", config=_cfg(), answer_fn=_refuse_local, peer_client=peer_client,
    )  # no peer_ranker -> config order [B, C]
    assert result.source_node == "node_b"
    assert result.peers_attempted == ["node_b"]
