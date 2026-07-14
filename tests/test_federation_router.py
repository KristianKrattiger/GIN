"""Router: delegates exactly when the local path refuses, and only then."""
from gin.eval.arms import ArmOutput
from gin.eval.claims import RawClaim
from gin.federation.client import PeerUnreachable
from gin.federation.config import NodeConfig, PeerConfig
from gin.federation.router import answer_or_delegate
from gin.federation.schema import FederatedAnswer, NodeRefusal, WireClaim

PEER = PeerConfig(node_id="node_b", url="http://peer-b")
CFG = NodeConfig(
    node_id="node_a", host="127.0.0.1", port=8471,
    database_url="postgresql://x/gin_node_a", cold_path="data/cold_node_a",
    model_path="", n_gpu_layers=0, n_ctx=4096,
    shared_secret="s", peer_timeout_s=5.0, peers=(PEER,),
)


def _grounded() -> ArmOutput:
    return ArmOutput(
        raw_text="local answer",
        claims=[RawClaim(text="local answer", span_type="EXACT",
                         cited_chunk_ids=["n1_doc_002:1"])],
        retrieval_manifest_hash="h",
        synthesis_mode="convergent",
    )


def _refusing(reason: str) -> ArmOutput:
    return ArmOutput(
        raw_text="[REFUSAL]", claims=[], retrieval_manifest_hash="",
        refused=True, refusal_reason=reason,
    )


class SpyPeer:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def query(self, peer, fq):
        self.calls.append(fq)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def test_local_success_never_calls_peer():
    peer = SpyPeer(None)
    out = answer_or_delegate(
        "q", config=CFG, answer_fn=lambda q: _grounded(), peer_client=peer
    )
    assert out.refused is False
    assert out.source_node == "node_a"
    assert out.federation is None
    assert peer.calls == []
    assert out.claims[0].cited_chunk_ids == ["n1_doc_002:1"]


def test_local_refusal_delegates_with_hop_one():
    answer = FederatedAnswer(
        request_id="r", node_id="node_b", answer_text="peer answer",
        claims=[WireClaim(text="peer answer", span_type="EXACT",
                          cited_chunk_ids=["n2_doc_002:3"])],
        corpus_fingerprint={"n_chunks": 46}, synthesis_mode="convergent",
    )
    peer = SpyPeer(answer)
    out = answer_or_delegate(
        "q", config=CFG, answer_fn=lambda q: _refusing("retrieval_floor"),
        peer_client=peer,
    )
    assert out.refused is False
    assert out.source_node == "node_b"
    assert out.federation is not None
    assert out.federation.answered_by == "node_b"
    assert out.federation.hop_count == 1
    assert out.corpus_fingerprint == {"n_chunks": 46}
    assert len(peer.calls) == 1
    assert peer.calls[0].hop_count == 1
    assert peer.calls[0].origin_node == "node_a"


def test_both_refuse_aggregates_reasons():
    refusal = NodeRefusal(request_id="r", node_id="node_b", reason="zero_cursors")
    peer = SpyPeer(refusal)
    out = answer_or_delegate(
        "q", config=CFG, answer_fn=lambda q: _refusing("retrieval_floor"),
        peer_client=peer,
    )
    assert out.refused is True
    assert out.refusal_reasons == {
        "node_a": "retrieval_floor", "node_b": "zero_cursors"
    }


def test_peer_unreachable_is_honest_refusal():
    peer = SpyPeer(PeerUnreachable(PEER, ConnectionError("down")))
    out = answer_or_delegate(
        "q", config=CFG, answer_fn=lambda q: _refusing("zero_cursors"),
        peer_client=peer,
    )
    assert out.refused is True
    assert out.refusal_reasons == {
        "node_a": "zero_cursors", "node_b": "unreachable"
    }


def test_no_peers_refuses_locally():
    cfg = NodeConfig(**{**CFG.__dict__, "peers": ()})
    out = answer_or_delegate(
        "q", config=cfg, answer_fn=lambda q: _refusing("retrieval_floor"),
        peer_client=SpyPeer(None),
    )
    assert out.refused is True
    assert out.refusal_reasons == {"node_a": "retrieval_floor"}
