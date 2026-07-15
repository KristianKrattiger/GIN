"""Peer ranking: dense + sparse RRF fusion, no-summary peers sort last,
deterministic and independent of input order."""
from gin.federation.peer_selection import cosine, dense_rank, rank_peers, sparse_rank
from gin.federation.schema import PeerSummaryResponse


def _summary(node_id, centroid, terms):
    return PeerSummaryResponse(
        node_id=node_id, embedding_centroid=centroid, distinctive_terms=terms
    )


def test_cosine_orthogonal_and_parallel():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([1.0, 0.0], [0.0, 0.0]) == 0.0  # zero vector safe


def test_dense_rank_orders_by_cosine_desc():
    order = dense_rank(
        [1.0, 0.0],
        {"b": [0.0, 1.0], "c": [0.9, 0.1]},
    )
    assert order == ["c", "b"]


def test_sparse_rank_orders_by_matched_idf_mass_desc():
    order = sparse_rank(
        {"inflation", "reserve"},
        {"b": {"landback": 3.0}, "c": {"inflation": 2.0, "reserve": 1.5}},
    )
    assert order == ["c", "b"]


def test_rank_peers_agreeing_signals():
    summaries = {
        "node_b": _summary("node_b", [0.0, 1.0], {"landback": 3.0, "indigenous": 2.5}),
        "node_c": _summary("node_c", [1.0, 0.0], {"inflation": 2.0, "reserve": 1.8}),
    }
    order = rank_peers(
        [1.0, 0.0], {"inflation", "reserve"}, summaries, ["node_b", "node_c"]
    )
    assert order[0] == "node_c"


def test_rank_peers_no_summary_sorts_last_in_config_order():
    summaries = {
        "node_c": _summary("node_c", [1.0, 0.0], {"inflation": 2.0}),
    }
    # node_b has no summary; must appear after node_c, never dropped.
    order = rank_peers(
        [1.0, 0.0], {"inflation"}, summaries, ["node_b", "node_c"]
    )
    assert order == ["node_c", "node_b"]


def test_rank_peers_empty_summaries_is_config_order():
    order = rank_peers([1.0, 0.0], {"x"}, {}, ["node_b", "node_c"])
    assert order == ["node_b", "node_c"]


def test_rank_peers_deterministic_under_input_reorder():
    summaries = {
        "node_b": _summary("node_b", [0.2, 0.9], {"justice": 2.0}),
        "node_c": _summary("node_c", [0.9, 0.2], {"inflation": 2.0}),
    }
    a = rank_peers([0.9, 0.2], {"inflation"}, summaries, ["node_b", "node_c"])
    b = rank_peers([0.9, 0.2], {"inflation"}, dict(reversed(list(summaries.items()))), ["node_b", "node_c"])
    assert a == b
