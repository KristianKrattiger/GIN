"""Selection metrics: precision@1 and avg peers tried over routed queries."""
from gin.federation.selection_eval import SelectionOutcome, compute_selection_metrics


def _routed(id, cls, source, attempted, verified=True):
    return SelectionOutcome(id=id, federation_class=cls, refused=False, routed=True,
                            source_node=source, peers_attempted=attempted,
                            attribution_verified=verified)


def test_precision_at_1_all_correct():
    outcomes = [
        _routed("b1", "b_only", "node_b", ["node_b"]),
        _routed("c1", "c_only", "node_c", ["node_c"]),
    ]
    m = compute_selection_metrics(outcomes)
    assert m["selection_precision_at_1"] == 1.0
    assert m["avg_peers_tried"] == 1.0
    assert m["routed_fabrication_rate"] == 0.0


def test_precision_at_1_penalizes_wrong_first_pick():
    outcomes = [
        _routed("b1", "b_only", "node_b", ["node_b"]),
        _routed("c1", "c_only", "node_c", ["node_b", "node_c"]),  # wrong first
    ]
    m = compute_selection_metrics(outcomes)
    assert m["selection_precision_at_1"] == 0.5
    assert m["avg_peers_tried"] == 1.5


def test_routing_false_positive_and_honest_refusal():
    outcomes = [
        SelectionOutcome(id="a1", federation_class="a_answerable", refused=False,
                         routed=False, source_node="node_a"),
        SelectionOutcome(id="n1", federation_class="neither", refused=True,
                         routed=True, peers_attempted=["node_b", "node_c"]),
    ]
    m = compute_selection_metrics(outcomes)
    assert m["routing_false_positives"] == 0
    assert m["honest_refusal_rate"] == 1.0
