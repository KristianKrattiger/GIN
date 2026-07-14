"""Federation bar metrics and claim verification, no DB required."""
import pytest

from gin.federation.eval import (
    QueryOutcome,
    claims_verify,
    compute_metrics,
    load_federation_queryset,
)
from gin.federation.schema import WireClaim


def test_load_queryset_validates_class(tmp_path):
    p = tmp_path / "qs.yaml"
    p.write_text(
        "queries:\n"
        "  - id: ok\n    query: q\n    federation_class: b_only\n"
        "    gold_chunk_ids:\n      - n2_doc_002:3\n",
        encoding="utf-8",
    )
    qs = load_federation_queryset(p)
    assert qs[0].federation_class == "b_only"
    assert qs[0].gold_chunk_ids == ("n2_doc_002:3",)

    p.write_text(
        "queries:\n  - id: bad\n    query: q\n    federation_class: nope\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_federation_queryset(p)


def test_claims_verify_substring_normalized():
    chunk = {"n2_doc_002:3": "LANDBACK is described as an organizing  and\nnarrative framework."}
    fetch = chunk.get
    good = [WireClaim(text="LANDBACK is described as an organizing and narrative framework.",
                      span_type="EXACT", cited_chunk_ids=["n2_doc_002:3"])]
    assert claims_verify(good, fetch) is True
    fabricated = [WireClaim(text="LANDBACK was invented in 2031",
                            span_type="EXACT", cited_chunk_ids=["n2_doc_002:3"])]
    assert claims_verify(fabricated, fetch) is False
    missing_chunk = [WireClaim(text="anything", span_type="EXACT",
                               cited_chunk_ids=["nope:0"])]
    assert claims_verify(missing_chunk, fetch) is False
    assert claims_verify([], fetch) is False  # no claims = nothing verified
    uncited = [WireClaim(text="anything", span_type="EXACT", cited_chunk_ids=[])]
    assert claims_verify(uncited, fetch) is False


def test_compute_metrics_perfect_run():
    outcomes = [
        QueryOutcome(id="a1", federation_class="a_answerable", refused=False,
                     routed=False, source_node="node_a"),
        QueryOutcome(id="b1", federation_class="b_only", refused=False,
                     routed=True, source_node="node_b", attribution_verified=True),
        QueryOutcome(id="n1", federation_class="neither", refused=True,
                     routed=True, refusal_reasons={"node_a": "zero_cursors",
                                                   "node_b": "zero_cursors"}),
    ]
    m = compute_metrics(outcomes)
    assert m["routing_false_positives"] == 0
    assert m["routing_recall"] == 1.0
    assert m["routed_answer_attribution_verified"] == 1.0
    assert m["routed_fabrication_rate"] == 0.0
    assert m["honest_refusal_rate"] == 1.0
    assert m["a_answered_locally"] == 1


def test_compute_metrics_failures_visible():
    outcomes = [
        QueryOutcome(id="a1", federation_class="a_answerable", refused=False,
                     routed=True, source_node="node_b"),   # false positive
        QueryOutcome(id="b1", federation_class="b_only", refused=False,
                     routed=True, source_node="node_b",
                     attribution_verified=False),           # fabrication
        QueryOutcome(id="b2", federation_class="b_only", refused=True,
                     routed=False),                          # missed routing
        QueryOutcome(id="n1", federation_class="neither", refused=False,
                     routed=False, source_node="node_a"),    # dishonest answer
    ]
    m = compute_metrics(outcomes)
    assert m["routing_false_positives"] == 1
    assert m["routing_recall"] == 0.5
    assert m["routed_fabrication_rate"] == 1.0
    assert m["honest_refusal_rate"] == 0.0
