"""Tests for metric aggregation over verified claim records."""
import pytest

from gin.eval.claims import ClaimRecord, NodeScope, SpanType, Verdict
from gin.eval.metrics import (
    aggregate,
    aggregate_by_layer,
    aggregate_retrieval_recall,
    chunk_quotation_rate,
    counterfactual_adherence,
    cross_node_integrity,
    divergence_fidelity,
    fabrication_rate,
    failing_query_relevance_ids,
    failure_state,
    gold_chunk_coverage,
    gold_chunk_coverage_for_query,
    query_passes_relevance,
    query_relevance_rate,
    retrieval_recall_at_k,
    supported_irrelevance_rate,
)
from gin.eval.metrics import QueryResult


def _supported(text, chunk, span=SpanType.EXACT.value, scope=NodeScope.WITHIN_NODE.value):
    return ClaimRecord(
        claim_text=text,
        verdict=Verdict.SUPPORTED.value,
        matched_chunk_id=chunk,
        score=0.9,
        node_scope=scope,
        span_type=span,
        cited_chunk_ids=[chunk],
    )


def _unsupported(text):
    return ClaimRecord(
        claim_text=text,
        verdict=Verdict.UNSUPPORTED.value,
        matched_chunk_id=None,
        score=0.1,
        node_scope=NodeScope.NONE.value,
        span_type=SpanType.GENERATED.value,
        cited_chunk_ids=[],
    )


def _refusal():
    return ClaimRecord(
        claim_text="The sources do not support an answer.",
        verdict=Verdict.REFUSAL.value,
        matched_chunk_id=None,
        score=0.0,
        node_scope=NodeScope.NONE.value,
        span_type=SpanType.REFUSAL.value,
        cited_chunk_ids=[],
    )


@pytest.fixture
def results():
    return [
        QueryResult(
            query_id="q1", query="a", arm="x", eval_layer="realism",
            expectation="answerable", refused=False,
            claims=[_supported("142 people", "c0"), _supported("23 arrests", "c0")],
        ),
        QueryResult(
            query_id="q2", query="b", arm="x", eval_layer="realism",
            expectation="answerable", refused=False,
            claims=[_unsupported("invented fact")],
        ),
        QueryResult(
            query_id="q3", query="c", arm="x", eval_layer="counterfactual",
            expectation="counterfactual", refused=False,
            counterfactual_answer="3.7 percent",
            claims=[_supported("the rate was 3.7 percent", "c1", span=SpanType.GENERATED.value)],
        ),
        QueryResult(
            query_id="q4", query="d", arm="x", eval_layer="out_of_scope",
            expectation="out_of_scope", refused=True,
            claims=[_refusal()],
        ),
        QueryResult(
            query_id="q5", query="e", arm="x", eval_layer="out_of_scope",
            expectation="out_of_scope", refused=False,
            claims=[_unsupported("fabricated answer")],
        ),
    ]


def test_fabrication_rate(results):
    # 2 unsupported out of 5 graded (refusal excluded).
    assert fabrication_rate(results) == pytest.approx(0.4)


def test_counterfactual_adherence(results):
    assert counterfactual_adherence(results) == pytest.approx(1.0)


def test_failure_state(results):
    precision, recall, confusion = failure_state(results)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
    assert confusion == {"tp": 1, "fp": 0, "fn": 1, "tn": 3}


def test_cross_node_integrity(results):
    within_ratio, violations = cross_node_integrity(results)
    assert within_ratio == pytest.approx(0.6)  # 3 within-node of 5 graded
    assert violations == 0


def test_aggregate_rollup(results):
    m = aggregate("x", results)
    assert m.n_queries == 5
    assert m.n_claims == 5
    assert m.fabrication_rate == pytest.approx(0.4)
    assert m.grounded_precision == pytest.approx(0.6)
    assert m.attribution_coverage == pytest.approx(0.6)


def test_aggregate_by_layer(results):
    by_layer = aggregate_by_layer("x", results)
    assert set(by_layer) == {"realism", "counterfactual", "out_of_scope"}
    assert by_layer["realism"].n_queries == 2


def test_empty_metrics_are_none():
    assert fabrication_rate([]) is None
    within_ratio, violations = cross_node_integrity([])
    assert within_ratio is None and violations == 0


def test_retrieval_recall_at_k():
    assert retrieval_recall_at_k(["a", "b"], ["a", "c"]) == pytest.approx(0.5)
    assert retrieval_recall_at_k([], ["a"]) is None
    assert retrieval_recall_at_k(["a"], ["a", "b"]) == pytest.approx(1.0)


def test_aggregate_retrieval_recall():
    rows = [
        QueryResult(
            query_id="q1",
            query="a",
            arm="x",
            eval_layer="realism",
            expectation="answerable",
            refused=False,
            claims=[],
            retrieval_recall_at_k=1.0,
        ),
        QueryResult(
            query_id="q2",
            query="b",
            arm="x",
            eval_layer="realism",
            expectation="answerable",
            refused=False,
            claims=[],
            retrieval_recall_at_k=0.5,
        ),
    ]
    assert aggregate_retrieval_recall(rows) == pytest.approx(0.75)


def _incident_claim(text: str, chunk: str) -> ClaimRecord:
    return _supported(text, chunk)


def test_transit_ridership_probe_fails_query_relevance():
    """Mirrors overlap run 192827Z: NC emits incident spans for transit query."""
    row = QueryResult(
        query_id="transit_ridership",
        query="What was the average daily ridership in the first week of the north line extension?",
        arm="no_continuation",
        eval_layer="realism",
        expectation="answerable",
        refused=False,
        claims=[
            _incident_claim(
                "Emergency services confirmed 142 people received treatment at area hospitals.",
                "incident_centralwire:0",
            ),
            _incident_claim(
                "Emergency services confirmed 98 people received treatment at area hospitals.",
                "incident_metrodaily:0",
            ),
            _incident_claim(
                "Police said 11 arrests were made before midnight.",
                "incident_regionalpost:0",
            ),
        ],
        retrieved_chunk_ids=[
            "incident_regionalpost:0",
            "incident_centralwire:0",
            "incident_metrodaily:0",
            "transit_authority_update:0",
            "labor_bureau_report:0",
        ],
        gold_chunk_ids=["transit_authority_update:0"],
        retrieval_recall_at_k=1.0,
    )
    assert not query_passes_relevance(row)
    assert gold_chunk_coverage_for_query(row) == pytest.approx(0.0)
    assert supported_irrelevance_rate([row]) == pytest.approx(1.0)


def test_weather_winds_probe_fails_query_relevance():
    """Mirrors 192827Z: supported incident/election spans, not weather gold."""
    row = QueryResult(
        query_id="weather_winds",
        query="What sustained wind speed is expected from the coastal storm system?",
        arm="no_continuation",
        eval_layer="realism",
        expectation="answerable",
        refused=False,
        claims=[
            _incident_claim(
                "Emergency services confirmed 142 people received treatment at area hospitals.",
                "incident_centralwire:0",
            ),
            _incident_claim(
                "Turnout reached 61 percent of registered voters.",
                "election_centralwire:0",
            ),
        ],
        retrieved_chunk_ids=[
            "incident_centralwire:0",
            "incident_metrodaily:0",
            "incident_regionalpost:0",
            "election_centralwire:0",
            "election_metrodaily:0",
            "weather_service_brief:0",
        ],
        gold_chunk_ids=["weather_service_brief:0"],
        retrieval_recall_at_k=1.0,
    )
    assert not query_passes_relevance(row)
    assert gold_chunk_coverage_for_query(row) == pytest.approx(0.0)


def test_unemployment_rate_zero_gold_coverage():
    """Mirrors 192827Z: gold labor chunks retrieved but incident spans cited."""
    row = QueryResult(
        query_id="unemployment_rate",
        query="What was the regional unemployment rate in the latest monthly survey?",
        arm="no_continuation",
        eval_layer="counterfactual",
        expectation="counterfactual",
        refused=False,
        counterfactual_answer="3.7 percent",
        claims=[
            _incident_claim(
                "Emergency services confirmed 142 people received treatment at area hospitals.",
                "incident_centralwire:0",
            ),
        ],
        gold_chunk_ids=["labor_bureau_report:0", "labor_independent_survey:0"],
        retrieval_recall_at_k=1.0,
    )
    assert gold_chunk_coverage_for_query(row) == pytest.approx(0.0)


def test_incident_divergence_fidelity_partial():
    row = QueryResult(
        query_id="incident_hospital",
        query="How many people received hospital treatment after the downtown incident?",
        arm="no_continuation",
        eval_layer="realism",
        eval_tag="incident_divergence",
        expectation="answerable",
        refused=False,
        claims=[
            _incident_claim(
                "Emergency services confirmed 142 people received treatment at area hospitals.",
                "incident_centralwire:0",
            ),
            _incident_claim(
                "Emergency services confirmed 98 people received treatment at area hospitals.",
                "incident_metrodaily:0",
            ),
        ],
        contradicts_pairs=[
            ["incident_centralwire:0", "incident_metrodaily:0"],
            ["incident_centralwire:0", "incident_regionalpost:0"],
        ],
    )
    assert divergence_fidelity([row]) == pytest.approx(0.5)


def test_chunk_quotation_rate():
    row = QueryResult(
        query_id="transit_ridership",
        query="ridership",
        arm="no_continuation",
        eval_layer="realism",
        expectation="answerable",
        refused=False,
        claims=[
            _incident_claim("incident text", "incident_centralwire:0"),
            _incident_claim("other incident", "incident_metrodaily:0"),
        ],
        retrieved_chunk_ids=[
            "incident_centralwire:0",
            "incident_metrodaily:0",
            "transit_authority_update:0",
        ],
    )
    assert chunk_quotation_rate([row]) == pytest.approx(2 / 3)


def test_aggregate_includes_epistemic_metrics(results):
    m = aggregate("x", results)
    assert m.query_relevance_rate is not None
    assert m.gold_chunk_coverage is None  # fixture rows lack gold_chunk_ids
    assert m.chunk_quotation_rate is None


def test_failing_query_relevance_ids(results):
    failed = failing_query_relevance_ids(results)
    assert "q5" in failed  # out_of_scope not refused
