"""Runner tests using a fake arm — no llama.cpp, no Postgres, no NLI model."""
import json

from gin.eval.arms import ArmOutput, FlaggedGenerationArm
from gin.eval.claims import NodeScope, RawClaim, SpanType, Verdict
from gin.eval.queryset import EvalQuery
from gin.eval.runner import make_meta, run_experiment, verify_output, write_run
from gin.eval.verifier import Verifier


class FakeArm:
    """Returns a canned ArmOutput per query text."""

    name = "fake"

    def __init__(self, outputs: dict[str, ArmOutput]):
        self.outputs = outputs

    def run(self, query: str, llm) -> ArmOutput:
        return self.outputs[query]


QUERIES = [
    EvalQuery(
        id="incident",
        query="How many people received treatment?",
        eval_layer="realism",
        expectation="answerable",
    ),
    EvalQuery(
        id="mars",
        query="Who won the championship?",
        eval_layer="out_of_scope",
        expectation="out_of_scope",
    ),
]


def _fake_outputs():
    chunks = [("c0", "Emergency services confirmed 142 people received treatment")]
    return {
        "How many people received treatment?": ArmOutput(
            raw_text="142 people received treatment [1]",
            claims=[
                RawClaim(
                    text="142 people received treatment",
                    span_type=SpanType.EXACT.value,
                    cited_chunk_ids=["c0"],
                )
            ],
            retrieval_manifest_hash="deadbeef",
            refused=False,
            node_of={"c0": "CentralWire"},
            chunks=chunks,
        ),
        "Who won the championship?": ArmOutput(
            raw_text="",
            claims=[],
            retrieval_manifest_hash="",
            refused=True,
        ),
    }


def test_run_experiment_scores_and_refuses():
    verifier = Verifier(mode="overlap", threshold=0.5)
    arms = {"fake": FakeArm(_fake_outputs())}
    results = run_experiment(QUERIES, arms, llm=None, verifier=verifier)

    rows = results["fake"]
    assert len(rows) == 2

    answerable = next(r for r in rows if r.query_id == "incident")
    assert not answerable.refused
    assert len(answerable.claims) == 1
    claim = answerable.claims[0]
    assert claim.verdict == Verdict.SUPPORTED.value
    assert claim.matched_chunk_id == "c0"
    assert claim.node_scope == NodeScope.WITHIN_NODE.value

    refused = next(r for r in rows if r.query_id == "mars")
    assert refused.refused
    assert refused.claims[0].verdict == Verdict.REFUSAL.value


def test_flagged_generation_arm_records_error():
    verifier = Verifier(mode="overlap", threshold=0.5)
    arms = {"flagged_generation": FlaggedGenerationArm()}
    results = run_experiment(QUERIES[:1], arms, llm=None, verifier=verifier)
    row = results["flagged_generation"][0]
    assert row.error is not None
    assert row.claims == []


def test_verify_output_exact_span_bypasses_low_nli_score():
    chunks = [
        (
            "c0",
            "Emergency services confirmed 142 people received treatment at area hospitals.",
        )
    ]
    output = ArmOutput(
        raw_text="Emergency services confirmed 142 people received treatment at area hospitals.",
        claims=[
            RawClaim(
                text="Emergency services confirmed 142 people received treatment at area hospitals.",
                span_type=SpanType.EXACT.value,
                cited_chunk_ids=["c0"],
            )
        ],
        retrieval_manifest_hash="abc",
        refused=False,
        node_of={"c0": "CentralWire"},
        chunks=chunks,
    )
    verifier = Verifier(
        mode="nli",
        threshold=0.5,
        scorer=lambda claim, chunk: 0.01,
    )
    records = verify_output(output, verifier)
    assert records[0].verdict == Verdict.SUPPORTED.value
    assert records[0].score == 1.0


def test_write_run_produces_report_and_json(tmp_path):
    verifier = Verifier(mode="overlap", threshold=0.5)
    arms = {"fake": FakeArm(_fake_outputs())}
    results = run_experiment(QUERIES, arms, llm=None, verifier=verifier)

    meta = make_meta(
        model="fake.gguf",
        verifier_mode="overlap",
        threshold=0.5,
        queryset="data/eval/queryset.yaml",
        arms=["fake"],
        n_queries=len(QUERIES),
    )
    run_dir = write_run(results, meta, tmp_path)

    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "SEAR vs RAG Evaluation Report" in report
    assert "fabrication_rate" in report
    assert "Epistemic quality" in report
    assert "query_relevance_rate" in report
    assert "fake" in report

    per_arm = json.loads((run_dir / "results" / "fake.json").read_text(encoding="utf-8"))
    assert len(per_arm) == 2
    assert per_arm[0]["claims"][0]["verdict"] in {
        Verdict.SUPPORTED.value,
        Verdict.REFUSAL.value,
    }

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "fake" in metrics
    assert "overall" in metrics["fake"]
    assert (run_dir / "meta.json").exists()


def test_regression_overlap_eval_with_fake_arm(tmp_path):
    """CI-style path: overlap verifier, fake arm, regression-sized query list."""
    from gin.eval.metrics import query_relevance_rate
    from gin.eval.runner import format_report

    verifier = Verifier(mode="overlap", threshold=0.5)
    arms = {"fake": FakeArm(_fake_outputs())}
    results = run_experiment(QUERIES, arms, llm=None, verifier=verifier)
    meta = make_meta(
        model="fake.gguf",
        verifier_mode="overlap",
        threshold=0.5,
        queryset="data/eval/queryset.yaml",
        arms=["fake"],
        n_queries=len(QUERIES),
    )
    report = format_report(results, meta)
    assert "query_relevance_rate" in report
    assert query_relevance_rate(results["fake"]) == 1.0
