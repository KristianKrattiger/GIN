"""The service seam converts arm output to wire claims losslessly."""
from gin.eval.arms import ArmOutput
from gin.eval.claims import RawClaim
from gin.federation.service import claims_to_wire


def test_claims_to_wire_preserves_fields():
    out = ArmOutput(
        raw_text="a | b",
        claims=[
            RawClaim(text="a", span_type="EXACT", cited_chunk_ids=["n2_doc_001:4"]),
            RawClaim(text="b", span_type="AMBIGUOUS", cited_chunk_ids=["x:0", "y:1"]),
        ],
        retrieval_manifest_hash="h",
    )
    wire = claims_to_wire(out)
    assert [w.text for w in wire] == ["a", "b"]
    assert wire[0].span_type == "EXACT"
    assert wire[1].cited_chunk_ids == ["x:0", "y:1"]


def test_claims_to_wire_empty():
    out = ArmOutput(raw_text="", claims=[], retrieval_manifest_hash="")
    assert claims_to_wire(out) == []
