"""ArmOutput carries the structured grounding-failure signal federation routes on."""
from gin.eval.arms import ArmOutput, _refusal_output


def test_refusal_output_default_reason():
    out = _refusal_output()
    assert out.refused is True
    assert out.refusal_reason == "zero_cursors"
    assert out.synthesis_mode == ""


def test_refusal_output_explicit_reason():
    out = _refusal_output(reason="retrieval_floor")
    assert out.refusal_reason == "retrieval_floor"


def test_success_output_defaults():
    out = ArmOutput(raw_text="x", claims=[], retrieval_manifest_hash="h")
    assert out.refusal_reason == ""
    assert out.synthesis_mode == ""
