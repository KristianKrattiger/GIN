"""Tests for gin.assay_proposer.runtime's pure parts (loading a GGUF is the smoke script's job)."""
from gin.assay_proposer.runtime import SEAR_MODEL_PREFIX, model_id_for


def test_the_model_id_is_the_gguf_stem_under_the_sear_prefix():
    assert SEAR_MODEL_PREFIX == "sear/"
    assert model_id_for("models/Qwen2.5-7B-Instruct-Q6_K.gguf") == "sear/Qwen2.5-7B-Instruct-Q6_K"
    assert model_id_for(r"C:\models\Mistral-7B-Instruct-v0.3-Q6_K.gguf") == "sear/Mistral-7B-Instruct-v0.3-Q6_K"
