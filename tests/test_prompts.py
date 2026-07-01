"""Tests for synthesis prompts."""
from uuid import uuid4

from gin.corpus.models import ChunkHit, EdgeRecord, SynthesisBundle
from gin.corpus.prompts import build_source_manifest, build_synthesis_prompt


def _hit(chunk_id: str, outlet: str, title: str, text: str) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        doc_id=uuid4(),
        text=text,
        head_sentence=text[:40],
        eval_layer="realism",
        eval_tag=None,
        content_hash="abc",
        outlet=outlet,
        title=title,
    )


def test_build_source_manifest_lists_sources():
    hits = [
        _hit("a:0", "CentralWire", "Incident", "Downtown incident at Main."),
        _hit("b:0", "MetroDaily", "Incident", "Downtown incident at Main."),
    ]
    manifest = build_source_manifest(hits)
    assert "[1] CentralWire" in manifest
    assert "[2] MetroDaily" in manifest
    assert "chunk a:0" in manifest


def test_build_synthesis_prompt_divergent_mode():
    left = _hit("a:0", "CentralWire", "Incident", "text a")
    right = _hit("b:0", "MetroDaily", "Incident", "text b")
    edge = EdgeRecord("a:0", "b:0", "contradicts", note="counts diverge")
    bundle = SynthesisBundle(
        hits=[left, right],
        edges=[edge],
        mode="divergent",
        pairs=[(left, right, edge)],
    )
    prompt = build_synthesis_prompt("incident details", bundle, chat_template="plain")
    assert "contradicts" in prompt
    assert "without resolving the conflict" in prompt
    assert "Do not paraphrase" in prompt
    assert "source marker" in prompt


def test_build_synthesis_prompt_mistral_template():
    hit = _hit("a:0", "CentralWire", "T", "text")
    bundle = SynthesisBundle(hits=[hit], edges=[], mode="convergent")
    prompt = build_synthesis_prompt("query", bundle, chat_template="mistral")
    assert prompt.startswith("[INST]")
    assert prompt.endswith("[/INST]")
