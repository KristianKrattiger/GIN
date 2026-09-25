"""Tests for gin.assay_proposer.runtime's pure parts (loading a GGUF is the smoke script's job)."""
from types import SimpleNamespace

import pytest

from gin.assay_proposer.runtime import SEAR_MODEL_PREFIX, make_render, model_id_for


def test_the_model_id_is_the_gguf_stem_under_the_sear_prefix():
    assert SEAR_MODEL_PREFIX == "sear/"
    assert model_id_for("models/Qwen2.5-7B-Instruct-Q6_K.gguf") == "sear/Qwen2.5-7B-Instruct-Q6_K"
    assert model_id_for(r"C:\models\Mistral-7B-Instruct-v0.3-Q6_K.gguf") == "sear/Mistral-7B-Instruct-v0.3-Q6_K"


class _AcceptingFormatter:
    """Stands in for a template that is fine with a leading system message."""

    def __init__(self):
        self.calls: list[list[dict]] = []

    def __call__(self, *, messages, **_):
        self.calls.append(messages)
        return SimpleNamespace(prompt=" | ".join(f"{m['role']}:{m['content']}" for m in messages))


class _AlternatingOnlyFormatter:
    """Stands in for Mistral v0.3's own template, which rejects a leading
    system message with the jinja2-rendered ValueError from raise_exception()."""

    def __init__(self):
        self.calls: list[list[dict]] = []

    def __call__(self, *, messages, **_):
        self.calls.append(messages)
        if any(m["role"] == "system" for m in messages):
            raise ValueError("Conversation roles must alternate user/assistant/user/assistant/...")
        return SimpleNamespace(prompt=" | ".join(f"{m['role']}:{m['content']}" for m in messages))


def test_make_render_uses_the_system_role_when_the_template_accepts_it():
    formatter = _AcceptingFormatter()
    render = make_render(formatter)
    assert render("SYS", "USER") == "system:SYS | user:USER"
    assert len(formatter.calls) == 1


def test_make_render_folds_system_into_user_when_the_template_rejects_it():
    formatter = _AlternatingOnlyFormatter()
    render = make_render(formatter)
    assert render("SYS", "USER") == "user:SYS\n\nUSER"
    assert len(formatter.calls) == 2
    assert formatter.calls[0] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
    assert formatter.calls[1] == [{"role": "user", "content": "SYS\n\nUSER"}]


def test_make_render_does_not_swallow_an_unrelated_exception():
    class _Broken:
        def __call__(self, *, messages, **_):
            raise RuntimeError("template engine exploded")

    render = make_render(_Broken())
    with pytest.raises(RuntimeError, match="template engine exploded"):
        render("SYS", "USER")
