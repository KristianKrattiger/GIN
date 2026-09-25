"""
gin.assay_proposer.runtime
--------------------------
Load a GGUF for the proposer: the model, a renderer for its own chat
template, and the id Receipts stamps on its manifest.
"""
from __future__ import annotations

from pathlib import PureWindowsPath
from typing import Any, Callable

SEAR_MODEL_PREFIX = "sear/"


def model_id_for(model_path: str) -> str:
    # PureWindowsPath splits on both / and \, so the id is the same on either OS.
    return f"{SEAR_MODEL_PREFIX}{PureWindowsPath(model_path).stem}"


def make_render(formatter: Any) -> Callable[[str, str], str]:
    """Render one (system, user) turn with a model's own chat template.

    Some templates -- Mistral v0.3's own included -- require roles to strictly
    alternate starting with "user" and reject a leading system message outright,
    raising ValueError("Conversation roles must alternate user/assistant/...")
    from the template's raise_exception() (see llama_cpp.llama_chat_format's
    Jinja2ChatFormatter, which runs the template through jinja2's sandboxed
    Environment.render()). Fall back to folding the system prompt into the
    single user turn those templates do accept.
    """

    def render(system: str, user: str) -> str:
        try:
            return formatter(messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]).prompt
        except ValueError:
            return formatter(messages=[
                {"role": "user", "content": f"{system}\n\n{user}"},
            ]).prompt

    return render


def load_model(
    model_path: str,
    *,
    n_ctx: int,
    n_gpu_layers: int,
) -> tuple[Any, Callable[[str, str], str], str]:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
    template = llm.metadata.get("tokenizer.chat_template")
    if not template:
        raise RuntimeError(f"{model_path} carries no tokenizer.chat_template; the proposer cannot render its prompt")

    def piece(token: int) -> str:
        return llm.detokenize([token]).decode("utf-8", errors="ignore") if token >= 0 else ""

    formatter = Jinja2ChatFormatter(
        template=template,
        eos_token=piece(llm.token_eos()),
        bos_token=piece(llm.token_bos()),
        add_generation_prompt=True,
    )

    return llm, make_render(formatter), model_id_for(model_path)
