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

    def render(system: str, user: str) -> str:
        return formatter(messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]).prompt

    return llm, render, model_id_for(model_path)
