"""Synthesis prompt templates for multi-document SEAR generation."""
from __future__ import annotations

from .models import ChunkHit, EdgeRecord, SynthesisBundle, SynthesisMode


def _truncate(text: str, max_len: int = 60) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _chunk_label(hit: ChunkHit, index: int) -> str:
    title = hit.title.strip() or hit.chunk_id
    return f"[{index}] {hit.outlet} — \"{_truncate(hit.head_sentence or hit.text)}\" (chunk {hit.chunk_id})"


def _relationship_lines(
    pairs: list[tuple[ChunkHit, ChunkHit, EdgeRecord]],
    cite_index: dict[str, int],
) -> list[str]:
    lines: list[str] = []
    for left, right, edge in pairs:
        li = cite_index.get(left.chunk_id, 0)
        ri = cite_index.get(right.chunk_id, 0)
        note = f" ({edge.note})" if edge.note else ""
        if edge.edge_type == "contradicts":
            lines.append(f"Relationship: [{li}] contradicts [{ri}]{note}.")
        elif edge.edge_type == "cites":
            lines.append(f"Relationship: [{li}] cites [{ri}]{note}.")
        else:
            lines.append(f"Relationship: [{li}] {edge.edge_type} [{ri}]{note}.")
    return lines


def build_source_manifest(
    hits: list[ChunkHit],
    edges: list[EdgeRecord] | None = None,
    pairs: list[tuple[ChunkHit, ChunkHit, EdgeRecord]] | None = None,
) -> str:
    """Metadata-only source list; chunk bodies live in the SEAR corpus."""
    if not hits:
        return "Sources: (none retrieved)\n"

    cite_index = {hit.chunk_id: i + 1 for i, hit in enumerate(hits)}
    lines = ["Sources:"]
    for i, hit in enumerate(hits, start=1):
        lines.append(_chunk_label(hit, i))

    pair_lines = _relationship_lines(pairs or [], cite_index)
    if pair_lines:
        lines.extend(pair_lines)
    elif edges:
        for edge in edges:
            li = cite_index.get(edge.src_chunk_id)
            ri = cite_index.get(edge.dst_chunk_id)
            if li is None or ri is None:
                continue
            note = f" ({edge.note})" if edge.note else ""
            lines.append(f"Relationship: [{li}] {edge.edge_type} [{ri}]{note}.")

    return "\n".join(lines) + "\n"


def _task_instruction(query: str, mode: SynthesisMode) -> str:
    if mode == "divergent":
        return (
            f"Using only verbatim spans from the sources above, answer: {query}\n"
            "Present both positions without resolving the conflict; use contrastive "
            "connectives between opposing spans."
        )
    return (
        f"Using only verbatim spans from the sources above, weave a single answer "
        f"to: {query}"
    )


def _format_hint(mode: SynthesisMode) -> str:
    base = (
        "Separate distinct extracted spans with '|' or natural connectives. "
        "Do not paraphrase."
    )
    if mode == "divergent":
        return (
            f"{base} After each extracted span, emit its source marker [n]. "
            'Between conflicting spans use "however" or "by contrast".'
        )
    return base


def build_synthesis_prompt(
    query: str,
    bundle: SynthesisBundle,
    *,
    chat_template: str = "mistral",
) -> str:
    """Assemble manifest + task + format hint for constrained synthesis."""
    manifest = build_source_manifest(bundle.hits, bundle.edges, bundle.pairs)
    task = _task_instruction(query, bundle.mode)
    fmt = _format_hint(bundle.mode)
    body = f"{manifest}\n{task}\n{fmt}"

    if chat_template == "mistral":
        return f"[INST] {body} [/INST]"
    return f"{body}\n\n"


def build_synthesis_prompt_from_hits(
    query: str,
    hits: list[ChunkHit],
    *,
    mode: SynthesisMode = "convergent",
    chat_template: str = "mistral",
) -> str:
    """Convenience wrapper when only flat hits are available."""
    bundle = SynthesisBundle(hits=hits, edges=[], mode=mode, pairs=[])
    return build_synthesis_prompt(query, bundle, chat_template=chat_template)
