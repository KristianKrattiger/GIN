"""Human-readable synthesis provenance manifest."""
from __future__ import annotations

from sear.connectives import connective_mode_label
from sear.processor import Segment

from .models import SynthesisContext
from .retrieval_manifest import RetrievalManifest
from .retrieve import RETRIEVAL_CONFIDENCE_FLOOR


def _quoted_docs_from_segments(segments: list[Segment]) -> set[int]:
    docs: set[int] = set()
    for seg in segments:
        if seg.kind != "extract":
            continue
        for doc, _start, _end in seg.sources:
            docs.add(doc)
    return docs


def _groups_satisfied(ctx: SynthesisContext, segments: list[Segment]) -> bool:
    if not ctx.required_doc_groups:
        return True
    quoted = _quoted_docs_from_segments(segments)
    return all(group <= quoted for group in ctx.required_doc_groups)


def _format_start(pos: tuple[int, int], *, label: str) -> str:
    doc, start = pos
    return f"(doc={doc}, pos={start}) [{label}]"


def render_synthesis_manifest(
    query: str,
    ctx: SynthesisContext,
    segments: list[Segment],
    render_output: str,
    *,
    retrieval_manifest: RetrievalManifest | None = None,
) -> str:
    lines: list[str] = ["=== Synthesis Manifest ===", "", f'Query: "{query}"', ""]

    lines.append("--- Retrieval ---")
    if retrieval_manifest is None:
        lines.append("Manifest hash: (not recorded)")
    else:
        rel_path = (
            f"data/retrieval_manifests/"
            f"{retrieval_manifest.manifest_hash[:2]}/"
            f"{retrieval_manifest.manifest_hash}.json"
        )
        lines.append(
            f"Manifest hash: {retrieval_manifest.manifest_hash}  [{rel_path}]"
        )
        lines.append(f"Mode: {retrieval_manifest.synthesis_mode}")
        edge_label = ", ".join(retrieval_manifest.edge_types) or "(none)"
        lines.append(f"Edge types: {edge_label}")
        lines.append("Chunks retrieved:")
        for i, entry in enumerate(retrieval_manifest.entries, start=1):
            lines.append(
                f'  [{i}] {entry.outlet} — "{entry.title}" (chunk {entry.chunk_id})'
            )
            dense = entry.dense_rank if entry.dense_rank is not None else "—"
            sparse = entry.sparse_rank if entry.sparse_rank is not None else "—"
            lines.append(
                f"      dense_rank={dense}  sparse_rank={sparse}  rrf={entry.rrf_score:.4f}"
            )
        if retrieval_manifest.entries:
            top_score = retrieval_manifest.entries[0].rrf_score
            if top_score >= RETRIEVAL_CONFIDENCE_FLOOR:
                conf = (
                    f"PASS (top score {top_score:.4f} "
                    f"≥ {RETRIEVAL_CONFIDENCE_FLOOR:.4f})"
                )
            else:
                conf = (
                    f"FAIL (top score {top_score:.4f} "
                    f"< {RETRIEVAL_CONFIDENCE_FLOOR:.4f})"
                )
        else:
            conf = f"FAIL (no chunks; floor {RETRIEVAL_CONFIDENCE_FLOOR:.4f})"
        lines.append(f"Confidence floor: {conf}")
    lines.append("")

    lines.append("--- Graph ---")
    if ctx.edges:
        for edge in ctx.edges:
            note = f'    note: "{edge.note}"' if edge.note else ""
            lines.append(
                f"Edges active: {edge.edge_type}({edge.src_chunk_id} → {edge.dst_chunk_id})"
            )
            if note:
                lines.append(note)
    else:
        lines.append("Edges active: (none)")
    groups_repr = [set(g) for g in ctx.required_doc_groups]
    lines.append(f"Required doc groups: {groups_repr}")
    satisfied = "YES" if _groups_satisfied(ctx, segments) else "NO"
    lines.append(f"Groups satisfied: {satisfied}")
    lines.append("")

    lines.append("--- Steering ---")
    mode_label = connective_mode_label(ctx.active_edge_types)
    if ctx.active_edge_types:
        active = ", ".join(sorted(ctx.active_edge_types))
        lines.append(f"Connective mode: {mode_label}  [from {active} edge]")
    else:
        lines.append(f"Connective mode: {mode_label}")
    if ctx.preferred_starts:
        preferred = ", ".join(
            _format_start(pos, label="query-match") for pos in sorted(ctx.preferred_starts)
        )
        lines.append(f"Preferred starts: {preferred}")
    else:
        lines.append("Preferred starts: (none)")
    if ctx.forbidden_starts:
        forbidden = ", ".join(
            _format_start(pos, label="shared lede") for pos in sorted(ctx.forbidden_starts)
        )
        lines.append(f"Forbidden starts: {forbidden}")
    else:
        lines.append("Forbidden starts: (none)")
    lines.append("")

    extract_segments = [s for s in segments if s.kind == "extract"]
    steered = sum(1 for s in extract_segments if s.guidance == "steered")
    divergence_steered = sum(
        1 for s in extract_segments if s.guidance == "divergence-steered"
    )
    free = sum(1 for s in extract_segments if s.guidance == "")

    lines.append("--- Generation ---")
    lines.append(f"Spans extracted: {len(extract_segments)}")
    lines.append(f"Free selections: {free}")
    lines.append(f"Steered selections: {steered}")
    lines.append(f"Divergence-steered: {divergence_steered}")
    lines.append("")

    lines.append("--- Attribution ---")
    lines.append(render_output)

    return "\n".join(lines)
