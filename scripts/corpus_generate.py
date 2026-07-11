"""
scripts/corpus_generate.py
Live inference test bridging Llama.cpp and the SEAR logits constraint.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from llama_cpp import Llama

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gin.corpus.db import DatabaseUnavailableError, ensure_postgres
from gin.corpus.generate import generate_no_continuation
from gin.corpus.retrieve import RetrievalConfidenceError
from gin.corpus.synthesis_manifest import render_synthesis_manifest
from sear.processor import ExtractiveCopyConstraint


def _verbose_constraint_log(
    constraint: ExtractiveCopyConstraint,
    cite_ids: dict[int, int],
    corpus,
    *,
    max_tokens: int,
    generated_text: str,
) -> None:
    print("[verbose] required_doc_groups:", constraint.required_doc_groups)
    cite_map = {tok: doc for tok, doc in sorted(cite_ids.items())}
    print("[verbose] cite_ids (final token -> doc):", cite_map)
    print("[verbose] preferred_starts:", sorted(constraint.preferred_starts))
    print("[verbose] forbidden_starts:", sorted(constraint.forbidden_starts))
    print("[verbose] divergence_starts:", {
        d: sorted(poses) for d, poses in constraint.divergence_starts.items()
    })
    print("[verbose] divergence_sentence_ends:", constraint.divergence_sentence_ends)
    ends_per_doc = {
        d: sorted(p for doc, p in corpus.sentence_ends if doc == d)
        for d in range(len(corpus.docs))
    }
    print("[verbose] sentence_ends per doc (sample):", ends_per_doc)
    print("[verbose] per-group satisfaction:", constraint.group_satisfaction_status())
    print("[verbose] quoted_docs after generation:", sorted(constraint.quoted_docs()))
    print(f"[verbose] generated length: {len(generated_text.split())} tokens (approx words)")
    print(f"[verbose] max_tokens budget: {max_tokens}")
    if constraint.groups_satisfied():
        print("[verbose] stop reason: groups_satisfied")
    else:
        print("[verbose] stop reason: eos_or_max_tokens (groups not satisfied)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to local GGUF model")
    parser.add_argument("--query", type=str, default="downtown incident hospital treatment")
    parser.add_argument("--chat-template", type=str, default="mistral", choices=["mistral", "plain"])
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--eval-layer", type=str, default="realism")
    parser.add_argument("--eval-tag", type=str, default=None, help="Optional eval_tag filter")
    parser.add_argument("--min-rrf-delta", type=float, default=0.25)
    parser.add_argument("--k-seed", type=int, default=5)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--require-cites", action="store_true", help="Force cite marker after each extract")
    parser.add_argument("--stop-when-satisfied", action="store_true", help="Allow only EOS once doc groups satisfied")
    parser.add_argument("--min-span-len", type=int, default=None, help="Minimum extract span length")
    parser.add_argument("--no-logit-bias", action="store_true", help="Disable cite/connective logit bias")
    parser.add_argument(
        "--verbose-constraint",
        action="store_true",
        help="Log constraint diagnostics (groups, cites, steering, stop reason)",
    )
    parser.add_argument("--n-gpu-layers", type=int, default=0, help="Layers to offload to GPU (-1 for all)")
    args = parser.parse_args()

    try:
        ensure_postgres()
    except DatabaseUnavailableError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"[*] Loading model from {args.model}...")
    llm = Llama(
        model_path=args.model,
        n_ctx=2048,
        n_gpu_layers=args.n_gpu_layers,
        verbose=True,
    )

    filters: dict = {"eval_layer": args.eval_layer}
    if args.eval_tag:
        filters["eval_tag"] = args.eval_tag

    print(f"[*] Retrieving and materializing synthesis bundle for: {args.query}")
    try:
        result = generate_no_continuation(
            args.query,
            llm,
            k_seed=args.k_seed,
            k_max=args.k_max,
            filters=filters,
            min_rrf_delta=args.min_rrf_delta,
            chat_template=args.chat_template,
            require_cites=args.require_cites,
            stop_when_satisfied=args.stop_when_satisfied,
            min_span_len=args.min_span_len,
            max_tokens=args.max_tokens,
            use_logit_bias=not args.no_logit_bias,
        )
    except RetrievalConfidenceError as exc:
        print(f"[!] {exc}", file=sys.stderr)
        return 1

    bundle = result.bundle
    corpus = result.corpus
    ctx = result.ctx
    constraint = result.constraint
    segments = result.segments
    chunk_ids = [h.chunk_id for h in bundle.hits]
    print(f"[*] Retrieved chunks: {chunk_ids}")
    print(f"[*] Synthesis mode: {bundle.mode} ({len(bundle.hits)} chunks, {len(bundle.pairs)} pairs)")

    if args.verbose_constraint:
        print("[verbose] ranked_sentence_starts:", ctx.ranked_sentence_starts[:8])

    detok = lambda ids: llm.detokenize(ids).decode("utf-8", errors="replace")

    print("\n--- GENERATED TEXT (attributed) ---")
    print(result.render_output)
    print("-----------------------------------\n")

    synthesis_manifest = render_synthesis_manifest(
        args.query,
        ctx,
        segments,
        result.render_output,
        retrieval_manifest=result.retrieval_manifest,
    )
    print(synthesis_manifest)
    print()

    print("--- RAW TOKEN STREAM ---")
    print(result.raw_text)
    print("------------------------\n")

    if constraint.groups_satisfied():
        print("[*] Required doc groups satisfied (per-doc exhaustion).")

    if args.verbose_constraint:
        _verbose_constraint_log(
            constraint,
            result.cite_ids,
            corpus,
            max_tokens=result.max_tokens,
            generated_text=result.raw_text,
        )
        for seg in segments:
            preview = detok(seg.token_ids)[:60]
            outlets = {
                constraint._doc_label(d) for d, _, _ in seg.sources
            } if seg.sources else set()
            print(
                f"[verbose] segment {seg.kind}: outlets={outlets} preview={preview!r}"
            )


if __name__ == "__main__":
    main()
