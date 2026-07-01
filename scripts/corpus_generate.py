"""
scripts/corpus_generate.py
Live inference test bridging Llama.cpp and the SEAR logits constraint.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from llama_cpp import Llama, LogitsProcessorList

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gin.corpus.db import DatabaseUnavailableError, ensure_postgres
from gin.corpus.materialize import materialize_from_synthesis
from gin.corpus.prompts import build_synthesis_prompt
from gin.corpus.synthesis_manifest import render_synthesis_manifest
from sear.bias import BiasedGINLogitsProcessor
from sear.connectives import build_cite_inventory
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
        n_gpu_layers=0,
        verbose=True,
    )
    tokenize = lambda b: llm.tokenize(b, add_bos=False)

    filters: dict = {"eval_layer": args.eval_layer}
    if args.eval_tag:
        filters["eval_tag"] = args.eval_tag

    print(f"[*] Retrieving and materializing synthesis bundle for: {args.query}")
    corpus, ctx, bundle, retrieval_manifest = materialize_from_synthesis(
        query=args.query,
        tokenize=tokenize,
        k_seed=args.k_seed,
        k_max=args.k_max,
        filters=filters,
        min_rrf_delta=args.min_rrf_delta,
    )
    chunk_ids = [h.chunk_id for h in bundle.hits]
    print(f"[*] Retrieved chunks: {chunk_ids}")
    print(f"[*] Synthesis mode: {bundle.mode} ({len(bundle.hits)} chunks, {len(bundle.pairs)} pairs)")

    prompt = build_synthesis_prompt(
        args.query, bundle, chat_template=args.chat_template
    )
    prompt_ids = llm.tokenize(prompt.encode("utf-8"))

    conn_starts = ctx.connective_starts
    conn_cont = ctx.connective_continuations
    conn_phrases = ctx.connective_phrases
    force_conn = ctx.force_connective_ids
    cite_ids, cite_sequences, cite_cont = build_cite_inventory(tokenize, len(corpus.docs))

    divergent = bundle.mode == "divergent"
    require_cites = args.require_cites or divergent
    stop_when_satisfied = args.stop_when_satisfied or divergent
    block_eos = divergent and stop_when_satisfied
    min_span_len = args.min_span_len
    if min_span_len is None:
        min_span_len = 8 if divergent else 3
    max_tokens = args.max_tokens
    if max_tokens is None:
        if divergent:
            max_tokens = 40 + 25 * len(ctx.required_doc_groups)
        else:
            max_tokens = 120

    focus_docs = frozenset(range(len(corpus.docs))) if divergent else None

    constraint = ExtractiveCopyConstraint(
        corpus=corpus,
        prompt_len=len(prompt_ids),
        eos_id=llm.token_eos(),
        delim_id=tokenize(b"|")[-1],
        min_span_len=min_span_len,
        connective_starts=conn_starts,
        connective_continuations=conn_cont,
        connective_phrases=conn_phrases,
        cite_ids=cite_ids,
        cite_sequences_by_doc=cite_sequences,
        cite_continuations=cite_cont,
        close_on_doc_divergence=divergent,
        required_doc_groups=ctx.required_doc_groups,
        focus_doc_indices=focus_docs,
        reject_ambiguous_spans=divergent,
        allow_shared_prefix=not divergent,
        span_must_start_at_sentence=divergent,
        span_must_close_at_sentence_end=divergent,
        require_cite_after_extract=require_cites,
        stop_when_groups_satisfied=stop_when_satisfied,
        block_eos_until_groups_satisfied=block_eos,
        force_connective_ids=force_conn,
        preferred_starts=ctx.preferred_starts if divergent else None,
        forbidden_starts=ctx.forbidden_starts if divergent else None,
        divergence_starts=ctx.divergence_starts if divergent else None,
        divergence_sentence_ends=ctx.divergence_sentence_ends if divergent else None,
        ranked_sentence_starts=ctx.ranked_sentence_starts if divergent else None,
        require_divergence_after_first=divergent,
    )

    if args.verbose_constraint:
        print("[verbose] ranked_sentence_starts:", ctx.ranked_sentence_starts[:8])

    processor = BiasedGINLogitsProcessor(
        constraint, dynamic_bias=not args.no_logit_bias
    )
    processor_list = LogitsProcessorList([processor])

    print("[*] Starting constrained generation...")
    output = llm.create_completion(
        prompt,
        max_tokens=max_tokens,
        logits_processor=processor_list,
        temperature=0.0,
        echo=False,
    )

    segments = constraint.finalize()
    detok = lambda ids: llm.detokenize(ids).decode("utf-8", errors="replace")

    print("\n--- GENERATED TEXT (attributed) ---")
    render_output = constraint.render(detok)
    print(render_output)
    print("-----------------------------------\n")

    synthesis_manifest = render_synthesis_manifest(
        args.query,
        ctx,
        segments,
        render_output,
        retrieval_manifest=retrieval_manifest,
    )
    print(synthesis_manifest)
    print()

    print("--- RAW TOKEN STREAM ---")
    raw_text = output["choices"][0]["text"]
    print(raw_text)
    print("------------------------\n")

    if constraint.groups_satisfied():
        print("[*] Required doc groups satisfied (per-doc exhaustion).")

    if args.verbose_constraint:
        _verbose_constraint_log(
            constraint,
            cite_ids,
            corpus,
            max_tokens=max_tokens,
            generated_text=raw_text,
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
