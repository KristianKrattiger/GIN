"""Reusable No-Continuation (SEAR extractive) generation.

Extracted from ``scripts/corpus_generate.py`` so both the CLI and the eval
harness share one code path: retrieve -> materialize -> constrained decode ->
attribute. ``llm`` is duck-typed to the llama.cpp ``Llama`` interface
(``tokenize``, ``detokenize``, ``token_eos``, ``create_completion``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from sear.bias import BiasedGINLogitsProcessor
from sear.connectives import build_cite_inventory
from sear.corpus import Corpus
from sear.processor import ExtractiveCopyConstraint, Segment

from .materialize import materialize_from_synthesis
from .models import SynthesisBundle, SynthesisContext
from .prompts import build_synthesis_prompt
from .retrieval_manifest import RetrievalManifest
from .retrieve import RETRIEVAL_CONFIDENCE_FLOOR


@dataclass
class NoContinuationResult:
    corpus: Corpus
    ctx: SynthesisContext
    bundle: SynthesisBundle
    retrieval_manifest: Optional[RetrievalManifest]
    constraint: ExtractiveCopyConstraint
    segments: list[Segment]
    render_output: str
    raw_text: str
    prompt: str
    cite_ids: dict[int, int]
    max_tokens: int


def _resolve_decode_params(
    bundle: SynthesisBundle,
    ctx: SynthesisContext,
    *,
    require_cites: bool,
    stop_when_satisfied: bool,
    min_span_len: Optional[int],
    max_tokens: Optional[int],
) -> dict[str, Any]:
    divergent = bundle.mode == "divergent"
    has_groups = bool(ctx.required_doc_groups)
    require_cites = require_cites or divergent
    # Group-based stopping/EOS-blocking is meaningless without groups: with an
    # empty group list _groups_satisfied() never becomes True, so blocking EOS
    # would force the decoder to ramble off-topic spans until max_tokens.
    stop_when_satisfied = stop_when_satisfied or (divergent and has_groups)
    block_eos = divergent and stop_when_satisfied and has_groups
    stop_after_first_extract = not divergent
    if min_span_len is None:
        min_span_len = 8 if divergent else 3
    # Convergent bundles whose top hits are the same eval_tag from different
    # outlets are corroboration (e.g. bureau + independent survey agreeing on
    # one statistic). Decode them as one shared-prefix span attributed to all
    # corroborating docs, running to sentence end so numeric answers like
    # "3.7 percent" are never truncated.
    competing_same_tag = (
        not divergent
        and len(bundle.hits) >= 2
        and bundle.hits[0].eval_tag is not None
        and bundle.hits[0].eval_tag == bundle.hits[1].eval_tag
        and bundle.hits[0].outlet != bundle.hits[1].outlet
    )
    if max_tokens is None:
        if divergent:
            # Each required group is one contradicts pair -> two extracted
            # sentences (both sides) plus cite markers and a delimiter. Real
            # institutional sentences run ~55 tokens, so the old 25/group budget
            # left the second side truncated to a fragment ("Elderly,
            # immunocompromised"). Give each group room for two full sentences;
            # EOS still fires the instant both sides are quoted
            # (stop_when_groups_satisfied), so surplus budget is never spent.
            max_tokens = 40 + 90 * len(ctx.required_doc_groups)
        elif competing_same_tag:
            max_tokens = 100
        else:
            max_tokens = 60
    if competing_same_tag:
        stop_after_first_extract = False
    return {
        "divergent": divergent,
        "require_cites": require_cites,
        "stop_when_satisfied": stop_when_satisfied,
        "block_eos": block_eos,
        "stop_after_first_extract": stop_after_first_extract,
        "min_span_len": min_span_len,
        "max_tokens": max_tokens,
        "competing_same_tag": competing_same_tag,
    }


def generate_no_continuation(
    query: str,
    llm: Any,
    *,
    k_seed: int = 5,
    k_max: int = 6,
    filters: Optional[dict] = None,
    min_rrf_delta: float = 0.25,
    confidence_floor: float = RETRIEVAL_CONFIDENCE_FLOOR,
    chat_template: str = "mistral",
    require_cites: bool = False,
    stop_when_satisfied: bool = False,
    min_span_len: Optional[int] = None,
    max_tokens: Optional[int] = None,
    use_logit_bias: bool = True,
    query_steered: bool = True,
    gold_chunk_ids: Optional[list[str]] = None,
) -> NoContinuationResult:
    """Run the full extractive-only synthesis path for a single query.

    Raises ``RetrievalConfidenceError`` when retrieval cannot ground the query.
    """
    from llama_cpp import LogitsProcessorList  # lazy: keep import off module load

    tokenize: Callable[[bytes], list[int]] = lambda b: llm.tokenize(b, add_bos=False)
    detok: Callable[[list[int]], str] = lambda ids: llm.detokenize(ids).decode(
        "utf-8", errors="replace"
    )

    corpus, ctx, bundle, retrieval_manifest = materialize_from_synthesis(
        query=query,
        tokenize=tokenize,
        k_seed=k_seed,
        k_max=k_max,
        filters=filters,
        min_rrf_delta=min_rrf_delta,
        confidence_floor=confidence_floor,
        gold_chunk_ids=gold_chunk_ids,
    )

    prompt = build_synthesis_prompt(query, bundle, chat_template=chat_template)
    prompt_ids = llm.tokenize(prompt.encode("utf-8"))

    cite_ids, cite_sequences, cite_cont = build_cite_inventory(
        tokenize, len(corpus.docs)
    )

    params = _resolve_decode_params(
        bundle,
        ctx,
        require_cites=require_cites,
        stop_when_satisfied=stop_when_satisfied,
        min_span_len=min_span_len,
        max_tokens=max_tokens,
    )
    divergent = params["divergent"]
    competing = params["competing_same_tag"]
    steered = query_steered and bool(ctx.preferred_starts or ctx.ranked_sentence_starts)
    if divergent:
        focus_docs = frozenset(range(len(corpus.docs)))
    elif steered and ctx.top_doc_idx is not None:
        focus_docs = frozenset({ctx.top_doc_idx})
        if competing:
            # Corroborating same-tag docs are equally valid sources; keeping
            # them in focus lets the shared sentence close as one AMBIGUOUS
            # span citing every corroborating doc.
            top_tag = ctx.doc_index_to_hit[ctx.top_doc_idx].eval_tag
            if top_tag is not None:
                focus_docs = frozenset(
                    i for i, hit in ctx.doc_index_to_hit.items()
                    if hit.eval_tag == top_tag
                ) | focus_docs
    else:
        focus_docs = None

    preferred_starts = ctx.preferred_starts if steered else None
    if competing and steered and preferred_starts and focus_docs:
        # Mirror the top doc's preferred sentence starts onto the other
        # corroborating docs (identical wire copy aligns token positions).
        mirrored = set(preferred_starts)
        for doc, pos in preferred_starts:
            for other in focus_docs:
                if other != doc and (other, pos) in corpus.sentence_starts:
                    mirrored.add((other, pos))
        preferred_starts = mirrored

    constraint = ExtractiveCopyConstraint(
        corpus=corpus,
        prompt_len=len(prompt_ids),
        eos_id=llm.token_eos(),
        delim_id=tokenize(b"|")[-1],
        min_span_len=params["min_span_len"],
        connective_starts=ctx.connective_starts,
        connective_continuations=ctx.connective_continuations,
        connective_phrases=ctx.connective_phrases,
        cite_ids=cite_ids,
        cite_sequences_by_doc=cite_sequences,
        cite_continuations=cite_cont,
        close_on_doc_divergence=divergent,
        required_doc_groups=ctx.required_doc_groups,
        focus_doc_indices=focus_docs,
        reject_ambiguous_spans=divergent,
        allow_shared_prefix=divergent or competing or not steered,
        span_must_start_at_sentence=divergent or competing,
        span_must_close_at_sentence_end=divergent or competing,
        require_cite_after_extract=params["require_cites"],
        stop_when_groups_satisfied=params["stop_when_satisfied"],
        stop_after_first_extract=params["stop_after_first_extract"],
        block_eos_until_groups_satisfied=params["block_eos"],
        force_connective_ids=ctx.force_connective_ids,
        preferred_starts=preferred_starts,
        forbidden_starts=ctx.forbidden_starts if divergent else None,
        divergence_starts=ctx.divergence_starts if divergent else None,
        divergence_sentence_ends=ctx.divergence_sentence_ends if divergent else None,
        ranked_sentence_starts=ctx.ranked_sentence_starts if steered else None,
        require_divergence_after_first=divergent,
    )

    processor = BiasedGINLogitsProcessor(constraint, dynamic_bias=use_logit_bias)
    processor_list = LogitsProcessorList([processor])

    output = llm.create_completion(
        prompt,
        max_tokens=params["max_tokens"],
        logits_processor=processor_list,
        temperature=0.0,
        echo=False,
    )

    segments = constraint.finalize()
    render_output = constraint.render(detok)
    raw_text = output["choices"][0]["text"]

    return NoContinuationResult(
        corpus=corpus,
        ctx=ctx,
        bundle=bundle,
        retrieval_manifest=retrieval_manifest,
        constraint=constraint,
        segments=segments,
        render_output=render_output,
        raw_text=raw_text,
        prompt=prompt,
        cite_ids=cite_ids,
        max_tokens=params["max_tokens"],
    )
