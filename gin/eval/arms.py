"""Generation arms for the designed experiment.

Every arm shares the same retrieval (held constant) and differs only in how it
turns retrieved chunks into an answer:

- ``RagArm``              unconstrained model completion with a cite-your-sources
                          prompt (traditional RAG baseline).
- ``NoContinuationArm``   today's SEAR extractive-only path (Mode 1).
- ``FlaggedGenerationArm`` stub for Mode 2 (INFERRED/PARAPHRASE); not yet built.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from gin.corpus.generate import generate_no_continuation
from gin.corpus.models import SynthesisBundle
from gin.corpus.retrieval_manifest import build_retrieval_manifest, write_retrieval_manifest
from gin.corpus.retrieve import (
    RETRIEVAL_CONFIDENCE_FLOOR,
    RetrievalConfidenceError,
    retrieve_for_synthesis,
)

from .claims import RawClaim, rag_text_to_raw_claims, segments_to_raw_claims
from .verifier import max_query_overlap, token_overlap
from gin.corpus.relevance import shares_query_keyword

REFUSAL_SENTINEL = "The sources do not support an answer."
_REFUSAL_MARKER = "not support an answer"
DEFAULT_RAG_MAX_TOKENS = 256
DEFAULT_RELEVANCE_FLOOR = 0.20


@dataclass
class ArmConfig:
    k_seed: int = 5
    k_max: int = 6
    filters: Optional[dict] = None
    min_rrf_delta: float = 0.25
    confidence_floor: float = RETRIEVAL_CONFIDENCE_FLOOR
    relevance_floor: float = DEFAULT_RELEVANCE_FLOOR
    max_tokens: Optional[int] = None
    chat_template: str = "mistral"
    use_logit_bias: bool = True
    boost_gold_chunks: bool = False
    gold_refuse_without_coverage: bool = False
    gold_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class ArmOutput:
    raw_text: str
    claims: list[RawClaim]
    retrieval_manifest_hash: str
    refused: bool = False
    # chunk_id -> node id, so the runner can classify claim node-scope.
    node_of: dict[str, str] = field(default_factory=dict)
    # (chunk_id, text) candidates the verifier scores claims against.
    chunks: list[tuple[str, str]] = field(default_factory=list)
    # Structured grounding-failure signal: "" when not refused, else
    # "retrieval_floor" (pre-decode confidence floor) or "zero_cursors"
    # (decode produced no grounded, query-relevant claims). Federation v1
    # delegates on exactly these pre-commitment reasons.
    refusal_reason: str = ""
    # Bundle mode on success ("convergent"/"divergent"), "" on refusal.
    synthesis_mode: str = ""


@runtime_checkable
class Arm(Protocol):
    name: str

    def run(self, query: str, llm: Any) -> ArmOutput: ...


def _node_map(bundle: SynthesisBundle) -> dict[str, str]:
    """Default node identity per chunk = its outlet (federation stand-in)."""
    return {hit.chunk_id: hit.outlet for hit in bundle.hits}


def _chunk_texts(bundle: SynthesisBundle) -> list[tuple[str, str]]:
    return [(hit.chunk_id, hit.text) for hit in bundle.hits]


def _retrieval_relevant(query: str, bundle: SynthesisBundle, relevance_floor: float) -> bool:
    return max_query_overlap(query, _chunk_texts(bundle)) >= relevance_floor


def _raw_claims_cite_gold(claims: list[RawClaim], gold_chunk_ids: list[str]) -> bool:
    if not gold_chunk_ids:
        return True
    gold = set(gold_chunk_ids)
    return any(gold.intersection(claim.cited_chunk_ids) for claim in claims)


def _claims_query_relevant(
    query: str,
    claims: list[RawClaim],
    relevance_floor: float,
) -> bool:
    """True when some emitted claim actually answers the query.

    Both tests judge the CLAIM's own text — never the cited chunk. Checking the
    whole cited chunk let an on-topic multi-sentence chunk vouch for an
    off-topic extracted sentence: the out-of-scope ``harbor district
    referendum`` chunk matches a vote-margin query, so its turnout-% span
    (a substring of that chunk) was wrongly judged relevant instead of refused
    (plan §6 #4). A claim qualifies when either:
      1. its direct token overlap with the query clears the floor, or
      2. it shares a normalized query keyword — the singular/plural fold
         (``wildfire`` ~ ``wildfires``) the divergence gate uses, so a diluted
         divergent extract (overlap well under the floor) still qualifies while
         a turnout span sharing no query keyword does not.
    """
    if not claims:
        return False
    for claim in claims:
        if token_overlap(claim.text, query) >= relevance_floor:
            return True
        if shares_query_keyword(claim.text, query):
            return True
    return False


def _refusal_output(
    manifest_hash: str = "",
    bundle: Optional[SynthesisBundle] = None,
    *,
    reason: str = "zero_cursors",
) -> ArmOutput:
    return ArmOutput(
        raw_text=REFUSAL_SENTINEL,
        claims=[],
        retrieval_manifest_hash=manifest_hash,
        refused=True,
        node_of=_node_map(bundle) if bundle else {},
        chunks=_chunk_texts(bundle) if bundle else [],
        refusal_reason=reason,
    )


def build_rag_prompt(
    query: str,
    bundle: SynthesisBundle,
    *,
    chat_template: str = "mistral",
) -> str:
    """RAG prompt: full chunk bodies in-context + cite-or-refuse instruction."""
    lines = ["Sources:"]
    for i, hit in enumerate(bundle.hits, start=1):
        title = hit.title.strip() or hit.chunk_id
        lines.append(f"[{i}] {hit.outlet} — {title} (chunk {hit.chunk_id})")
        lines.append(hit.text.strip())
        lines.append("")
    body = "\n".join(lines).rstrip()
    body += (
        "\n\nAnswer the question using only the sources above. "
        "Cite each claim with its source marker [n]. "
        f'If the sources do not contain the answer, reply exactly: "{REFUSAL_SENTINEL}"\n'
        f"Question: {query}"
    )
    if chat_template == "mistral":
        return f"[INST] {body} [/INST]"
    return f"{body}\n\n"


class RagArm:
    name = "rag"

    def __init__(self, config: Optional[ArmConfig] = None) -> None:
        self.config = config or ArmConfig()

    def run(self, query: str, llm: Any) -> ArmOutput:
        cfg = self.config
        try:
            bundle = retrieve_for_synthesis(
                query,
                k_seed=cfg.k_seed,
                k_max=cfg.k_max,
                filters=cfg.filters,
                min_rrf_delta=cfg.min_rrf_delta,
                confidence_floor=cfg.confidence_floor,
            )
        except RetrievalConfidenceError:
            return _refusal_output()

        if not _retrieval_relevant(query, bundle, cfg.relevance_floor):
            manifest = build_retrieval_manifest(query, bundle)
            write_retrieval_manifest(manifest)
            return _refusal_output(manifest.manifest_hash, bundle)

        manifest = build_retrieval_manifest(query, bundle)
        write_retrieval_manifest(manifest)

        prompt = build_rag_prompt(query, bundle, chat_template=cfg.chat_template)
        output = llm.create_completion(
            prompt,
            max_tokens=cfg.max_tokens or DEFAULT_RAG_MAX_TOKENS,
            temperature=0.0,
            echo=False,
        )
        raw_text = output["choices"][0]["text"]

        refused = _REFUSAL_MARKER in raw_text.lower()
        cite_index_to_chunk_id = {
            i + 1: hit.chunk_id for i, hit in enumerate(bundle.hits)
        }
        claims = (
            [] if refused else rag_text_to_raw_claims(raw_text, cite_index_to_chunk_id)
        )
        return ArmOutput(
            raw_text=raw_text,
            claims=claims,
            retrieval_manifest_hash=manifest.manifest_hash,
            refused=refused,
            node_of=_node_map(bundle),
            chunks=_chunk_texts(bundle),
        )


class NoContinuationArm:
    name = "no_continuation"

    def __init__(self, config: Optional[ArmConfig] = None) -> None:
        self.config = config or ArmConfig()

    def run(self, query: str, llm: Any) -> ArmOutput:
        cfg = self.config
        detok = lambda ids: llm.detokenize(ids).decode("utf-8", errors="replace")
        try:
            result = generate_no_continuation(
                query,
                llm,
                k_seed=cfg.k_seed,
                k_max=cfg.k_max,
                filters=cfg.filters,
                min_rrf_delta=cfg.min_rrf_delta,
                confidence_floor=cfg.confidence_floor,
                chat_template=cfg.chat_template,
                max_tokens=cfg.max_tokens,
                use_logit_bias=cfg.use_logit_bias,
                gold_chunk_ids=cfg.gold_chunk_ids if cfg.boost_gold_chunks else None,
            )
        except RetrievalConfidenceError:
            return _refusal_output(reason="retrieval_floor")

        bundle = result.bundle
        if not _retrieval_relevant(query, bundle, cfg.relevance_floor):
            manifest_hash = (
                result.retrieval_manifest.manifest_hash
                if result.retrieval_manifest
                else ""
            )
            return _refusal_output(manifest_hash, bundle)

        doc_index_to_chunk_id = {
            i: hit.chunk_id for i, hit in result.ctx.doc_index_to_hit.items()
        }
        claims = segments_to_raw_claims(result.segments, detok, doc_index_to_chunk_id)
        if not _claims_query_relevant(query, claims, cfg.relevance_floor):
            manifest_hash = (
                result.retrieval_manifest.manifest_hash
                if result.retrieval_manifest
                else ""
            )
            return _refusal_output(manifest_hash, bundle)
        if (
            cfg.gold_refuse_without_coverage
            and cfg.gold_chunk_ids
            and not _raw_claims_cite_gold(claims, cfg.gold_chunk_ids)
        ):
            manifest_hash = (
                result.retrieval_manifest.manifest_hash
                if result.retrieval_manifest
                else ""
            )
            return _refusal_output(manifest_hash, bundle)
        manifest_hash = (
            result.retrieval_manifest.manifest_hash if result.retrieval_manifest else ""
        )
        return ArmOutput(
            raw_text=result.raw_text,
            claims=claims,
            retrieval_manifest_hash=manifest_hash,
            refused=False,
            node_of=_node_map(result.bundle),
            chunks=_chunk_texts(result.bundle),
            synthesis_mode=result.bundle.mode,
        )


class FlaggedGenerationArm:
    name = "flagged_generation"

    def __init__(self, config: Optional[ArmConfig] = None) -> None:
        self.config = config or ArmConfig()

    def run(self, query: str, llm: Any) -> ArmOutput:
        raise NotImplementedError(
            "Flagged Generation (Mode 2, INFERRED/PARAPHRASE spans) is not "
            "implemented yet. It is registered so the harness reserves a column; "
            "see the plan's out-of-scope section."
        )


ARM_REGISTRY: dict[str, type] = {
    RagArm.name: RagArm,
    NoContinuationArm.name: NoContinuationArm,
    FlaggedGenerationArm.name: FlaggedGenerationArm,
}


def build_arm(name: str, config: Optional[ArmConfig] = None) -> Arm:
    try:
        cls = ARM_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown arm {name!r}; available: {sorted(ARM_REGISTRY)}"
        ) from exc
    return cls(config)
