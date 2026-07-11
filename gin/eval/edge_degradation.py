"""Decode-in-the-loop degradation under a noisy (class-C) edge.

Step 2 of docs/nc_reasoning_robustness_noisy_edges.plan.md. Step 1 proved the
reasoning gate cannot catch *mislabeled corroboration* (two agreeing, on-query
chunks mistyped ``contradicts``). This harness measures what the DECODE does when
such an edge gets through: it drives the real materialize + constrained-decode
path over hand-constructed clean vs. noisy edges and scores the answers with the
production metrics — showing that a grounded-but-wrong divergent answer passes
every existing metric (fabrication 0, divergence_fidelity 1.0,
supported_irrelevance 0), which is the motivation for a divergence-*validity*
metric distinct from divergence-*fidelity*.

Faithful without a specific model: the plan established SEAR's decode is
constraint-determined, not model-determined (fidelity 1.0 across Mistral/Qwen).
Pass a real llama.cpp ``Llama`` for an artifact, or ``GreedyMaskDecoder`` for a
deterministic CI regression — both drive the identical constraint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from gin.corpus.generate import decode_bundle
from gin.corpus.materialize import materialize_synthesis_bundle
from gin.corpus.models import ChunkHit, EdgeRecord, SynthesisBundle
from uuid import uuid4

from .claims import segments_to_raw_claims
from .metrics import (
    QueryResult,
    divergence_fidelity_for_query,
    fabrication_rate,
    supported_irrelevance_rate,
)
from .runner import verify_output
from .arms import ArmOutput
from .verifier import Verifier

_DOC_ID = uuid4()


# --- Deterministic decoder ---------------------------------------------------

class GreedyMaskDecoder:
    """Deterministic stand-in for llama.cpp that argmax-decodes under the mask.

    Faithful because SEAR's decode is constraint-determined: the constraint masks
    disallowed tokens to NEG_INF, biases preferred starts / cites above the 0
    baseline, and — once required doc-groups are satisfied — allows only ``{eos}``
    (sear/processor.py:307). So argmax reproduces the same extract any model
    would. A whitespace tokenizer is self-consistent with the corpus because the
    same ``tokenize`` builds the Corpus and drives the decode.
    """

    N_VOCAB = 8192

    def __init__(self) -> None:
        self._tok_to_id: dict[str, int] = {}
        self._id_to_tok: dict[int, str] = {}
        self._eos = self.N_VOCAB - 1

    def _id(self, tok: str) -> int:
        if tok not in self._tok_to_id:
            nid = len(self._tok_to_id) + 1  # ids start at 1; eos reserved high
            if nid >= self._eos:
                raise RuntimeError("GreedyMaskDecoder vocab overflow")
            self._tok_to_id[tok] = nid
            self._id_to_tok[nid] = tok
        return self._tok_to_id[tok]

    def tokenize(self, b: bytes, add_bos: bool = False) -> list[int]:
        return [self._id(w) for w in b.decode("utf-8", errors="replace").split()]

    def detokenize(self, ids: list[int]) -> bytes:
        toks = [self._id_to_tok.get(int(i), "") for i in ids]
        return " ".join(t for t in toks if t).encode("utf-8")

    def token_eos(self) -> int:
        return self._eos

    def create_completion(
        self,
        prompt: str,
        *,
        max_tokens: int,
        logits_processor: Any,
        temperature: float = 0.0,
        echo: bool = False,
    ) -> dict:
        input_ids = list(self.tokenize(prompt.encode("utf-8")))
        generated: list[int] = []
        for _ in range(max_tokens):
            scores = np.zeros(self.N_VOCAB, dtype=np.float32)
            out = logits_processor(input_ids, scores)
            nxt = int(np.argmax(out))
            if nxt == self._eos:
                break
            generated.append(nxt)
            input_ids.append(nxt)
        text = self.detokenize(generated).decode("utf-8", errors="replace")
        return {"choices": [{"text": text}]}


# --- Fixture chunks (real divergence-demo corpus text) -----------------------
# Two agreeing institutional wildfire statistics (class-C target) plus the
# grassroots reframing that genuinely contradicts them (true-divergence control).

_WF_STAT = (
    "In 2023, 56,580 wildfires burned 2,693,910 acres across the United States, "
    "with acreage burned below both the five- and ten-year averages."
)
_WF_FEDERAL = (
    "About one-quarter of the nation's wildfires in 2023 occurred on federally "
    "protected lands."
)
_WF_GRASSROOTS = (
    "Elderly, immunocompromised, and low-income populations face heightened risk "
    "from wildfire smoke exposure."
)
_QUERY = "What is the main concern about wildfires in the United States?"


def _hit(chunk_id: str, text: str, outlet: str) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        doc_id=_DOC_ID,
        text=text,
        head_sentence="",
        eval_layer="realism",
        eval_tag=None,
        content_hash="",
        outlet=outlet,
        title="t",
    )


@dataclass
class Scenario:
    id: str
    label: str
    query: str
    bundle: SynthesisBundle
    relation_is_real: bool  # ground truth: do the paired chunks actually conflict?
    contradicts_pairs: list[list[str]] = field(default_factory=list)


def default_scenarios() -> list[Scenario]:
    stat = _hit("wf_stat:0", _WF_STAT, "NIFC")
    federal = _hit("wf_fed:0", _WF_FEDERAL, "USFS")
    grass = _hit("wf_grass:0", _WF_GRASSROOTS, "Collective")

    contra_agree = EdgeRecord("wf_stat:0", "wf_fed:0", "contradicts")
    contra_real = EdgeRecord("wf_stat:0", "wf_grass:0", "contradicts")

    return [
        # Clean: two agreeing institutional facts, no edge → convergent.
        Scenario(
            "clean_convergent", "clean (agreeing pair, no edge)", _QUERY,
            SynthesisBundle(hits=[stat, federal], edges=[], mode="convergent"),
            relation_is_real=False,
        ),
        # NOISY (class C): same agreeing pair, mistyped contradicts → divergent.
        Scenario(
            "noisy_divergent", "NOISY class-C (agreeing pair, contradicts)", _QUERY,
            SynthesisBundle(
                hits=[stat, federal],
                edges=[contra_agree],
                mode="divergent",
                pairs=[(stat, federal, contra_agree)],
            ),
            relation_is_real=False,
            contradicts_pairs=[["wf_stat:0", "wf_fed:0"]],
        ),
        # Control: genuine institutional-vs-grassroots contradiction → divergent.
        Scenario(
            "true_divergent", "control (real contradiction)", _QUERY,
            SynthesisBundle(
                hits=[stat, grass],
                edges=[contra_real],
                mode="divergent",
                pairs=[(stat, grass, contra_real)],
            ),
            relation_is_real=True,
            contradicts_pairs=[["wf_stat:0", "wf_grass:0"]],
        ),
    ]


@dataclass
class ScenarioResult:
    id: str
    label: str
    edge_type: str
    materialized_mode: str
    raw_text: str
    fabrication_rate: Optional[float]
    divergence_fidelity: Optional[float]
    supported_irrelevance_rate: Optional[float]
    spurious_divergence: bool
    relation_is_real: bool

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "edge_type": self.edge_type,
            "materialized_mode": self.materialized_mode,
            "raw_text": self.raw_text,
            "fabrication_rate": self.fabrication_rate,
            "divergence_fidelity": self.divergence_fidelity,
            "supported_irrelevance_rate": self.supported_irrelevance_rate,
            "spurious_divergence": self.spurious_divergence,
            "relation_is_real": self.relation_is_real,
        }


def run_scenario(
    scenario: Scenario,
    llm: Any,
    verifier: Optional[Verifier] = None,
) -> ScenarioResult:
    verifier = verifier or Verifier(mode="overlap", threshold=0.5)
    tokenize = lambda b: llm.tokenize(b, add_bos=False)
    detok = lambda ids: llm.detokenize(ids).decode("utf-8", errors="replace")

    corpus, ctx = materialize_synthesis_bundle(
        scenario.bundle, tokenize, query=scenario.query
    )
    result = decode_bundle(scenario.query, corpus, ctx, scenario.bundle, llm)

    doc_index_to_chunk_id = {
        i: hit.chunk_id for i, hit in ctx.doc_index_to_hit.items()
    }
    claims = segments_to_raw_claims(result.segments, detok, doc_index_to_chunk_id)
    output = ArmOutput(
        raw_text=result.raw_text,
        claims=claims,
        retrieval_manifest_hash="",
        refused=False,
        node_of={h.chunk_id: h.outlet for h in scenario.bundle.hits},
        chunks=[(h.chunk_id, h.text) for h in scenario.bundle.hits],
    )
    records = verify_output(output, verifier)

    qr = QueryResult(
        query_id=scenario.id,
        query=scenario.query,
        arm="no_continuation",
        eval_layer="realism",
        expectation="answerable",
        refused=False,
        claims=records,
        eval_tag="wildfire_divergence",
        retrieved_chunk_ids=[h.chunk_id for h in scenario.bundle.hits],
        gold_chunk_ids=[],
        contradicts_pairs=scenario.contradicts_pairs,
    )

    mode = ctx.mode
    return ScenarioResult(
        id=scenario.id,
        label=scenario.label,
        edge_type=(scenario.bundle.edges[0].edge_type if scenario.bundle.edges else "none"),
        materialized_mode=mode,
        raw_text=result.raw_text,
        fabrication_rate=fabrication_rate([qr]),
        divergence_fidelity=divergence_fidelity_for_query(qr),
        supported_irrelevance_rate=supported_irrelevance_rate([qr]),
        spurious_divergence=(mode == "divergent" and not scenario.relation_is_real),
        relation_is_real=scenario.relation_is_real,
    )


def run_degradation(
    llm: Any,
    scenarios: Optional[list[Scenario]] = None,
    verifier: Optional[Verifier] = None,
) -> list[ScenarioResult]:
    scenarios = scenarios or default_scenarios()
    return [run_scenario(s, llm, verifier) for s in scenarios]


def format_degradation_report(results: list[ScenarioResult]) -> str:
    lines = ["# Decode-in-the-loop degradation under a noisy edge", ""]
    lines.append("| scenario | edge | mode | fabrication | div_fidelity | supp_irrel | spurious |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r.label} | {r.edge_type} | {r.materialized_mode} "
            f"| {_fmt(r.fabrication_rate)} | {_fmt(r.divergence_fidelity)} "
            f"| {_fmt(r.supported_irrelevance_rate)} | {r.spurious_divergence} |"
        )
    lines.append("")
    lines.append("## Answers")
    lines.append("")
    for r in results:
        lines.append(f"- **{r.label}** ({r.materialized_mode}): {r.raw_text!r}")
    return "\n".join(lines)


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"
