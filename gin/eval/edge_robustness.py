"""Reasoning-layer robustness under noisy Cartographer edges.

Every `contradicts` edge in the corpus today is hand-curated and fully trusted;
the reasoning layer assumes each one is real and query-relevant. Before an
automated Cartographer feeds the reasoning layer *proposed* edges, we need to
know which classes of bad edge the reasoning layer's existing divergence gate
already deflects, and which slip through to corrupt synthesis. See
docs/nc_reasoning_robustness_noisy_edges.plan.md.

This harness is deliberately DB-free and LLM-free: it runs a taxonomy of noisy
edges through the *actual* gate the reasoning layer uses
(`gin.corpus.retrieve._is_ambiguous`), so the measurement is faithful to
production without a decode. IDF is built from the real two-node corpus so the
gate behaves exactly as in the working divergence demo rather than against a toy
corpus whose IDF would be an artifact (divergence plan §7.1).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import uuid4

from gin.corpus.models import ChunkHit, EdgeRecord
from gin.corpus.relevance import corpus_idf
from gin.corpus.retrieve import _is_ambiguous

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CORPUS_FILES = ("corpus_node1.json", "corpus_node2.json")
_SHARED_DOC_ID = uuid4()


class EdgeNoiseClass(str, Enum):
    """Failure modes a Cartographer could plausibly emit."""

    IRRELEVANT_PARTNER = "irrelevant_partner"       # A: one side off-query
    GENERIC_OVERLAP = "generic_overlap"             # B: shared word is generic
    MISLABELED_CORROBORATION = "mislabeled_corroboration"  # C: agreeing, mistyped
    DANGLING_ANCHOR = "dangling_anchor"             # D: endpoint absent
    TRUE_CONTRADICTION = "true_contradiction"       # E: real (control)


# Classes where a robust reasoning layer should NOT enter divergent mode.
_SHOULD_REJECT = frozenset({
    EdgeNoiseClass.IRRELEVANT_PARTNER,
    EdgeNoiseClass.GENERIC_OVERLAP,
    EdgeNoiseClass.MISLABELED_CORROBORATION,
    EdgeNoiseClass.DANGLING_ANCHOR,
})


@dataclass
class NoisyEdgeCase:
    """One (hits, edge, query) probe and the mode a robust layer should pick."""

    id: str
    noise_class: EdgeNoiseClass
    query: str
    hits: list[ChunkHit]
    edge: EdgeRecord
    should_force_divergent: bool
    note: str = ""


@dataclass
class CaseResult:
    id: str
    noise_class: EdgeNoiseClass
    gate_forced_divergent: bool
    should_force_divergent: bool

    @property
    def correct(self) -> bool:
        return self.gate_forced_divergent == self.should_force_divergent


@dataclass
class StressSummary:
    results: list[CaseResult]
    noise_rejection_rate: Optional[float]
    true_positive_retention: Optional[float]
    by_class: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "noise_rejection_rate": self.noise_rejection_rate,
            "true_positive_retention": self.true_positive_retention,
            "by_class": self.by_class,
            "results": [
                {
                    "id": r.id,
                    "noise_class": r.noise_class.value,
                    "gate_forced_divergent": r.gate_forced_divergent,
                    "should_force_divergent": r.should_force_divergent,
                    "correct": r.correct,
                }
                for r in self.results
            ],
        }


# --- Corpus / IDF ------------------------------------------------------------

def _load_corpus_texts(paths: Optional[list[Path]] = None) -> list[str]:
    files = paths or [_REPO_ROOT / name for name in _CORPUS_FILES]
    texts: list[str] = []
    for path in files:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for doc in data.get("documents", []):
            for chunk in doc.get("chunks", []):
                text = chunk.get("text", "")
                if text:
                    texts.append(text)
    return texts


def default_corpus_idf(paths: Optional[list[Path]] = None) -> dict[str, float]:
    """IDF over the real two-node corpus, matching the production gate.

    Falls back to IDF over the case fixture texts if the corpus JSON is absent,
    so the harness still runs (with a less faithful IDF) in a stripped checkout.
    """
    texts = _load_corpus_texts(paths)
    if not texts:
        texts = [h.text for c in default_stress_cases() for h in c.hits]
    return corpus_idf(texts)


# --- Fixture text (real divergence-demo sentences) ---------------------------
# Institutional (node 1) vs grassroots (node 2), verbatim from the two-node
# corpus the working demo runs on.

_INST_EMISSIONS = (
    "Global low-carbon transformations are needed to deliver cuts to predicted "
    "2030 greenhouse gas emissions of roughly 28 percent for a 2 degree C pathway "
    "and 42 percent for a 1.5 degree C pathway."
)
_GRASS_EMISSIONS = (
    "Indigenous-led resistance efforts are estimated to have stopped or delayed "
    "greenhouse gas pollution equivalent to roughly one-quarter of annual U.S. "
    "and Canadian emissions."
)
_INST_WILDFIRE = (
    "In 2023, 56,580 wildfires burned 2,693,910 acres across the United States, "
    "with acreage burned below both the five- and ten-year averages."
)
_INST_WILDFIRE_FEDERAL = (
    "About one-quarter of the nation's wildfires in 2023 occurred on federally "
    "protected lands."
)
_GRASS_WILDFIRE = (
    "Elderly, immunocompromised, and low-income populations face heightened risk "
    "from wildfire smoke exposure."
)
_INST_WATER = (
    "As of April 3, 2023, California's statewide snowpack held a snow water "
    "equivalent of 61.1 inches, or 237 percent of the April 1 average, one of the "
    "largest snowpacks on record."
)
_GRASS_WATER = (
    "Disadvantaged and cumulatively burdened communities are found to be "
    "disproportionately affected by water shortages, reflecting underlying "
    "inequities in water resource management."
)

_Q_EMISSIONS = "What is the most important thing the world should do about greenhouse gas emissions?"
_Q_WILDFIRE = "What is the main concern about wildfires in the United States?"
_Q_WATER = "How should water scarcity in California be understood?"


def _hit(chunk_id: str, text: str, outlet: str) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        doc_id=_SHARED_DOC_ID,
        text=text,
        head_sentence="",
        eval_layer="realism",
        eval_tag=None,
        content_hash="",
        outlet=outlet,
        title="t",
    )


def _contradicts(src: str, dst: str) -> EdgeRecord:
    return EdgeRecord(src_chunk_id=src, dst_chunk_id=dst, edge_type="contradicts")


def default_stress_cases() -> list[NoisyEdgeCase]:
    """Taxonomy grounded in the real divergence-demo corpus."""
    inst_em = _hit("inst_em:0", _INST_EMISSIONS, "IPCC")
    grass_em = _hit("grass_em:0", _GRASS_EMISSIONS, "Grist")
    inst_wf = _hit("inst_wf:0", _INST_WILDFIRE, "NIFC")
    inst_wf_fed = _hit("inst_wf_fed:0", _INST_WILDFIRE_FEDERAL, "NIFC")
    grass_wf = _hit("grass_wf:0", _GRASS_WILDFIRE, "Collective")
    inst_wa = _hit("inst_wa:0", _INST_WATER, "DWR")
    grass_wa = _hit("grass_wa:0", _GRASS_WATER, "PacInst")

    return [
        # E — real contradictions (control): should diverge.
        NoisyEdgeCase(
            "e_emissions", EdgeNoiseClass.TRUE_CONTRADICTION, _Q_EMISSIONS,
            [inst_em, grass_em], _contradicts("inst_em:0", "grass_em:0"),
            should_force_divergent=True,
            note="institutional mitigation target vs grassroots resistance framing",
        ),
        NoisyEdgeCase(
            "e_wildfire", EdgeNoiseClass.TRUE_CONTRADICTION, _Q_WILDFIRE,
            [inst_wf, grass_wf], _contradicts("inst_wf:0", "grass_wf:0"),
            should_force_divergent=True,
            note="acreage statistics vs health-vulnerability framing",
        ),
        NoisyEdgeCase(
            "e_water", EdgeNoiseClass.TRUE_CONTRADICTION, _Q_WATER,
            [inst_wa, grass_wa], _contradicts("inst_wa:0", "grass_wa:0"),
            should_force_divergent=True,
            note="snowpack measurement vs equity/justice framing",
        ),
        # A — irrelevant partner: wildfire chunk contradicts a water chunk,
        # asked under the wildfire query. The water side is off-query.
        NoisyEdgeCase(
            "a_wildfire_water", EdgeNoiseClass.IRRELEVANT_PARTNER, _Q_WILDFIRE,
            [inst_wf, grass_wa], _contradicts("inst_wf:0", "grass_wa:0"),
            should_force_divergent=False,
            note="off-query partner (water) under a wildfire query",
        ),
        NoisyEdgeCase(
            "a_emissions_water", EdgeNoiseClass.IRRELEVANT_PARTNER, _Q_EMISSIONS,
            [inst_em, grass_wa], _contradicts("inst_em:0", "grass_wa:0"),
            should_force_divergent=False,
            note="off-query partner (water) under an emissions query",
        ),
        # C — mislabeled corroboration: two institutional wildfire chunks that
        # AGREE (both 2023 wildfire statistics), mistyped as contradicts.
        NoisyEdgeCase(
            "c_wildfire_agree", EdgeNoiseClass.MISLABELED_CORROBORATION, _Q_WILDFIRE,
            [inst_wf, inst_wf_fed], _contradicts("inst_wf:0", "inst_wf_fed:0"),
            should_force_divergent=False,
            note="two agreeing 2023 wildfire statistics mistyped as contradicts",
        ),
        # D — dangling anchor: partner not in the retrieved hit set.
        NoisyEdgeCase(
            "d_dangling", EdgeNoiseClass.DANGLING_ANCHOR, _Q_WILDFIRE,
            [inst_wf], _contradicts("inst_wf:0", "ghost:0"),
            should_force_divergent=False,
            note="edge endpoint absent from retrieval",
        ),
    ]


# --- Runner ------------------------------------------------------------------

def run_case(case: NoisyEdgeCase, idf: dict[str, float]) -> CaseResult:
    forced = _is_ambiguous(case.hits, [case.edge], case.query, idf)
    return CaseResult(
        id=case.id,
        noise_class=case.noise_class,
        gate_forced_divergent=forced,
        should_force_divergent=case.should_force_divergent,
    )


def run_stress(
    cases: Optional[list[NoisyEdgeCase]] = None,
    idf: Optional[dict[str, float]] = None,
) -> StressSummary:
    cases = cases or default_stress_cases()
    idf = idf if idf is not None else default_corpus_idf()
    results = [run_case(c, idf) for c in cases]

    reject = [r for r in results if r.noise_class in _SHOULD_REJECT]
    retain = [r for r in results if r.noise_class == EdgeNoiseClass.TRUE_CONTRADICTION]

    rejection_rate = (
        sum(1 for r in reject if not r.gate_forced_divergent) / len(reject)
        if reject else None
    )
    retention_rate = (
        sum(1 for r in retain if r.gate_forced_divergent) / len(retain)
        if retain else None
    )

    by_class: dict[str, dict[str, int]] = {}
    for r in results:
        bucket = by_class.setdefault(
            r.noise_class.value, {"n": 0, "forced_divergent": 0, "correct": 0}
        )
        bucket["n"] += 1
        bucket["forced_divergent"] += int(r.gate_forced_divergent)
        bucket["correct"] += int(r.correct)

    return StressSummary(
        results=results,
        noise_rejection_rate=rejection_rate,
        true_positive_retention=retention_rate,
        by_class=by_class,
    )


def format_stress_report(summary: StressSummary) -> str:
    lines = ["# Reasoning-layer edge-robustness stress", ""]
    lines.append(f"- noise_rejection_rate: {_fmt(summary.noise_rejection_rate)}")
    lines.append(f"- true_positive_retention: {_fmt(summary.true_positive_retention)}")
    lines.append("")
    lines.append("| class | n | forced_divergent | correct |")
    lines.append("|---|---|---|---|")
    for cls, b in sorted(summary.by_class.items()):
        lines.append(f"| {cls} | {b['n']} | {b['forced_divergent']} | {b['correct']} |")
    lines.append("")
    lines.append("| case | class | forced_divergent | should | ok |")
    lines.append("|---|---|---|---|---|")
    for r in summary.results:
        ok = "✅" if r.correct else "❌"
        lines.append(
            f"| {r.id} | {r.noise_class.value} | {r.gate_forced_divergent} "
            f"| {r.should_force_divergent} | {ok} |"
        )
    return "\n".join(lines)


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":  # pragma: no cover
    print(format_stress_report(run_stress()))
