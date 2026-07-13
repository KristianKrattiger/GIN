"""Load hand-curated gold contradicts edges for scan-vs-gold evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import yaml

from .evaluation import GoldPair, _key
from .models import Relation

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_GOLD_SOURCES = (
    ROOT / "data" / "corpus_edges.yaml",
    ROOT / "data" / "fixtures" / "disclosure_framing.yaml",
    ROOT / "data" / "fixtures" / "housing_framing.yaml",
    ROOT / "data" / "fixtures" / "wildfire_multipara.yaml",
    ROOT / "data" / "synthetic" / "news_corpus.yaml",
)

_REGISTER_BY_SOURCE: dict[str, str] = {
    "corpus_edges.yaml": "twonode",
    "disclosure_framing.yaml": "legal",
    "housing_framing.yaml": "housing",
    "wildfire_multipara.yaml": "multipara",
    "news_corpus.yaml": "news",
}


@dataclass(frozen=True)
class GoldContradictsEdge:
    src_chunk_id: str
    dst_chunk_id: str
    register: str
    note: str = ""
    # "story": same-story conflict, machine-recoverable by the scan.
    # "issue_frame": same issue, opposing frames, no shared story entities —
    # machine-undetectable (2026-07-12 signal audit); curated ingest only.
    relation_class: str = "story"


def _register_for(path: Path) -> str:
    return _REGISTER_BY_SOURCE.get(path.name, path.stem)


def load_gold_contradicts(path: Path) -> list[GoldContradictsEdge]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    reg = _register_for(path)
    out: list[GoldContradictsEdge] = []
    for edge in data.get("edges", []):
        if edge.get("type") != "contradicts":
            continue
        out.append(
            GoldContradictsEdge(
                src_chunk_id=edge["src"],
                dst_chunk_id=edge["dst"],
                register=reg,
                note=edge.get("note", ""),
                relation_class=edge.get("relation_class", "story"),
            )
        )
    return out


def load_all_gold_contradicts(
    sources: Optional[Iterable[Path]] = None,
) -> list[GoldContradictsEdge]:
    paths = list(sources) if sources is not None else list(DEFAULT_GOLD_SOURCES)
    edges: list[GoldContradictsEdge] = []
    for path in paths:
        if path.is_file():
            edges.extend(load_gold_contradicts(path))
    return edges


def gold_pairs(sources: Optional[Iterable[Path]] = None) -> list[GoldPair]:
    return [
        GoldPair(
            e.src_chunk_id,
            e.dst_chunk_id,
            Relation.CONTRADICTS,
            e.register,
            relation_class=e.relation_class,
        )
        for e in load_all_gold_contradicts(sources)
    ]


def gold_contradicts_keys(sources: Optional[Iterable[Path]] = None) -> set[frozenset]:
    return {_key(e.src_chunk_id, e.dst_chunk_id) for e in load_all_gold_contradicts(sources)}


# Corroborating pairs that must NOT be typed contradicts (class-C control).
CLASS_C_CONTROLS: tuple[tuple[str, str, str], ...] = (
    ("n1_doc_008:0", "n1_doc_008:2", "twonode"),
    # Identical unemployment stats; independent closing line challenges
    # interpretation, not any asserted fact (corroboration-with-a-caveat).
    ("labor_bureau_report:0", "labor_independent_survey:0", "news"),
)


# --- Escalation-judge calibration extras -------------------------------------
# Deliberately SEPARATE from CLASS_C_CONTROLS: scan_eval counts its class-C
# metric over that tuple and the denominator is pinned by closed label
# decisions. These pairs are consumed only by escalation_eval.

# Cross-source corroboration (want NOT DIVERGENT). The wage/inflation/export
# pairs mirror the labor control — identical stats, closing line challenges
# interpretation only — the exact sub-pattern the one-word judge failed.
# The NOAA/WMO pair is clean caveat-free corroboration across sources.
ESCALATION_CLASS_C_EXTRA: tuple[tuple[str, str, str], ...] = (
    ("wage_bureau_report:0", "wage_independent_survey:0", "news"),
    ("inflation_bureau_report:0", "inflation_independent_survey:0", "news"),
    ("export_trade_report:0", "export_independent_review:0", "news"),
    ("n1_doc_002:0", "n1_doc_006:2", "twonode"),  # NOAA & WMO: 2023 warmest year
)

# Topically-close cross-issue pairs (want NOT DIVERGENT; ideally UNRELATED).
# The first two reuse the issue_frame gold texts with the pairings CROSSED
# (wildfire-institutional vs water-justice and vice versa), so the register
# axis is present while the issue does not match — a judge that fires on
# register alone fails exactly here.
ESCALATION_UNRELATED_CONTROLS: tuple[tuple[str, str, str], ...] = (
    ("n1_doc_008:0", "n2_doc_008:2", "twonode"),  # wildfire acreage vs water equity
    ("n1_doc_009:0", "n2_doc_005:1", "twonode"),  # snowpack vs wildfire smoke equity
    ("n1_doc_008:0", "n1_doc_009:0", "twonode"),  # both institutional, cross-issue
    ("transit_authority_update:0", "school_district_report:0", "news"),
)
