"""Labeled pair set for Cartographer relation-detector development.

Expanded from the initial 5-pair stub because two relation signals (NLI, zero-shot
LLM) collapsed and five hand-picked pairs cannot separate a real signal from
prompt bias (docs/nc_cartographer_design.plan.md §6). Grounded in real corpus
text across three framing registers, with explicit corroborating and cross-topic
negatives so precision and class-C discrimination are both measurable.

Label provenance:
- DIVERGENT (contradicts): the hand-curated fixture edges — author-labeled
  institutional/technical/corporate framing vs. grassroots/organizing/regulatory
  framing of the SAME event (data/corpus_edges.yaml, data/fixtures/*.yaml).
- CORROBORATES: same-stance, same-topic pairs (e.g. two institutional statements
  about observed warming). Labeled here; criterion is shared stance on one topic.
  Note inst_em corroborates its institutional sibling yet diverges from the
  grassroots framing — relation is a property of the pair, not the chunk.
- UNRELATED: cross-topic pairs the relatedness gate should reject outright.
"""
from __future__ import annotations

from .models import LabeledChunk, Relation

# --- Chunk text (verbatim from the real corpora / framing fixtures) ----------

_TEXT: dict[str, str] = {
    # Climate — institutional (node 1)
    "clim_warming1": (
        "Human activities, principally through emissions of greenhouse gases, have "
        "unequivocally caused global warming, with global surface temperature "
        "reaching 1.1 degrees C above 1850-1900 in 2011-2020."
    ),
    "clim_warming2": (
        "Global surface temperature has increased faster since 1970 than in any "
        "other 50-year period over at least the last 2000 years."
    ),
    "clim_pledges": (
        "Current pledges under the Paris Agreement put the world on track for a "
        "2.5-2.9 degree C temperature rise above pre-industrial levels this century."
    ),
    "inst_em": (
        "Global low-carbon transformations are needed to deliver cuts to predicted "
        "2030 greenhouse gas emissions of roughly 28 percent for a 2 degree C pathway "
        "and 42 percent for a 1.5 degree C pathway."
    ),
    "inst_wf": (
        "In 2023, 56,580 wildfires burned 2,693,910 acres across the United States, "
        "with acreage burned below both the five- and ten-year averages."
    ),
    "inst_wf_fed": (
        "About one-quarter of the nation's wildfires in 2023 occurred on federally "
        "protected lands."
    ),
    "inst_wa": (
        "As of April 3, 2023, California's statewide snowpack held a snow water "
        "equivalent of 61.1 inches, or 237 percent of the April 1 average, one of the "
        "largest snowpacks on record."
    ),
    # Climate — grassroots (node 2)
    "grass_em": (
        "Indigenous-led resistance efforts are estimated to have stopped or delayed "
        "greenhouse gas pollution equivalent to roughly one-quarter of annual U.S. "
        "and Canadian emissions."
    ),
    "grass_wf": (
        "Elderly, immunocompromised, and low-income populations face heightened risk "
        "from wildfire smoke exposure."
    ),
    "grass_wa": (
        "Disadvantaged and cumulatively burdened communities are found to be "
        "disproportionately affected by water shortages, reflecting underlying "
        "inequities in water resource management."
    ),
    # Legal / securities — corporate (node 1) vs regulatory (node 2)
    "disc_nw_pr": (
        "Northwind Systems reported record third quarter revenue of 418 million "
        "dollars, up 31 percent year over year on strong momentum across its cloud "
        "portfolio."
    ),
    "disc_nw_complaint": (
        "The complaint alleges Northwind materially overstated third quarter revenue "
        "by prematurely recognizing 60 million dollars from channel-stuffed "
        "distributor orders that lacked economic substance."
    ),
    "disc_mer_pr": (
        "Meridian Health remains deeply committed to customer trust and maintains "
        "industry-leading safeguards across its patient platform."
    ),
    "disc_mer_complaint": (
        "Regulators allege Meridian concealed a known customer data breach that "
        "exposed 2.1 million patient records for eight months before any notification."
    ),
    # Housing — technical (node 1) vs organizing (node 2)
    "hf_af_staff": (
        "The Alder Flats rezoning application proposes reclassifying twelve parcels "
        "from R-2 to R-4, permitting a floor-area ratio of 2.5 with a density bonus "
        "for on-site affordable units."
    ),
    "hf_af_tenants": (
        "Longtime renters in Alder Flats say the wave of luxury construction is "
        "pushing working families out of the neighborhood they built."
    ),
    "hf_kc_inspection": (
        "Code enforcement cited Kestrel Court apartments for seventeen habitability "
        "violations, including inoperable heating and water intrusion in nine units."
    ),
    "hf_kc_tenants": (
        "Families at Kestrel Court have gone two winters without reliable heat while "
        "repair promises went nowhere."
    ),
}

# (src, dst, relation, register). register groups the per-register breakdown.
_GOLD: list[tuple[str, str, Relation, str]] = [
    # DIVERGENT — institutional/technical/corporate vs grassroots/regulatory framing.
    ("inst_em", "grass_em", Relation.CONTRADICTS, "climate"),
    ("inst_wf", "grass_wf", Relation.CONTRADICTS, "climate"),
    ("inst_wa", "grass_wa", Relation.CONTRADICTS, "climate"),
    ("disc_nw_pr", "disc_nw_complaint", Relation.CONTRADICTS, "legal"),
    ("disc_mer_pr", "disc_mer_complaint", Relation.CONTRADICTS, "legal"),
    ("hf_af_staff", "hf_af_tenants", Relation.CONTRADICTS, "housing"),
    ("hf_kc_inspection", "hf_kc_tenants", Relation.CONTRADICTS, "housing"),
    # CORROBORATES — same stance, same topic (the class-C discrimination cases).
    ("clim_warming1", "clim_warming2", Relation.CORROBORATES, "climate"),
    ("inst_wf", "inst_wf_fed", Relation.CORROBORATES, "climate"),
    ("inst_em", "clim_pledges", Relation.CORROBORATES, "climate"),
    # UNRELATED — cross-topic; the relatedness gate should reject these.
    ("inst_wf", "grass_wa", Relation.UNRELATED, "cross"),
    ("disc_nw_pr", "hf_kc_inspection", Relation.UNRELATED, "cross"),
    ("inst_em", "disc_mer_complaint", Relation.UNRELATED, "cross"),
    # --- Expanded set (17 pairs) for threshold calibration ---
    # CORROBORATES — additional same-stance institutional pairs.
    ("clim_warming1", "clim_pledges", Relation.CORROBORATES, "climate"),
    ("clim_warming2", "inst_em", Relation.CORROBORATES, "climate"),
    ("clim_warming1", "inst_em", Relation.CORROBORATES, "climate"),
    ("clim_pledges", "clim_warming2", Relation.CORROBORATES, "climate"),
    ("inst_wa", "clim_warming2", Relation.CORROBORATES, "climate"),
    ("inst_wa", "clim_warming1", Relation.CORROBORATES, "climate"),
    ("hf_af_staff", "hf_kc_inspection", Relation.CORROBORATES, "housing"),
    # UNRELATED — additional cross-topic negatives.
    ("inst_wf", "disc_nw_pr", Relation.UNRELATED, "cross"),
    ("grass_em", "hf_af_staff", Relation.UNRELATED, "cross"),
    ("clim_warming1", "disc_mer_complaint", Relation.UNRELATED, "cross"),
    ("grass_wa", "disc_nw_pr", Relation.UNRELATED, "cross"),
    ("inst_em", "hf_kc_tenants", Relation.UNRELATED, "cross"),
    ("grass_wf", "clim_pledges", Relation.UNRELATED, "cross"),
    ("disc_mer_pr", "hf_af_tenants", Relation.UNRELATED, "cross"),
    ("inst_wa", "disc_nw_complaint", Relation.UNRELATED, "cross"),
    ("grass_em", "hf_kc_inspection", Relation.UNRELATED, "cross"),
    ("clim_warming2", "hf_af_staff", Relation.UNRELATED, "cross"),
    ("disc_nw_pr", "grass_wa", Relation.UNRELATED, "cross"),
    ("inst_wf", "hf_af_tenants", Relation.UNRELATED, "cross"),
    ("grass_wf", "disc_mer_complaint", Relation.UNRELATED, "cross"),
]


def _chunk_id(local: str) -> str:
    return f"{local}:0"


def chunks() -> list[LabeledChunk]:
    return [LabeledChunk(_chunk_id(k), v) for k, v in _TEXT.items()]


def gold() -> list[tuple[str, str, Relation, str]]:
    """(src_chunk_id, dst_chunk_id, relation, register) tuples."""
    return [(_chunk_id(s), _chunk_id(d), r, reg) for s, d, r, reg in _GOLD]
