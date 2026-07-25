"""The frozen evaluation surfaces of the cheap pipeline.

Two things live here, both about what must NOT move:

``eval_pair_keys()`` is the set of pairs that scan_eval/evaluation measure
against. Calibration must exclude them, or the accuracy it reports is partly a
restatement of its own training data.

``BAR_PAIR_IDS`` pins the escalation bar. The bar is pre-registered — four LLM
judges and one learned detector have been scored on exactly these 14 pairs — so
a silent change to it would invalidate every published comparison. The pinning
test is the guard.

Pair identity is ``frozenset((src, dst))``, matching gold_edges'
``gold_contradicts_keys()``. The curator package defines an alternative ``pair_key``
helper, but cartographer deliberately avoids importing it to maintain layering
boundaries: cartographer must never depend on curator.
"""
from __future__ import annotations

from functools import lru_cache

from .gold_edges import gold_pairs
from .labeled_set import gold as labeled_set_gold

# Captured from default_calibration_sets() — issue_frame, then corroboration,
# then unrelated, in list order. Regenerate ONLY when deliberately changing the
# bar, which invalidates prior published comparisons.
BAR_PAIR_IDS: tuple[tuple[str, str, str], ...] = (
    # issue_frame (4)
    ('n1_doc_005:2', 'n2_doc_001:4', 'twonode'),
    ('n1_doc_005:1', 'n2_doc_001:1', 'twonode'),
    ('n1_doc_008:0', 'n2_doc_005:1', 'twonode'),
    ('n1_doc_009:0', 'n2_doc_008:2', 'twonode'),
    # corroboration (6)
    ('n1_doc_008:0', 'n1_doc_008:2', 'twonode'),
    ('labor_bureau_report:0', 'labor_independent_survey:0', 'news'),
    ('wage_bureau_report:0', 'wage_independent_survey:0', 'news'),
    ('inflation_bureau_report:0', 'inflation_independent_survey:0', 'news'),
    ('export_trade_report:0', 'export_independent_review:0', 'news'),
    ('n1_doc_002:0', 'n1_doc_006:2', 'twonode'),
    # unrelated (4)
    ('n1_doc_008:0', 'n2_doc_008:2', 'twonode'),
    ('n1_doc_009:0', 'n2_doc_005:1', 'twonode'),
    ('n1_doc_008:0', 'n1_doc_009:0', 'twonode'),
    ('transit_authority_update:0', 'school_district_report:0', 'news'),
)


@lru_cache(maxsize=1)
def eval_pair_keys() -> frozenset[frozenset[str]]:
    """Pairs the cheap pipeline is EVALUATED on; calibration must exclude them."""
    keys: set[frozenset[str]] = set()
    for src, dst, _relation, _register in labeled_set_gold():
        keys.add(frozenset((src, dst)))
    for pair in gold_pairs():
        keys.add(frozenset((pair.src_chunk_id, pair.dst_chunk_id)))
    return frozenset(keys)
