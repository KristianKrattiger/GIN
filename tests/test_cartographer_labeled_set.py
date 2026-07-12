"""Labeled set composition — the gold the relation detector is developed against.

Expanded from 5 to 13 pairs across three framing registers because two signals
collapsed on the small set (docs/nc_cartographer_design.plan.md §6). Pins size,
composition, and referential integrity so the gold cannot silently drift.
"""
from collections import Counter

from gin.cartographer import default_chunks, default_gold_pairs
from gin.cartographer.models import Relation


def test_set_spans_three_registers_with_negatives():
    gold = default_gold_pairs()
    by_relation = Counter(g.relation.value for g in gold)
    assert by_relation[Relation.CONTRADICTS.value] == 7
    assert by_relation[Relation.CORROBORATES.value] == 10
    assert by_relation[Relation.UNRELATED.value] == 16
    registers = {g.register for g in gold}
    assert {"climate", "legal", "housing"} <= registers


def test_divergent_pairs_cover_every_register():
    """Each framing register contributes at least one divergence."""
    div_registers = {
        g.register for g in default_gold_pairs() if g.relation == Relation.CONTRADICTS
    }
    assert {"climate", "legal", "housing"} <= div_registers


def test_gold_pairs_reference_known_chunks():
    ids = {c.chunk_id for c in default_chunks()}
    for g in default_gold_pairs():
        assert g.src_chunk_id in ids
        assert g.dst_chunk_id in ids


def test_relation_is_a_property_of_the_pair_not_the_chunk():
    """inst_em corroborates its institutional sibling yet diverges from the
    grassroots framing — the same chunk carries different relations."""
    gold = {(g.src_chunk_id, g.dst_chunk_id): g.relation for g in default_gold_pairs()}
    assert gold[("inst_em:0", "grass_em:0")] == Relation.CONTRADICTS
    assert gold[("inst_em:0", "clim_pledges:0")] == Relation.CORROBORATES
