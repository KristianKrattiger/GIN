"""Quantity extraction for the same-story stance channel."""
from gin.cartographer.quantity import QuantityMention, extract_mentions, _stem
from gin.cartographer.quantity import (
    STANCE_PRECEDENCE,
    align,
    evidence_for,
    judge,
    stance_for,
)


def _only(text: str) -> QuantityMention:
    mentions = extract_mentions(text)
    assert len(mentions) == 1, f"expected 1 mention, got {[m.value for m in mentions]}"
    return mentions[0]


def test_stem_folds_verb_and_noun_forms_to_one_stem():
    # "34 people were evacuated" must align with "Evacuations totaled 34".
    assert _stem("evacuated") == _stem("evacuations") == _stem("evacuation")
    assert _stem("voters") == "voter"
    assert _stem("cases") == "case"


def test_extracts_a_plain_count():
    m = _only("Officials confirmed 34 people were evacuated from nearby buildings.")
    assert m.value == 34.0
    assert m.unit_class == "count"
    assert _stem("evacuated") in m.measure
    assert "people" in m.measure
    assert m.revised is False
    assert m.as_of is None


def test_extracts_currency_with_a_scale_word():
    m = _only("Auditors identified an $18 million shortfall in the bond fund's reserves.")
    assert m.value == 18_000_000.0
    assert m.unit_class == "currency"


def test_extracts_percent_and_keeps_points_a_separate_class():
    pct = _only("Turnout was recorded at 47 percent of registered voters.")
    assert (pct.value, pct.unit_class) == (47.0, "percent")
    pts = _only("The referendum passed by a margin of 6 percentage points.")
    assert (pts.value, pts.unit_class) == (6.0, "points")


def test_extracts_speed_area_and_thousands_separators():
    assert _only("Forecasters measured sustained winds at 90 mph.").unit_class == "speed"
    area = _only("The bloom covered about 8.5 square kilometers of the basin.")
    assert (area.value, area.unit_class) == (8.5, "area")
    cnt = _only("The utility said 210,000 customers were without power.")
    assert (cnt.value, cnt.unit_class) == (210_000.0, "count")


def test_extracts_a_date_as_its_own_unit_class():
    m = _only("The bridge will remain closed until at least September 3.")
    assert m.unit_class == "date"
    assert m.value == 903.0   # month * 100 + day, so ordering compares


def test_skips_single_digit_ordinals_that_are_not_measurements():
    # "Ward 3" is a room label, not a quantity. A bare single digit with no
    # currency, unit or scale word carries no measurement.
    mentions = extract_mentions("Ward 3 alone has recorded 21 confirmed cases.")
    assert [m.value for m in mentions] == [21.0]


def test_scope_captures_narrowing_qualifiers():
    wide = _only("Administrators said 34 cases have been confirmed hospital-wide.")
    narrow = _only("Ward 3 alone has recorded 21 confirmed cases as of Thursday.")
    assert wide.scope != narrow.scope
    assert "wide" in wide.scope
    assert "ward" in narrow.scope
    assert narrow.scope == frozenset({"ward", "alone"})
    assert narrow.as_of == 3


def test_scope_excludes_measure_describing_words():
    # "total" and "standing-room" DESCRIBE the measure rather than narrowing it.
    # Treating them as scope turns n5_doc_036 <-> 038 -- a real conflict,
    # 42,000 vs 39,000 total capacity -- into a compatible partial.
    m = _only(
        "The ruling sets the stadium's total capacity, including temporary "
        "standing-room sections, at 42,000 for the coming season."
    )
    assert m.scope == frozenset()


def test_as_of_reads_a_weekday_marker_as_an_ordinal():
    monday = _only("Administrators said 34 cases have been confirmed since Monday.")
    thursday = _only("The hospital reported 58 confirmed cases as of Thursday.")
    assert monday.as_of == 0
    assert thursday.as_of == 3


def test_a_revision_construction_collapses_to_one_revised_mention():
    # Two mentions would let greedy alignment match the STALE value against the
    # other text's figure, score agreement, and hide the revision entirely --
    # on n5_doc_019 <-> 020 (cos 0.993) that produces a confident CORROBORATES
    # for a supersedes pair.
    m = _only(
        "Sustained winds at landfall, initially reported at 90 mph, were "
        "revised to 105 mph after a full review."
    )
    assert m.value == 105.0
    assert m.revised is True


def test_a_bare_initial_estimate_is_not_marked_revised():
    m = _only("The reservoir authority initially estimated the bloom's extent at 8.5 square kilometers.")
    assert m.value == 8.5
    assert m.revised is False


def test_spans_point_at_the_real_text_across_sentences():
    # _SENTENCE_SPLIT consumes a whole whitespace run, so accumulating
    # len(sentence) + 1 desynchronises every span after the first irregular
    # separator. span is documented as the field a rationale quotes from, so a
    # drifted offset quotes the wrong characters entirely.
    text = "Alpha reports 15 cases in the region.\n\nBeta reports 26 cases downtown."
    mentions = extract_mentions(text)
    assert [m.value for m in mentions] == [15.0, 26.0]
    for mention in mentions:
        assert text[mention.span[0]:mention.span[1]] == f"{int(mention.value)}"


def test_span_starts_at_the_number_not_the_preceding_space():
    m = _only("Officials confirmed 34 people were evacuated from nearby buildings.")
    text = "Officials confirmed 34 people were evacuated from nearby buildings."
    assert text[m.span[0]:m.span[1]] == "34"


def test_span_of_a_currency_mention_includes_the_symbol():
    text = "Auditors identified an $18 million shortfall in the bond fund's reserves."
    m = _only(text)
    assert text[m.span[0]:m.span[1]].startswith("$18")


# --- alignment and judgment -------------------------------------------------

# The four kinds, in the corpus's own words.
EVAC_34 = ("RIVERPORT - Fire crews responded to a warehouse blaze on the east "
           "waterfront Tuesday evening. Officials confirmed 34 people were "
           "evacuated from nearby buildings as crews worked to contain the flames.")
EVAC_19 = ("RIVERPORT - Fire crews responded to a warehouse blaze on the east "
           "waterfront Tuesday evening. Officials confirmed 19 people were "
           "evacuated from the surrounding block as smoke spread through the area.")
CASES_34_MON = ("NORTHGATE - Health officials confirmed a gastrointestinal illness "
                "outbreak at Northgate General Hospital this week. Hospital "
                "administrators said 34 cases have been confirmed hospital-wide "
                "since Monday.")
CASES_58_THU = ("NORTHGATE - Health officials confirmed a gastrointestinal illness "
                "outbreak at Northgate General Hospital this week. The hospital "
                "reported 58 confirmed cases hospital-wide as of Thursday, "
                "according to administrators.")
CASES_WARD_21 = ("NORTHGATE - Health officials confirmed a gastrointestinal illness "
                 "outbreak at Northgate General Hospital this week. Ward 3 alone "
                 "has recorded 21 confirmed cases as of Thursday, according to "
                 "hospital records.")


def test_conflict_same_measure_same_scope_different_value():
    assert stance_for(EVAC_34, EVAC_19) == "conflict"


def test_revision_when_a_later_as_of_marker_separates_the_values():
    # No explicit "revised to" here -- only "since Monday" vs "as of Thursday".
    # Without as_of this reads as a conflict, and 3 of the 5 supersedes pairs
    # would be typed CONTRADICTS.
    assert stance_for(CASES_34_MON, CASES_58_THU) == "revision"


def test_partial_when_the_scope_narrows():
    # 34 hospital-wide vs 21 in Ward 3 alone: compatible, not conflicting.
    assert stance_for(CASES_34_MON, CASES_WARD_21) == "partial"


def test_agreement_when_the_aligned_values_match():
    a = "The utility said 210,000 customers were without power."
    b = "The regional utility reported 210,000 customers without power."
    assert stance_for(a, b) == "agreement"


def test_none_when_nothing_aligns():
    a = "The bloom covered about 8.5 square kilometers of the northern basin."
    b = "Jurors awarded the plaintiff $2.4 million in total damages."
    assert stance_for(a, b) is None


def test_precedence_is_conflict_first():
    assert STANCE_PRECEDENCE == ("conflict", "revision", "partial", "agreement")


def test_an_incidental_agreement_cannot_swallow_a_real_conflict():
    # n5_doc_017 <-> 019: agreement on 210,000 customers AND conflict on
    # 65 vs 40 shelters. Conflict must win.
    a = ("CAPE ARDEN - Tropical Storm Elva made landfall near Cape Arden early "
         "Wednesday. Utility officials said roughly 210,000 customers lost power. "
         "Emergency officials said 65 shelters had opened along the coast.")
    b = ("CAPE ARDEN - Tropical Storm Elva made landfall near Cape Arden early "
         "Wednesday. The regional utility reported 210,000 customers without "
         "power. Emergency officials said 40 shelters had opened along the coast.")
    ev = evidence_for(a, b)
    assert ev.conflicts, "the shelter divergence must be found"
    assert ev.agreements, "the customer agreement must also be found"
    assert stance_for(a, b) == "conflict"


def test_align_never_reuses_a_mention():
    a = extract_mentions("Officials said 34 people were evacuated from the block.")
    b = extract_mentions(
        "Officials said 19 people were evacuated from the block. "
        "Officials said 22 people were evacuated from the block."
    )
    pairs = align(a, b)
    assert len(pairs) == 1, "one mention on the left cannot align twice"


def test_judge_is_deterministic_for_each_evidence_kind():
    a = extract_mentions(CASES_34_MON)
    b = extract_mentions(CASES_58_THU)
    pairs = align(a, b)
    assert pairs, "the case counts must align"
    assert judge(pairs[0]) == "revision"


def test_every_supersedes_pair_reads_as_revision_not_agreement():
    """The `agreement` arm is the one place this module makes a POSITIVE claim
    (CORROBORATES) rather than abstaining, so a supersedes pair reaching it is
    worse than one abstaining.

    This is floor-dependent and was measured. At a floor of 0.20 the winds arm
    of n5_doc_019 <-> 020 does not clear alignment, the equal shelter and
    customer counts do, and the pair reads `agreement` -- which at cos 0.993
    would emit a confident CORROBORATES for a revision. At the tuned floor the
    revised fact aligns and all five read `revision`. Pinned so a later floor
    change cannot silently reintroduce that.
    """
    from gin.cartographer.models import Relation
    from gin.curator.node5_labels import node5_pairs, node5_texts

    texts = node5_texts()
    supersedes = [p for p in node5_pairs() if p.relation is Relation.SUPERSEDES]
    assert len(supersedes) == 5
    for pair in supersedes:
        assert stance_for(texts[pair.src], texts[pair.dst]) == "revision", \
            f"{pair.src} <-> {pair.dst}"


def test_align_floor_is_the_value_tuned_on_the_development_events():
    # Tuned on the 13 within-event pairs from the 7 development events only.
    # The 3 held-out events (lakeshore_algae_bloom, civic_bond_audit,
    # stadium_capacity_ruling) were not consulted. Changing this value means
    # re-running that measurement, not nudging the constant.
    from gin.cartographer.quantity import ALIGN_FLOOR
    assert ALIGN_FLOOR == 0.05
