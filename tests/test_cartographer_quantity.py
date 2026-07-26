"""Quantity extraction for the same-story stance channel."""
from gin.cartographer.quantity import QuantityMention, extract_mentions, _stem


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
