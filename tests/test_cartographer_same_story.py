"""Same-story precondition for contradicts typing (scan-scale precision fix).

The mid-band-default-contradicts rule inverted at scan scale (run
20260712T074956Z: 120 FPs, 2/8 gold admitted): measured on the 136-chunk DB,
true framing divergences sit ABOVE the corroborate ceiling (kestrel cos 0.698,
meridian 0.546, wf_multi 0.567) while the mid band is cross-topic noise
(cos 0.36-0.48, p_contra <= 0.1). The one signal that separates them is shared
rare entities: every recoverable gold pair shares >= 2 rare tokens (the story
entity), 22/24 sampled FPs share <= 1.

Same-story is a stage-1 relatedness claim (design §2: stage 1 MAY use IDF /
shared entities), threaded into stage-2 classification as a precondition for
any contradicts typing. The mid-band default flips to RELATED_UNTYPED.
"""
import pytest

from gin.cartographer.combined import (
    CombinedRelationProposer,
    Thresholds,
    classify_relation,
)
from gin.cartographer.models import LabeledChunk, Relation
from gin.cartographer.relatedness import (
    CALENDAR_WORDS,
    anchor_tokens,
    make_same_story,
    shared_rare_token_count,
)

T = Thresholds(gate_floor=0.14, corroborate_ceiling=0.485, contra_threshold=0.686)


# --- stage-1 same-story helper ----------------------------------------------


def test_shared_rare_token_count_counts_tokens_rare_in_corpus():
    corpus = [
        "the alderflats tower inspection found mold",
        "alderflats tenants say mold complaints were ignored",
        "the housing office issued permits downtown",
        "downtown transit ridership rose this quarter",
        "the port authority reported cargo throughput this quarter",
    ]
    # "alderflats" and "mold" each appear in exactly 2 docs -> rare; "the",
    # "downtown", "quarter" spread wider or are shared boilerplate.
    n = shared_rare_token_count(corpus[0], corpus[1], corpus)
    assert n >= 2


def test_shared_rare_token_count_zero_for_boilerplate_overlap():
    corpus = [
        "the bureau report said exports fell three percent this quarter",
        "the bureau report said wages rose two percent this quarter",
        "the bureau report said inflation held at four percent this quarter",
        "the bureau report said ridership fell one percent this quarter",
    ]
    # Every shared token appears in all four docs -> nothing rare is shared.
    assert shared_rare_token_count(corpus[0], corpus[1], corpus) == 0


def test_make_same_story_requires_two_shared_rare_tokens():
    corpus = [
        "The Kestrel Court inspection cited heating failures.",
        "At Kestrel Court, tenants reported heating outages all winter.",
        "The election margin was reported at two points.",
        "Port cargo throughput rose in the spring.",
    ]
    same_story = make_same_story(corpus)
    assert same_story(corpus[0], corpus[1]) is True
    assert same_story(corpus[2], corpus[3]) is False


def test_same_story_requires_an_anchor_token():
    """Corpus-rare boilerplate is not a story: 'remain in effect' drove the
    water/weather scan FP (run 20260712T091415Z) — two unrelated advisories
    sharing lowercase phrasing must not count as one story."""
    corpus = [
        "Conservation appeals remain in effect for suburban districts.",
        "Flood watches remain in effect for low-lying neighborhoods.",
        "The city budget passed after a second reading.",
        "Harvest totals rose across the northern valley.",
    ]
    same_story = make_same_story(corpus)
    assert same_story(corpus[0], corpus[1]) is False


def test_sentence_initial_capitalization_is_not_an_anchor():
    """'Combined reservoir storage stood...' — sentence-initial caps carry no
    entity signal; both texts share rare tokens but no true anchor."""
    corpus = [
        "Combined reservoir storage stood at 74 percent of capacity.",
        "Combined reservoir storage gained water after the storm.",
        "The transit line opened early on weekdays.",
        "Retail hiring slowed before the holidays.",
    ]
    same_story = make_same_story(corpus)
    assert same_story(corpus[0], corpus[1]) is False


def test_all_caps_dateline_is_an_anchor():
    corpus = [
        "RIVERPORT - Officials responded to a downtown incident on Tuesday evening.",
        "RIVERPORT - Emergency crews treated dozens after the downtown incident.",
        "The port authority reported cargo throughput this quarter.",
        "Election turnout reached record levels in the spring.",
    ]
    same_story = make_same_story(corpus)
    assert same_story(corpus[0], corpus[1]) is True


def test_multi_digit_number_is_an_anchor_but_decimal_fragment_is_not():
    corpus = [
        "Police said 11 arrests were made before midnight near the plaza gates.",
        "Union leaders disputed that 11 arrests were made near the plaza gates.",
        "Earnings grew 4.8 percent while hospitality pay lagged notably.",
        "Earnings grew 4.8 percent while hospitality hiring slowed notably.",
        "The museum extended weekend hours through the fall season.",
        "Cyclists asked for protected lanes along the river route.",
    ]
    same_story = make_same_story(corpus)
    # '11' + 'plaza'/'gates' anchor the arrest story.
    assert same_story(corpus[0], corpus[1]) is True
    # '4'/'8' are decimal fragments; 'hospitality'/'earnings' are lowercase.
    assert same_story(corpus[2], corpus[3]) is False


def test_union_anchor_collision_is_not_a_story():
    """Variant D (stage-1 anchor findings): the anchor must be entity-grade in
    one text and at least capitalized in the other. 'Union Yard' (proper noun)
    colliding with 'the union local' (common noun) was the collision that held
    n5_doc_023 to n5_doc_026 — the last node5 cross-event false positive."""
    corpus = [
        "Crews staged at Union Yard while pickets formed along the fence.",
        "The union local said pickets would continue through the weekend.",
        "The city budget passed after a second reading.",
        "Harvest totals rose across the northern valley.",
    ]
    same_story = make_same_story(corpus)
    assert same_story(corpus[0], corpus[1]) is False


def test_sentence_initial_entity_still_anchors_against_midsentence_use():
    """The legal-register pairs the withdrawn intersection fix regressed:
    'Northwind Systems reported...' is sentence-initial (not entity-grade) but
    still capitalized, and the complaint names Northwind mid-sentence. Variant
    D must keep this pair alive — entity-grade on one side, capitalized on the
    other."""
    corpus = [
        "Northwind Systems reported record third quarter revenue this week.",
        "The complaint alleges Northwind materially overstated quarterly revenue.",
        "The transit line opened early on weekdays.",
        "Retail hiring slowed before the holidays.",
    ]
    same_story = make_same_story(corpus)
    assert same_story(corpus[0], corpus[1]) is True


# --- stage-2 classification with the story precondition ----------------------


def test_mid_band_without_story_is_related_untyped_not_contradicts():
    """The scan-scale flood: cross-topic pairs at cos 0.36-0.48, low p_contra."""
    rel, channel = classify_relation(0.408, 0.007, T, same_story=False)
    assert rel == Relation.RELATED_UNTYPED
    assert channel == "band"


def test_mid_band_with_story_is_contradicts():
    """Recovered gold alderflats: cos 0.379, p_contra 0.159, shared story."""
    rel, channel = classify_relation(0.379, 0.159, T, same_story=True)
    assert rel == Relation.CONTRADICTS
    assert channel == "band"


def test_high_cos_with_story_is_contradicts_not_corroborates():
    """Missed gold kestrel: cos 0.698 sat above the ceiling and was mistyped
    corroborates; same-story framing pairs are exactly this similar."""
    rel, channel = classify_relation(0.698, 0.004, T, same_story=True)
    assert rel == Relation.CONTRADICTS
    assert channel == "band"


def test_high_cos_without_story_is_corroborates():
    rel, channel = classify_relation(0.654, 0.025, T, same_story=False)
    assert rel == Relation.CORROBORATES
    assert channel == "band"


def test_nli_without_story_is_not_contradicts():
    """Scan NLI FPs: northwind PR vs export report, p_contra 0.977 across
    topics — the cross-encoder numeric artifact."""
    rel, _channel = classify_relation(0.235, 0.977, T, same_story=False)
    assert rel != Relation.CONTRADICTS


def test_nli_with_story_is_contradicts():
    """Gold northwind PR vs its own complaint: p_contra 0.991, shared story."""
    rel, channel = classify_relation(0.603, 0.991, T, same_story=True)
    assert rel == Relation.CONTRADICTS
    assert channel == "nli"


def test_nli_without_story_evidence_still_fires():
    """No stage-1 provider wired (same_story=None): the propositional channel
    stands on its own; only the band channel needs story evidence."""
    rel, channel = classify_relation(0.552, 0.899, T, same_story=None)
    assert rel == Relation.CONTRADICTS
    assert channel == "nli"


def test_mid_band_without_story_evidence_is_related_untyped():
    rel, _channel = classify_relation(0.40, 0.05, T, same_story=None)
    assert rel == Relation.RELATED_UNTYPED


def test_gate_still_wins_over_story():
    rel, channel = classify_relation(0.10, 0.05, T, same_story=True)
    assert rel == Relation.UNRELATED
    assert channel == "gate"


# --- proposer integration -----------------------------------------------------


def _proposer_with(cos: float, p_contra: float, story: bool) -> CombinedRelationProposer:
    return CombinedRelationProposer(
        embed_cos=lambda a, b: cos,
        nli_scores=lambda a, b: (p_contra, 0.001, 1.0 - p_contra),
        same_story=lambda a, b: story,
        thresholds=T,
    )


def test_proposer_threads_story_signal_into_assessment():
    a = LabeledChunk("kestrel_inspection:0", "inspection cited heating failures")
    b = LabeledChunk("kestrel_tenants:0", "tenants reported heating outages")
    prop = _proposer_with(0.698, 0.004, story=True)
    assessment = prop.assess_pair(a, b)
    assert assessment.relation == Relation.CONTRADICTS
    assert assessment.method.endswith(":band")


def test_proposer_without_story_does_not_type_band_contradicts():
    a = LabeledChunk("export:0", "exports fell three percent")
    b = LabeledChunk("labor:0", "wages rose two percent")
    prop = _proposer_with(0.408, 0.007, story=False)
    assessment = prop.assess_pair(a, b)
    assert assessment.relation == Relation.RELATED_UNTYPED


# --- scan wiring ---------------------------------------------------------------


_SCAN_CHUNKS = [
    LabeledChunk("kestrel_inspection:0", "The Kestrel Court inspection cited mold and heating failures."),
    LabeledChunk("kestrel_tenants:0", "At Kestrel Court, tenants say mold complaints were ignored."),
    LabeledChunk("port:0", "Port cargo throughput rose this quarter."),
    LabeledChunk("election:0", "The election margin was two points."),
]


def test_wire_same_story_builds_provider_from_scanned_chunks():
    from gin.cartographer.scan import wire_same_story

    prop = CombinedRelationProposer(
        embed_cos=lambda a, b: 0.5,
        nli_scores=lambda a, b: (0.0, 0.0, 1.0),
        thresholds=T,
    )
    wire_same_story(prop, _SCAN_CHUNKS)
    assert prop.same_story is not None
    assert prop.same_story(_SCAN_CHUNKS[0].text, _SCAN_CHUNKS[1].text) is True
    assert prop.same_story(_SCAN_CHUNKS[2].text, _SCAN_CHUNKS[3].text) is False


def test_wire_same_story_keeps_an_injected_provider():
    from gin.cartographer.scan import wire_same_story

    injected = lambda a, b: True  # noqa: E731
    prop = CombinedRelationProposer(
        embed_cos=lambda a, b: 0.5,
        nli_scores=lambda a, b: (0.0, 0.0, 1.0),
        same_story=injected,
        thresholds=T,
    )
    wire_same_story(prop, _SCAN_CHUNKS)
    assert prop.same_story is injected


def test_wired_scan_proposals_are_story_gated():
    """End to end over in-memory chunks: the same-story pair (even above the
    old corroborate ceiling) yields a contradicts proposal; the cross-topic
    mid-band pair yields none."""
    from gin.cartographer.scan import proposals_from_pairs, wire_same_story

    cos_table = {
        frozenset({"kestrel_inspection:0", "kestrel_tenants:0"}): 0.698,
        frozenset({"port:0", "election:0"}): 0.408,
    }
    by_text = {c.text: c.chunk_id for c in _SCAN_CHUNKS}
    prop = CombinedRelationProposer(
        embed_cos=lambda a, b: cos_table[frozenset({by_text[a], by_text[b]})],
        nli_scores=lambda a, b: (0.01, 0.001, 0.989),
        thresholds=T,
    )
    wire_same_story(prop, _SCAN_CHUNKS)
    pairs = [
        (_SCAN_CHUNKS[0], _SCAN_CHUNKS[1]),
        (_SCAN_CHUNKS[2], _SCAN_CHUNKS[3]),
    ]
    proposals = proposals_from_pairs(prop, pairs)
    keys = {frozenset({p.src_chunk_id, p.dst_chunk_id}): p for p in proposals}
    kestrel = keys.get(frozenset({"kestrel_inspection:0", "kestrel_tenants:0"}))
    assert kestrel is not None and kestrel.relation == Relation.CONTRADICTS
    assert frozenset({"port:0", "election:0"}) not in keys


# --- calendar word exclusion (Task 4) -----------------------------------------


def test_anchor_tokens_rejects_mid_sentence_weekdays():
    # anchor_tokens tests mid-sentence capitalization as a proxy for proper
    # nouns, and every weekday and month in English prose satisfies it. On the
    # node5 labels "Monday" was the ONLY anchor holding n5_doc_007 (a hospital
    # outbreak) to n5_doc_012 (a bridge closure) -- a calendar word anchoring
    # a story.
    text = "Engineers closed the Sable Bridge after inspectors found cracking Monday."
    tokens = anchor_tokens(text)
    assert "sable" in tokens
    assert "bridge" in tokens
    assert "monday" not in tokens


def test_anchor_tokens_rejects_mid_sentence_months():
    text = "Officials said the bridge will remain closed until at least September 3."
    tokens = anchor_tokens(text)
    assert "september" not in tokens
    # A multi-digit number is still a story figure; a bare "3" was never
    # entity-grade (the len >= 2 digit rule), so nothing is asserted about it.


def test_calendar_words_covers_weekdays_and_months():
    assert len(CALENDAR_WORDS) == 19
    for word in ("monday", "sunday", "january", "may", "december"):
        assert word in CALENDAR_WORDS
