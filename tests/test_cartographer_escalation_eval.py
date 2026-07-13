"""Escalation-judge calibration eval — issue_frame gold + expanded controls.

Deliberately separate from scan_eval: its CLASS_C metric denominator is pinned
by closed label decisions, so calibration extras live in their own tuples and
are consumed only here.
"""
from gin.cartographer.escalation_eval import (
    default_calibration_sets,
    evaluate_escalation_judge,
    labeled_set_pairs,
)
from gin.cartographer.gold_edges import (
    CLASS_C_CONTROLS,
    ESCALATION_CLASS_C_EXTRA,
    ESCALATION_UNRELATED_CONTROLS,
)

TEXTS = {
    "g1:0": "gold one",
    "g2:0": "gold two",
    "c1:0": "corr a",
    "c2:0": "corr b",
    "u1:0": "unrel a",
    "u2:0": "unrel b",
}
IF = [("g1:0", "g2:0", "twonode")]
CC = [("c1:0", "c2:0", "news")]
UN = [("u1:0", "u2:0", "news")]


def _eval(judge, **kw):
    return evaluate_escalation_judge(
        judge,
        TEXTS,
        issue_frame_pairs=IF,
        corroboration_pairs=CC,
        unrelated_pairs=UN,
        **kw,
    )


def test_calibration_extras_are_separate_from_closed_class_c():
    # scan_eval._class_c_from_proposals counts over CLASS_C_CONTROLS; that
    # denominator is pinned by closed label decisions. Extras must not leak in.
    assert len(CLASS_C_CONTROLS) == 2
    base = {frozenset({s, d}) for s, d, _ in CLASS_C_CONTROLS}
    extra = {frozenset({s, d}) for s, d, _ in ESCALATION_CLASS_C_EXTRA}
    unrel = {frozenset({s, d}) for s, d, _ in ESCALATION_UNRELATED_CONTROLS}
    assert not (base & extra) and not (base & unrel) and not (extra & unrel)
    assert len(ESCALATION_CLASS_C_EXTRA) >= 3
    assert len(ESCALATION_UNRELATED_CONTROLS) >= 3


def test_default_calibration_sets_cover_all_classes():
    sets = default_calibration_sets()
    assert len(sets["issue_frame"]) == 4
    assert len(sets["corroboration"]) == len(CLASS_C_CONTROLS) + len(
        ESCALATION_CLASS_C_EXTRA
    )
    assert len(sets["unrelated"]) == len(ESCALATION_UNRELATED_CONTROLS)


def test_oracle_judge_hits_all_bars():
    def judge(a, b):
        if {a, b} == {"gold one", "gold two"}:
            return "DIVERGENT"
        if {a, b} == {"corr a", "corr b"}:
            return "AGREE"
        return "UNRELATED"

    m = _eval(judge)
    assert m["issue_frame_recall"] == 1.0
    assert m["class_c_discrimination"] == 1.0
    assert m["unrelated_discrimination"] == 1.0
    assert m["direction_flip_count"] == 0


def test_constant_divergent_judge_fails_both_discriminations():
    m = _eval(lambda a, b: "DIVERGENT")
    assert m["issue_frame_recall"] == 1.0  # trivially — everything is divergent
    assert m["class_c_discrimination"] == 0.0
    assert m["unrelated_discrimination"] == 0.0
    # Forward-direction labels only, so runs stay comparable.
    assert m["label_distribution"] == {"DIVERGENT": 3}


def test_direction_flips_are_counted():
    def judge(a, b):
        return "DIVERGENT" if a == "gold one" else "AGREE"

    m = _eval(judge)
    assert m["direction_flip_count"] == 1
    gold_row = m["gold_labels"][0]
    assert gold_row["label"] == "DIVERGENT"
    assert gold_row["label_reverse"] == "AGREE"
    assert gold_row["flip"] is True
    # Primary metric scores the forward direction (matches production's call).
    assert m["issue_frame_recall"] == 1.0


def test_single_direction_mode_skips_reverse():
    m = _eval(lambda a, b: "DIVERGENT", both_directions=False)
    assert m["direction_flip_count"] == 0
    assert m["gold_labels"][0]["label_reverse"] is None


def test_missing_chunk_rows_are_skipped():
    m = evaluate_escalation_judge(
        lambda a, b: "DIVERGENT",
        TEXTS,
        issue_frame_pairs=[("nope:0", "g2:0", "twonode")],
        corroboration_pairs=[],
        unrelated_pairs=[],
    )
    assert m["issue_frame_scorable_count"] == 0
    assert m["issue_frame_recall"] is None
    assert m["gold_labels"][0]["skipped"] == "chunk_not_in_db"


def test_reasoning_text_is_recorded_when_judge_exposes_it():
    class VerboseJudge:
        def __init__(self):
            self.last_completion_text = None

        def __call__(self, a, b):
            self.last_completion_text = f"weighing {a[:6]}…\nFINAL: AGREE"
            return "AGREE"

    m = _eval(VerboseJudge())
    row = m["class_c_labels"][0]
    assert "FINAL: AGREE" in row["reasoning"]


def test_labeled_set_block_scores_per_expected_class():
    pairs = labeled_set_pairs()
    assert len(pairs) == 33
    by_pair = {frozenset({a, b}): want for a, b, want in pairs}

    def oracle(a, b):
        return by_pair[frozenset({a, b})]

    m = evaluate_escalation_judge(
        oracle,
        {},
        issue_frame_pairs=[],
        corroboration_pairs=[],
        unrelated_pairs=[],
        labeled_pairs=pairs,
    )
    ls = m["labeled_set"]
    assert ls["total"] == 33
    assert ls["accuracy"] == 1.0
    assert ls["by_expected"]["DIVERGENT"]["total"] == 7
    assert ls["by_expected"]["AGREE"]["total"] == 10
    assert ls["by_expected"]["UNRELATED"]["total"] == 16


def test_labeled_set_exposes_constant_answer_collapse():
    pairs = labeled_set_pairs()
    m = evaluate_escalation_judge(
        lambda a, b: "DIVERGENT",
        {},
        issue_frame_pairs=[],
        corroboration_pairs=[],
        unrelated_pairs=[],
        labeled_pairs=pairs,
    )
    ls = m["labeled_set"]
    assert ls["by_expected"]["DIVERGENT"]["correct"] == 7
    assert ls["by_expected"]["AGREE"]["correct"] == 0
    assert ls["by_expected"]["UNRELATED"]["correct"] == 0
    assert ls["accuracy"] == 7 / 33
