"""Tests for gin.assay_proposer.consistency — one relation type per claimant/independent quote pair."""
from gin.assay_proposer.consistency import conflicting_pairs


def _p(type_: str, from_quote: str = "Tesla plans to widen availability further next year.",
       to_quote: str | None = "Tesla Full Self Driving requires human intervention every 13 miles") -> dict:
    return {
        "type": type_,
        "from": {"docId": "tesla", "quote": from_quote},
        "to": None if to_quote is None else {"docId": "hn", "quote": to_quote},
    }


def test_two_types_on_one_pair_conflict():
    conflicts = conflicting_pairs([_p("updates"), _p("corroborates")])
    assert len(conflicts) == 1
    assert conflicts[0]["types"] == ["corroborates", "updates"]
    assert conflicts[0]["from"] == {"docId": "tesla", "quote": "Tesla plans to widen availability further next year."}


def test_same_type_twice_is_a_duplicate_not_a_conflict():
    assert conflicting_pairs([_p("contradicts"), _p("contradicts")]) == []


def test_different_pairs_do_not_conflict():
    assert conflicting_pairs([_p("contradicts"), _p("corroborates", to_quote="Other line")]) == []


def test_unsupported_has_no_pair_to_conflict_on():
    assert conflicting_pairs([_p("unsupported", to_quote=None), _p("contradicts")]) == []


def test_quote_whitespace_does_not_hide_a_conflict():
    a = _p("contradicts", from_quote="Tesla plans to widen availability further next year. ")
    assert len(conflicting_pairs([a, _p("corroborates")])) == 1


def test_three_types_on_one_pair_report_once():
    conflicts = conflicting_pairs([_p("updates"), _p("corroborates"), _p("contradicts")])
    assert [c["types"] for c in conflicts] == [["contradicts", "corroborates", "updates"]]
