"""Model-free quantity-stance evidence for same-story pairs.

`combined.py` typed ANY same-story pair CONTRADICTS on story membership alone,
with no stance evidence. The 24 node5 curator labels (2026-07-26) put that
branch's precision at 12/24, and the NLI channel cannot replace it: measured
over those pairs at the shipped contra_threshold, the two highest p_contra
scores in the whole set are a `corroborates` (0.983) and a `supersedes`
(0.980), above every real conflict but two.

Reading the 19 within-event texts, the discriminator is per-fact and
structural. ALL 19 contain a numeric divergence, so "numbers differ ->
contradicts" also scores 12/19 and changes nothing. What separates them:

  conflict      same measure, same scope, different value
                "34 people were evacuated" / "19 people were evacuated"
  supersedes    a revision marker or a later as-of marker ON THAT FACT
                "initially reported at 8.5 ... revised to 12";
                "since Monday" -> "as of Thursday"
  corroborates  the numbers attach to DIFFERENT measures or scopes
                "total capacity incl. standing-room 42,000" /
                "fixed seats in the bowl 36,500"

Two of the 12 conflicts (n5_doc_005<->006, 017<->020) carry revision language
on a fact OTHER than the conflicting one, so a pair-level revision veto costs
real conflicts and a fact-aligned one does not. That is why this module aligns
before it judges.

No models, no network, no corpus statistics, no I/O -- the relation-type stage
may not use relevance signals (design section 2), and this uses none.

Spec: docs/superpowers/specs/2026-07-26-same-story-stance-channel-design.md
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# --- vocabularies, reviewed as data -----------------------------------------

CALENDAR_ORDINALS: dict[str, int] = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

_MONTH_ORDINALS: dict[str, int] = {
    name: i
    for i, name in enumerate(
        "january february march april may june july august september october "
        "november december".split(),
        start=1,
    )
}

# Qualifiers that genuinely change a measure's DENOMINATOR, so two figures
# carrying different ones are not in conflict.
#
# Deliberately EXCLUDES "total", "standing-room", "fixed", "permanent" and
# "at the port itself". On the labeled pairs those describe the measure rather
# than narrowing it: treating "standing-room" as scope turns n5_doc_036 <-> 038
# (a real conflict, 42,000 vs 39,000 total capacity) into a compatible partial,
# and treating "at the port itself" as scope does the same to 022 <-> 023
# (650 vs 420 dockworkers).
SCOPE_TOKENS = frozenset({
    "wide",        # hospital-wide, city-wide (tokenizes to two words)
    "ward", "alone",
    "citywide", "downtown",
    "nationwide", "statewide", "country",
})

_SCALE = {"thousand": 1_000.0, "million": 1_000_000.0, "billion": 1_000_000_000.0}

_STOPWORDS = frozenset("""
a an the and or but of in on at to for from by with without as is are was were
be been being has have had said say says it its this that these those they
their there then than not no nor so if while during after before over under
about up out off down more most less least new newly than which who whom whose
will would can could may might must shall should do does did done also very
""".split())

# --- primitives -------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT = re.compile(r"[,;]")

_SUFFIXES = ("ions", "ion", "ings", "ing", "ed", "es", "s")


def _stem(word: str) -> str:
    """Crude suffix stripper, ordered so verb and noun forms land together.

    "evacuated" -> "evacuat" (drop "ed"); "evacuations" -> "evacuat" (drop
    "ions", checked before "s"); "evacuation" -> "evacuat" (drop "ion"). The
    order matters: checking "s" first would give "evacuation" and break the
    n5_doc_002 <-> 003 alignment, which is the pair that needs it.
    """
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _content(words) -> set[str]:
    return {
        _stem(w) for w in words
        if len(w) > 2 and not w.isdigit() and w not in _STOPWORDS
    }


_NUMBER = re.compile(
    r"(?P<currency>\$)?\s*"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s+(?P<scale>thousand|million|billion))?"
    r"(?:\s+(?P<unit>percentage\s+points?|percent|points?|mph|"
    r"square\s+kilometers?|kilometers?))?"
    r"|(?P<pct>%)",
    re.IGNORECASE,
)

_DATE = re.compile(
    r"\b(?P<month>january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+(?P<day>\d{1,2})\b",
    re.IGNORECASE,
)

REVISED_TO = re.compile(r"\b(?:revised|updated)\s+to\b", re.IGNORECASE)
_AS_OF = re.compile(
    r"\b(?:as\s+of|since|by|through)\s+(?P<day>monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QuantityMention:
    value: float
    unit_class: str            # count | currency | percent | points | speed | area | date
    measure: frozenset[str]    # content tokens governing the numeral
    scope: frozenset[str]      # narrowing qualifiers from SCOPE_TOKENS
    revised: bool              # sits in a "revised to X" construction
    as_of: Optional[int]       # weekday ordinal of the clause's temporal marker
    span: tuple[int, int]      # offsets in the full text, for rationales


def _unit_class(currency: Optional[str], scale: Optional[str], unit: Optional[str]) -> str:
    if currency:
        return "currency"
    if unit is None:
        return "count"
    u = " ".join(unit.lower().split())
    if u in {"percent", "%"}:
        return "percent"
    if u.startswith("percentage point") or u.startswith("point"):
        return "points"
    if u == "mph":
        return "speed"
    if u.startswith("square kilometer") or u.startswith("kilometer"):
        return "area"
    return "count"


def _window_words(sentence: str, start: int, window: int) -> list[str]:
    """Lowercased word tokens within +/-``window`` positions of offset ``start``.

    Shared by the measure and scope extractors, which differ only in window size
    and in what they do with the result.
    """
    lowered = sentence.lower()
    spans = [m.span() for m in _WORD.finditer(lowered)]
    words = [lowered[a:b] for a, b in spans]
    idx = next((i for i, (a, _b) in enumerate(spans) if a >= start), len(words))
    return words[max(0, idx - window): idx + window + 1]


def _measure_tokens(sentence: str, start: int, end: int, window: int = 5) -> frozenset[str]:
    """Content tokens governing the numeral: its clause UNIONED with a +/-window
    token span that crosses clause boundaries.

    Neither alone works on the labeled pairs. The clause alone loses "total
    capacity" in "...total capacity, including temporary standing-room
    sections, at 42,000..." -- the numeral sits in a trailing clause. The window
    alone loses heads that sit further out. The union keeps both, at the cost of
    a looser measure; ALIGN_FLOOR (Task 8) is what compensates.
    """
    bounds = [0]
    for m in _CLAUSE_SPLIT.finditer(sentence):
        bounds.extend((m.start(), m.end()))
    bounds.append(len(sentence))
    clause = sentence
    for i in range(0, len(bounds) - 1, 2):
        lo, hi = bounds[i], bounds[i + 1]
        if lo <= start < hi:
            clause = sentence[lo:hi]
            break

    near = _window_words(sentence, start, window)

    return frozenset(_content(_WORD.findall(clause.lower())) | _content(near))


def _scope_tokens(sentence: str, start: int, window: int = 6) -> frozenset[str]:
    return frozenset(w for w in _window_words(sentence, start, window) if w in SCOPE_TOKENS)


def extract_mentions(text: str) -> tuple[QuantityMention, ...]:
    """Every quantity mention in ``text``, in order of appearance.

    A bare single digit with no currency, scale word or unit is skipped: "Ward
    3" is a room label, not a measurement.
    """
    out: list[QuantityMention] = []
    cursor = 0
    for sentence in _SENTENCE_SPLIT.split(text):
        if not sentence.strip():
            continue
        # Recover the sentence's TRUE offset instead of accumulating lengths:
        # _SENTENCE_SPLIT consumes a whole whitespace run, which is one char for
        # "a. b" but two for a double space and two for a "\n\n" paragraph break.
        # Adding len(sentence) + 1 desynchronises every following span.
        offset = text.index(sentence, cursor)
        cursor = offset + len(sentence)

        as_of_match = _AS_OF.search(sentence)
        as_of = CALENDAR_ORDINALS[as_of_match.group("day").lower()] if as_of_match else None

        revised_match = REVISED_TO.search(sentence)
        cut = revised_match.end() if revised_match else None

        date_spans: list[tuple[int, int]] = []
        for m in _DATE.finditer(sentence):
            date_spans.append(m.span())
            if cut is not None and m.start() < cut:
                continue
            month = _MONTH_ORDINALS[m.group("month").lower()]
            out.append(QuantityMention(
                value=float(month * 100 + int(m.group("day"))),
                unit_class="date",
                measure=_measure_tokens(sentence, m.start(), m.end()),
                scope=_scope_tokens(sentence, m.start()),
                revised=cut is not None,
                as_of=as_of,
                span=(offset + m.start(), offset + m.end()),
            ))

        for m in _NUMBER.finditer(sentence):
            if m.group("num") is None:
                continue
            if any(lo <= m.start() < hi for lo, hi in date_spans):
                continue          # the day-of-month in a date, already handled
            if cut is not None and m.start() < cut:
                continue          # the stale value of a revision construction
            currency, scale, unit = m.group("currency"), m.group("scale"), m.group("unit")
            digits = m.group("num").replace(",", "")
            if len(digits.split(".")[0]) < 2 and not (currency or scale or unit):
                continue          # "Ward 3"
            value = float(digits) * _SCALE.get((scale or "").lower(), 1.0)
            start = m.start("currency") if m.group("currency") else m.start("num")
            out.append(QuantityMention(
                value=value,
                unit_class=_unit_class(currency, scale, unit),
                measure=_measure_tokens(sentence, m.start(), m.end()),
                scope=_scope_tokens(sentence, m.start()),
                revised=cut is not None,
                as_of=as_of,
                span=(offset + start, offset + m.end()),
            ))
    return tuple(out)


# --- alignment and judgment -------------------------------------------------

# Measure-overlap floor for two mentions to be about the same fact. Tuned on
# the 7 DEVELOPMENT events only (see the plan's Step 5); the 3 held-out events
# are not consulted. The measure representation is deliberately loose -- clause
# UNION window -- so this floor is what stops unrelated facts pairing up.
#
# Measured on the 13 development pairs: precision is 1.000 at EVERY floor from
# 0.02 to 0.25, so recall is the only axis and the rule reduces to "the highest
# floor that loses no real conflict". That is 0.05 (9/9); 0.08-0.10 lose the
# n5_doc_002 <-> 003 pair ("Evacuations totaled 34 residents" vs "34 people were
# evacuated"), and 0.12+ additionally lose the dockworkers and sable-bridge
# conflicts.
#
# HAZARD, stated because a low floor looks free and is not: at 0.05 measure
# overlap is barely constraining, so alignment is close to "same unit_class plus
# one shared token" -- and "numbers of the same kind differ -> conflict" is the
# naive rule the spec measured at 12/19. What keeps it honest is that scope and
# revision still veto, and that the cross-event pairs and the 3 held-out events
# are the test of whether it generalizes (Task 11).
ALIGN_FLOOR = 0.05

# Fixed and explicit, because a pair routinely yields more than one kind of
# evidence. Conflict first so an incidental agreement elsewhere in the text
# cannot swallow a real divergence (n5_doc_017 <-> 019 does exactly that).
STANCE_PRECEDENCE = ("conflict", "revision", "partial", "agreement")


@dataclass(frozen=True)
class StanceEvidence:
    conflicts: tuple[tuple[QuantityMention, QuantityMention], ...] = ()
    revisions: tuple[tuple[QuantityMention, QuantityMention], ...] = ()
    partials: tuple[tuple[QuantityMention, QuantityMention], ...] = ()
    agreements: tuple[tuple[QuantityMention, QuantityMention], ...] = ()


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def align(
    a: tuple[QuantityMention, ...],
    b: tuple[QuantityMention, ...],
    *,
    floor: float = ALIGN_FLOOR,
) -> tuple[tuple[QuantityMention, QuantityMention], ...]:
    """Mention pairs plausibly about the same fact, best-overlap first.

    Same ``unit_class`` and measure Jaccard >= ``floor``. Reduced greedily so no
    mention is used twice: a text mentioning one figure must not align against
    three figures in the other and manufacture three pieces of evidence.
    """
    scored = [
        (_jaccard(x.measure, y.measure), i, j, x, y)
        for i, x in enumerate(a)
        for j, y in enumerate(b)
        if x.unit_class == y.unit_class
    ]
    scored = [row for row in scored if row[0] >= floor]
    # Sort by descending overlap, then by index so ties are deterministic.
    scored.sort(key=lambda row: (-row[0], row[1], row[2]))
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs: list[tuple[QuantityMention, QuantityMention]] = []
    for _score, i, j, x, y in scored:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((x, y))
    return tuple(pairs)


def judge(pair: tuple[QuantityMention, QuantityMention]) -> str:
    """Evidence kind for one aligned mention pair.

    Order is fixed and total, so the result never depends on check sequence:
      1. equal values                      -> agreement
      2. scopes differ                     -> partial   (different denominators)
      3. revised, or a strictly later as_of -> revision
      4. otherwise                         -> conflict
    """
    x, y = pair
    if x.value == y.value:
        return "agreement"
    if x.scope != y.scope:
        return "partial"
    if x.revised or y.revised:
        return "revision"
    if x.as_of is not None and y.as_of is not None and x.as_of != y.as_of:
        return "revision"
    return "conflict"


def evidence_for(
    a_text: str, b_text: str, *, floor: float = ALIGN_FLOOR
) -> StanceEvidence:
    """All aligned-fact evidence for a pair, bucketed by kind."""
    buckets: dict[str, list] = {kind: [] for kind in STANCE_PRECEDENCE}
    for pair in align(extract_mentions(a_text), extract_mentions(b_text), floor=floor):
        buckets[judge(pair)].append(pair)
    return StanceEvidence(
        conflicts=tuple(buckets["conflict"]),
        revisions=tuple(buckets["revision"]),
        partials=tuple(buckets["partial"]),
        agreements=tuple(buckets["agreement"]),
    )


def stance_for(
    a_text: str, b_text: str, *, floor: float = ALIGN_FLOOR
) -> Optional[str]:
    """The pair's single stance verdict, or None when no mentions aligned.

    This is what classify_relation consumes. Precedence is STANCE_PRECEDENCE.
    """
    ev = evidence_for(a_text, b_text, floor=floor)
    for kind, bucket in (
        ("conflict", ev.conflicts),
        ("revision", ev.revisions),
        ("partial", ev.partials),
        ("agreement", ev.agreements),
    ):
        if bucket:
            return kind
    return None
