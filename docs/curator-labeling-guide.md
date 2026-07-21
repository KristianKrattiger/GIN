# Curator labeling guide

Five relations, two `relation_class` values seen so far. When unsure, prefer
`related_untyped` over forcing a typed relation — it's a valid, storeable
answer, not a cop-out (see `gin/cartographer/models.py:14-26`).

## contradicts

Two chunks describe the same underlying event/topic but frame or diverge on
it. Split into two `relation_class` flavors:

- **issue_frame** — same topic, different institutional-vs-grassroots framing
  of the *issue itself* (not a specific breaking story):
  - "emissions: institutional reduction-target framing vs grassroots treaty
    demands"
  - "wildfire: acreage / suppression metrics vs air quality in low-income
    zones"
  - "rezoning: parcel/FAR/density technical framing vs displacement/
    right-to-return organizing framing"

- **story** — same breaking story, sharing a lede/anchor fact, but the two
  chunks diverge on a specific reported number:
  - "hospital treatment count and arrest count diverge after shared lede"
  - "turnout percentage diverges after shared margin"

If it's same-topic-divergent-numbers *without* a shared lede/anchor, it's
probably `issue_frame` or `related_untyped`, not `story` — `story` requires
that shared anchor.

## corroborates

Same stance, same topic — one chunk backs up or gives a concrete instance of
the other's claim:
- "Snippet B provides specific empirical evidence (historical US warming
  data) that corroborates the broader claim"
- "Snippet B provides a specific empirical example of a record-breaking
  extreme heat event"

## supersedes

One chunk is a later, corrected/updated version of the same claim (rare in
the corpus so far — no worked example yet; if you hit one, add it here).

## related_untyped

Same broad topic, but the two chunks can't be directly logically linked
(different metrics, different aspects) — genuinely related, but not
`corroborates`/`contradicts`:
- "Both snippets discuss climate warming temperature metrics, but they
  cannot be directly linked logically"
- "Both snippets provide data regarding record-breaking warm temperatures,
  but they are independent metrics"

## unrelated

Different topics entirely — this is what the relatedness gate should reject
outright:
- "Different types of facts, statistics that are unrelated to elderly,
  low-income populations..."

## The hardest borderline: related_untyped vs unrelated vs contradicts

Same-topic-different-metric pairs *feel* related but often aren't a typed
relation. Ask in order:
1. Do the two chunks discuss the same real-world topic at all? If no →
   `unrelated`.
2. If yes: can you draw a direct logical line between them (one confirms,
   extends, or conflicts with the other's specific claim)? If no → 
   `related_untyped`.
3. If yes, and the conflict is framing/stance rather than the same
   number/fact → `contradicts` (pick `issue_frame` or `story` per above).
