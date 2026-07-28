# Spec: the SUPERSEDES channel (stance=revision typing)

**Date:** 2026-07-28
**Status:** Approved for planning
**Depends on:** the stance-quantity channel (`docs/superpowers/specs/2026-07-26-same-story-stance-channel-design.md`), the variant-D anchor fix (`docs/superpowers/specs/2026-07-26-stage1-anchor-findings.md`, amended 2026-07-28)
**Motivates:** node6 (`docs/node6-collaborator-setup.md`), authored update-heavy so it can score this channel blind

## Problem

`gin/cartographer/quantity.py`'s stance channel already detects revisions —
`judge()` buckets an aligned mention pair as `"revision"` when one side carries
a `revised to` marker or the two sides' `as_of` weekdays differ — but
`classify_relation` in `combined.py` has nowhere to put that verdict. Every
`stance == "revision"` pair falls through to `RELATED_UNTYPED, "abstain"`,
identical to a pair the channel found no evidence in at all. The 4-way
confusion table in `scripts/eval_node5_stance.py`'s output confirms this is
live, not hypothetical: all 5 gold `supersedes` pairs in the node5 labels
already resolve to `stance == "revision"` with correctly aligned quantities —
they just have no typed relation to become.

`Relation.SUPERSEDES` and `GRAPH_EDGE_RELATIONS` already exist
(`gin/cartographer/models.py`), the Bookkeeper already admits and
cycle-checks ordering edges for it (`gin/bookkeeper/graph.py`,
`ORDERING_RELATIONS`), and the curator UI already lets a human label a pair
`supersedes` (`gin/curator/app.py`). Only the automated detector's typing
decision is missing.

## Why now, not after node6

Node6 is being authored update-heavy (~12 `update`-kind pairs — see
`docs/node6-collaborator-setup.md`, composition targets) specifically to
stress-test this channel. If the channel is built after his corpus lands, it
will inevitably be developed against it, and node6 loses its value as an
independent check — the same leakage caveat that qualifies every held-out
number measured so far in this project. Building now means node6 arrives as a
genuinely blind test, written by someone who has never seen this spec.

## Goal

Type `stance == "revision"` pairs as `SUPERSEDES` with a directed edge (the
newer report supersedes the older one), when — and only when — the pair's
textual evidence resolves which side is newer. Leave every other decision
`classify_relation` makes untouched.

## Non-goals

- Timestamp-based direction (rejected — see Decisions). The relation typer
  stays text-only; it does not consult chunk publication metadata.
- Calibrating thresholds for SUPERSEDES. The stance channel is model-free
  and rule-based; there is nothing to calibrate, so `calibration_export.py`
  keeps excluding it from `_CLASSIFIER_RELATIONS` (see Decisions).
- Undirected SUPERSEDES. If direction cannot be resolved, the pair abstains
  (`RELATED_UNTYPED`) exactly as it does today — it does not get a weaker,
  undirected edge as a consolation.
- Touching `judge()`'s revision *detection* logic, `align()`, or any
  extraction rule in `quantity.py`. This spec only adds a direction resolver
  and a new branch in `classify_relation` / `type_relation`.

## Preliminary check: the five gold pairs, read by hand

Before committing to the design above, the five node5 `supersedes` pairs
(`n5_doc_019↔020`, `030↔032`, `031↔032`, `007↔008`, `007↔010`) were read
directly against `corpus_node5.json`'s `published` timestamps. In every one,
the later-published report carries either a `revised to X` construction
naming the earlier value, or a later `as_of` weekday than the earlier report
— i.e. `judge()`'s existing revision detection already has a marker to key
off, and in each case the marked/later side matches the later-published
document. This is read-only evidence that the mechanism in §2 has something
to act on for all 5 pairs; it is not a run of the resolver (which does not
exist yet) and is not the pre-registered bar itself — the bar in §4 (>=4/5)
stays the number graded against, not 5/5, since implementation details
(greedy alignment pairing, the sentence-scoped revision-bleed limitation
already documented in `quantity.py`) could still cost one pair once the code
exists. These five pairs currently abstain with `stance == "revision"`, which is
distinct from `UNALIGNED` (§3) — they are not part of, and do not affect,
`test_examined_but_unaligned_same_story_pairs_abstain_end_to_end`'s pinned
set.

## Decisions

**Direction: textual cues only, no timestamps.** The side whose sentence
carries the `revised to` marker is the corrected (newer) report — the
correction *is* the newer claim by construction. For `as_of`-only pairs (no
`revised to` on either side but both carry a weekday marker), the later
weekday ordinal is the newer side. If neither cue resolves — both sides
revised, or a genuine tie — the pair does not get a direction, and
`classify_relation` abstains rather than emitting an edge with an arbitrary
side. Rejected the alternative (plumb `published` timestamps into the typer):
it changes the text-in/relation-out interface that `combined.py` and
`quantity.py` have held throughout this project, trusts corpus metadata nothing
else here depends on, and isn't available in every deployment. Text-only
costs recall on any revision that states no self-orienting marker; that cost
is accepted, not designed around.

**Calibration stays excluded.** SUPERSEDES was already carved out of
`_CLASSIFIER_RELATIONS` in `gin/curator/calibration_export.py` with the
comment "SUPERSEDES is a graph relation, not a detector output" — no longer
literally true once this ships, but the practical reason to exclude it is
unchanged: nothing about this channel is threshold-tuned, so there is no
calibration target to add, and the held-out-40 baseline (0.725) should stay
comparable across this change rather than being redefined mid-project.
Revisit when node6 gives the class enough labels (~12) to be worth including.

## Design

### 1. Direction resolver — `gin/cartographer/quantity.py`

```python
def revision_direction(evidence: StanceEvidence) -> Optional[str]:
    """Which side of a revision pair is the newer report: "a", "b", or None.

    Consults only the `revisions` bucket. Walks it in order (align()'s
    best-overlap-first ordering); a marker (revised=True) on exactly one side
    of a pair is decisive. Where no pair carries a marker, falls back to
    as_of comparison. Returns None — not a guess — when both sides of every
    revision pair are marked, when as_of is absent or tied, or when different
    revision pairs disagree about which side is newer: an undirected
    conclusion abstains rather than picking arbitrarily.
    """
```

Semantics, precisely:

- For each `(x, y)` in `evidence.revisions`: if `x.revised != y.revised`, the
  revised side is newer for that pair. Collect a vote (`"a"` or `"b"`) per
  pair that has a decisive marker.
- If no pair has a decisive `revised` marker, fall back to `as_of`: for each
  pair where both `as_of` are set and differ, the later ordinal's side votes
  newer. (`judge()` already requires this to differentiate `"revision"` from
  `"conflict"`, so every pair in the bucket has *some* orienting signal —
  either a marker or a differing `as_of` — by construction. The resolver
  reads whichever one `judge()` used.)
- If all votes agree, return that side. If votes disagree, or there are no
  votes (should not happen given `judge()`'s contract, but handled
  defensively), return `None`.

No models, no I/O — same purity contract as the rest of the module. Unit
tested directly against `StanceEvidence` fixtures, no corpus needed.

### 2. Wiring — `gin/cartographer/combined.py`

`classify_relation` gains exactly one new parameter, `stance_direction:
Optional[str] = None`, threaded alongside the existing `stance` parameter.
It stays a pure function over scalars (`cos`, `p_contra`, `same_story`,
`stance`, `stance_direction`) shared with the calibrator — `stance_direction`
is just `"a"` / `"b"` / `None`, not a `StanceEvidence` object. Resolving
*which* string to pass is `type_relation`'s job, one layer up, since that
requires the full `StanceEvidence`; `classify_relation` only branches on the
result. Threading the `StanceEvidence` object itself into `classify_relation`
would leak a `quantity.py` type into the calibrator's pure-scalar contract for
no benefit.

Concretely:

- `type_relation` calls `quantity.evidence_for(a_text, b_text)` once (already
  computing stance via a call that does equivalent work — see Implementation
  note below) and, when the top-precedence bucket is `"revision"`, also calls
  `revision_direction(evidence)` and passes its result as
  `stance_direction`.
- Inside `classify_relation`: when `stance == "revision"` and
  `stance_direction is not None`, return `(Relation.SUPERSEDES, "stance")`.
  When `stance == "revision"` and `stance_direction is None`, fall through to
  the existing abstain branch — byte-identical to today's behavior for
  undirected revisions. Every other caller of `classify_relation` (e.g. the
  calibrator) that does not pass `stance_direction` keeps its current
  behavior unchanged, since the default is `None`.
- The NLI-veto branch (`stance_disagrees`) already treats any non-`None`,
  non-`"conflict"` stance as disagreement with a firing NLI; `"revision"`
  already qualifies, so a firing NLI on a directed-revision pair already
  routes to `stance_disagrees` and returns `RELATED_UNTYPED, "abstain"` —
  **not** `SUPERSEDES`. This spec keeps that behavior: the veto's job is to
  stop a wrong NLI CONTRADICTS, not to make SUPERSEDES win over a
  disagreeing NLI. A pair where NLI fires and stance says directed-revision is
  genuinely ambiguous (propositional contradiction signal vs. structured
  revision signal); abstaining is the same conservative choice already made
  for corroborates/partial disagreement. Revisit only if node6 shows this
  case is common and one signal is reliably right.
- `assess_pair` (or `type_relation`, wherever direction is resolved) orders
  the `Assessment`'s `src_chunk_id`/`dst_chunk_id` so **src supersedes
  dst** — matching `gin/bookkeeper/graph.py`'s directed-edge convention
  (`edge_key` treats `SUPERSEDES` as `(relation, src, dst)`, not symmetric).
  Concretely: if the resolver says side `"a"` (the caller's first argument)
  is newer, the caller is responsible for emitting the edge as
  `(a_chunk, b_chunk, SUPERSEDES)` — i.e. **the newer text is src**. This
  matches curator intuition (`gin/curator/app.py`'s `supersedes` button
  already records `prior.id`, the OLDER record, as the value on the newer
  one — an inverse-pointer convention. This spec's edge direction is
  independent of that UI field; it establishes what `type_relation`/
  `assess_pair` emit as a Cartographer `Assessment`, which is what the
  Bookkeeper consumes.)

**Implementation note on redundant computation.** `type_relation` currently
calls `self.stance_provider(a_text, b_text)` (`quantity.stance_for`), which
internally calls `extract_mentions` twice and `_evidence_from_mentions` once,
then discards the `StanceEvidence` and returns only the winning bucket's
name. Resolving direction needs the `StanceEvidence` object, not just the
name `stance_for` returns. Rather than call both `stance_for` and
`evidence_for` (duplicating extraction), `type_relation` switches to calling
`quantity.evidence_for` once and deriving both the precedence-ranked stance
string (via `StanceEvidence.first()` and `STANCE_PRECEDENCE`, or a small
`stance_from_evidence(evidence) -> Optional[str]` extracted from
`stance_for`'s tail) and the direction from the one `StanceEvidence`. This is
a refactor of `type_relation`'s stance-computation call, not of
`quantity.stance_for` itself (which stays as the public entry point other
callers use, e.g. `CombinedRelationProposer(stance_provider=...)` injection
in tests — the default provider becomes `evidence_for` composed with the
extracted stance-from-evidence helper, so the injectable-callable contract
`stance_provider: Callable[[str, str], Optional[str]]` is unchanged and
existing injected-stance tests keep working unmodified).

### 3. What must not move

Enforced by re-running the existing pinned tests, not by new assertions:

- `tests/test_cartographer_stance_node5.py`: contradicts P 1.000, R 1.000,
  the empty false-positive-by-channel dict.
- `tests/test_cartographer_stance_branch.py`: the abstain-set pin (currently
  `[("n5_doc_036:0", "n5_doc_037:0")]`) is unaffected. It filters specifically
  on `ev.get("stance") != UNALIGNED` (`UNALIGNED` is the sentinel meaning
  "both sides had quantities, none aligned" — see `quantity.py`), which
  excludes `stance == "revision"` pairs already: they were never in this
  pinned set before this change and are not affected by it. Re-run to
  confirm, but no edit to this pin is expected.
- `tests/test_cartographer_calibration*.py`, the 39-sample fixture, the
  14-pair bar pin: untouched, since `_CLASSIFIER_RELATIONS` does not change.
- `scripts/verify_node5_surfacing.py`: 42/42 — SUPERSEDES is a
  `TYPED_EDGE_RELATIONS` member already (`gin/cartographer/models.py`), so
  a newly-typed SUPERSEDES pair still counts as "surfaced", not lost.
- `scripts/recalibrate_cheap_pipeline.py --score-only`: held-out-40 accuracy
  0.725 unchanged (SUPERSEDES pairs are already outside its scored classes).

### 4. Pre-registered bar

Written now, before implementation, so it cannot be chosen to flatter a
result already seen:

**On node5 (5 gold `supersedes` pairs, all within-event):**
- At least 4 of 5 type `SUPERSEDES` with direction matching the manifest's
  `published` order (the later-published report as src). A pair that
  abstains for lack of a resolvable textual cue is an accepted miss, not a
  failure of the bar, provided it is not silently mistyped in the wrong
  direction.
- Zero `SUPERSEDES` edges on any node5 pair not gold-labeled `supersedes`.
- All contradicts-channel metrics named in §3 unchanged.

**On node6 (once authored, scored once, blind — before reading his intent
matrix):**
- SUPERSEDES precision >= 0.8 over his `update`-kind pairs (of pairs the
  detector types SUPERSEDES, at least 80% are gold `update`).
- Zero SUPERSEDES edges on his `conflict` or `corroboration`-kind pairs.
- Recall on `update` pairs is measured and reported, not gated — node6's
  update pairs may use revision language this module's vocabulary doesn't
  cover (`REVISED_TO`, `_AS_OF` are node5-derived regexes), and a recall miss
  there is exactly the generalization signal node6 exists to produce, not a
  shipped-code failure.

## Testing plan (TDD, per the project's standing practice)

1. `revision_direction` unit tests: marker on side a / side b, as_of-only
   ordering both directions, both-sides-revised → None, conflicting votes
   across multiple aligned pairs → None, empty evidence → None.
2. `classify_relation` branch tests: directed revision → SUPERSEDES/"stance";
   undirected revision → unchanged abstain; NLI-firing + directed-revision →
   unchanged abstain (veto keeps priority, per Decisions).
3. `CombinedRelationProposer.assess_pair` / `type_relation` integration:
   directed-revision pair emits an `Assessment` with SUPERSEDES and src/dst
   ordered newer-as-src; confirm the injectable `stance_provider` contract
   (`Callable[[str, str], Optional[str]]`) still works unmodified for tests
   that inject a bare string stance with no direction available (those must
   still abstain, matching current behavior, since injected callables cannot
   supply a `StanceEvidence` for direction resolution — this is the expected,
   documented limitation of the simple injection path; tests wanting
   directed SUPERSEDES must inject at the `evidence_for`/`same_story` level,
   not the `stance_provider` level, or use real text through the default
   provider).
4. Re-run every pin named in §3 to confirm none need edits (§3 argues none
   should, given the `UNALIGNED`-vs-`"revision"` distinction — this step
   verifies that argument rather than trusting it).
5. Extend `scripts/eval_node5_stance.py` with a SUPERSEDES section: per-pair
   channel/direction printout for the 5 gold pairs, scored against the §4
   bar.
6. Full suite + `verify_node5_surfacing.py` + `recalibrate_cheap_pipeline.py
   --score-only` per the project's standing WSL test workflow.

## Open questions for the plan (not blocking this spec)

- Exact extraction point/name for the stance-from-evidence helper pulled out
  of `stance_for`'s tail (§2, Implementation note) — an implementation
  planning detail, not a design decision.
- Whether `gin/curator/app.py`'s existing `supersedes` UI field (an
  inverse-pointer convention distinct from this spec's edge direction) needs
  any documentation cross-reference once this ships, to avoid confusing the
  two conventions. Flagged, not required by this spec.
