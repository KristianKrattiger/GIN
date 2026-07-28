# Findings: the stage-1 anchor fix is withdrawn, and stage 2 does not need it

**Date:** 2026-07-26
**Status:** Stage-1 change NOT shipped. Recorded for a future redesign.
**Branch:** `stance-channel`
**Decision:** user, 2026-07-26 — "stop and reconsider the whole stage-1 fix"
**Parent spec:** `docs/superpowers/specs/2026-07-26-same-story-stance-channel-design.md`

## What was attempted

The parent spec's Component 3 proposed two independent corrections to
`gin/cartographer/relatedness.py`:

| # | change | status |
|---|---|---|
| B | `anchor_tokens` excludes calendar words | **shipped** (`231d55d`) |
| C | `make_same_story` requires the anchor be entity-grade in **both** texts (`&`, not `|`) | **withdrawn** |

Measured on the 24 node5 curator labels, the pair was expected to take
cross-event false positives from 5/5 to 0/5 while keeping 19/19 within-event
pairs, with no threshold moved.

## Why C was withdrawn

The intersection requirement **regresses two pre-registered gold `contradicts`
pairs** — `disc_nw_pr:0 ↔ disc_nw_complaint:0` and
`disc_mer_pr:0 ↔ disc_mer_complaint:0`, the SEC/securities register. Legal-register
recall drops 4/7 → 2/7.

The cause is an interaction the spec did not anticipate. `anchor_tokens`
deliberately treats **sentence-initial** capitalization as carrying no entity
signal — a rule added because corpus-rare boilerplate (`Combined reservoir
storage…`) drove scan false positives (run `20260712T091415Z`). But a proper
noun that happens to open a sentence is still a proper noun:

```
disc_nw_pr:0        "Northwind Systems reported record third quarter revenue…"
                     ^^^^^^^^^ sentence-initial -> NOT entity-grade
disc_nw_complaint:0 "The complaint alleges Northwind materially overstated…"
                                           ^^^^^^^^^ mid-sentence -> entity-grade
```

Under the union, `northwind` was anchor-grade because of the *complaint* text.
Under the intersection it is anchor-grade in neither direction, and the pair
fails.

**This is the opposite of the case the fix targets.** The spec's own guidance was
"investigate before reverting: the union anchor may have been carrying a pair for
the wrong reason." Investigation shows the union was carrying these two for
exactly the **right** reason — `Northwind` and `Meridian` genuinely are the shared
story entity. The `Sable Bridge` / `bridge the gap` collision the fix was aimed at
is a different phenomenon that happens to be caught by the same test.

## Two refinements were measured and both work

Over `default_text_index()` (274 docs, `_rare_df_ceiling` 9):

| variant | within-event | cross-event FP | legal pairs |
|---|---|---|---|
| union — current shipped state, post-calendar-fix | 19/19 | 4/5 | **2/2** |
| plain intersection — what the spec proposed | 19/19 | **0/5** | **0/2** |
| **C′** sentence-initial is entity-grade when the next word is also capitalized | 19/19 | **0/5** | **2/2** |
| **D** entity-grade on one side, merely capitalized on the other | 19/19 | **0/5** | **2/2** |

`C′` admits `Northwind Systems` and `Meridian Health` while still rejecting
`Combined reservoir storage` and `The complaint`. `D` leaves `anchor_tokens`
untouched and changes only `make_same_story`'s test.

Both were withdrawn along with plain intersection: the user's judgment is that an
anchor heuristic now needing a third patch to keep two eval pairs alive wants a
redesign, not another special case. `scripts/sweep_same_story.py` (Task 6) is the
committed artifact carrying this design space so the redesign starts from data.

## What the full sweep added: robustness, not just optimality

Running the committed sweep over the whole grid (4 anchor modes × `story_floor`
{2,3,4} × `df_ceiling` {4,6,7,9,12}) turned up a result neither the spec nor the
withdrawal anticipated. Nine cells reach a perfect 19/0/4/0:

| mode | floor | ceilings reaching 19 / 0 / 4 / 0 |
|---|---|---|
| `union` (shipped anchor rule) | 2 | **6 only** |
| `inter_cap` | 2 | 6, 7, 9, 12 |
| `mixed` | 2 | 6, 7, 9, 12 |

**A threshold-only fix exists.** `union / floor 2 / ceiling 6` — no anchor change
whatsoever, just a tighter rare-token ceiling — scores 19/19 within-event, 0/5
cross-event, 4/4 gold contradicts, 0 gold false positives. It is the cheapest
possible route to what the withdrawn anchor fix was chasing.

**And it is a knife edge.** One step to `ceiling 7` and cross-event false
positives jump 0 → 4. The natural ceiling at 274 documents is
`_rare_df_ceiling(274) = 9`, so reaching that cell means overriding the formula
and pinning a value that the corpus growing by ~30 documents would move off.
That is precisely the "tuning a global predicate on n=24" trap the parent spec
declined to walk into.

The two anchor refinements are **flat across ceilings 6–12**. They achieve the
same outcome without depending on a threshold sitting in a narrow window, which
is a structural property rather than a fitted one.

So the sweep does not merely rank the options — it separates them by kind. The
threshold route is fitted and fragile; the anchor route is robust but needs the
`anchor_tokens` redesign this document leaves open. That distinction is the most
useful thing to carry into the redesign, and it is the reason the artifact scores
every cell rather than reporting a single winner.

## The finding that makes withdrawal cheap: stage 2 absorbs stage 1's residue

**Note, added after amendment (2026-07-26):** the table and the single-error
framing immediately below were measured under the `None → abstain` variant of
Component 2 — the variant the parent spec's Results section ("Amendment to
Component 2 — `None` versus `UNALIGNED`") later rejected in favor of the
`UNALIGNED` sentinel. That rejected variant is exactly where the `P_all` 0.923
figure below comes from. At the shipped state, `stance=None` falls through to
the pre-stance band branch instead of abstaining — the price of keeping the
three quantity-free gold contradicts pairs (housing habitability, the
securities PR/complaint pair) alive — so the measured figure at head is
`P_all` **0.857**, not 0.923, and there are **two** cross-event false positives,
not one: `n5_doc_023↔024` (discussed below, through the `stance` channel) and
`n5_doc_023↔026` (through the `band` channel, `stance=None`). This document
predates that amendment. The conclusion in the paragraph below it — that the
stage-1 fix is not required to clear the pre-registered bar — still holds at
0.857.

Measured with the stance channel over the **unfixed** stage 1 (union anchors,
calendar fix only):

| metric | baseline (`ebceb46`) | stage 2 over unfixed stage 1 | bar |
|---|---|---|---|
| `P` within-event precision | 0.632 | **1.000** | must improve |
| `R` recall | 1.000 | **1.000** | ≥ 0.75 |
| `P_all` incl. cross-event | 0.500 | **0.923** | must improve |

**The stage-1 fix is not required to clear the pre-registered bar.** Four of the
five cross-event pairs still pass stage 1, but stage 2 finds no aligned quantity
in them and abstains, so they never become CONTRADICTS edges. Stage 1's precision
problem is largely invisible downstream once the branch requires stance evidence —
which is a result about where the defect actually mattered.

## The two surviving errors, and what they tell us

**Note, added after a second amendment (2026-07-27):** sub-project G
(`docs/superpowers/specs/2026-07-27-quantity-hedge-word-stopword-design.md`)
removed `n5_doc_023↔024` by adding the hedge-adverb "roughly" to
`quantity.py`'s stopword list — the shared token that let stage 2
manufacture its `conflict` in the first place. At head, `P_all` is **0.923**,
not 0.857, and there is **one** cross-event false positive, not two: only
`n5_doc_023↔026` remains (discussed below). The paragraphs immediately
below are preserved as the historical record of a defect that existed and
was fixed by a different route than the union-anchor redesign this document
was written to evaluate — `023↔024` is now the regression case
`tests/test_cartographer_quantity.py::test_roughly_does_not_align_two_unrelated_quantities`
pins, exactly as the paragraph below already recommended keeping it for.

At the shipped state (`UNALIGNED` sentinel, `P_all` 0.857) there are **two**
cross-event false positives, not one.

`n5_doc_023:0 ↔ n5_doc_024:0` (Delacroix port strike vs Crosstown transit
suspension) requires **both** weaknesses at once:

1. stage 1 passes it, because `Union Yard` in the transit report anchors against
   `the union local` in the strike report — the union-anchor defect; and
2. stage 2 manufactures a `conflict` from it, because at `ALIGN_FLOOR = 0.05` two
   unrelated counts clear the measure-overlap test.

Either fix alone removes it. It is also the **first concrete instance of the
low-floor hazard** the plan pre-registered as its main generalization concern —
at 0.05, alignment is close to "same unit class plus one shared stem", and here
that produced a conflict between a dockworker headcount and a commuter delay in
minutes. Worth keeping as a regression case for any future floor change.

`n5_doc_023:0 ↔ n5_doc_026:0` is different in kind: stance reads `None` on this
pair (at least one side states no quantity the channel can judge), so
`classify_relation` falls through to the pre-stance band branch and types it
CONTRADICTS on story membership alone — the original defect A, still reachable
by design. This is the measured price of keeping `None` fall through to that
branch rather than abstaining: the same `None` path is what preserves the three
quantity-free gold contradicts pairs. There is no single fix that removes this
one without also removing that path's benefit; it is a tradeoff, not a bug in
the ordinary sense.

## What is left on the branch

- **Kept:** `231d55d`, the calendar-word exclusion. Independent of the interaction
  above, regresses nothing (suite 679 passed / 16 skipped, both legal pairs
  intact), and justified on its own terms — a weekday is not a story entity.
- **Withdrawn:** the union → intersection change, uncommitted and reverted.
- **Unchanged:** `story_floor` (2), `_rare_df_ceiling`, and
  `data/cartographer_thresholds.json`.

## Known defect recorded but deliberately NOT fixed

`scripts/verify_node5_surfacing.py` and `scripts/curator_serve.py` both build the
predicate over `node5 texts + default_text_index()`. That was correct before node5
was registered in `CORPUS_NODES` (`c039edd`); it is now **double-counting**, since
`default_text_index()` already contains node5. Doubling node5's document
frequencies pushes tokens above the rare ceiling and *masks* cross-event false
positives — it reported union at 0/5 instead of the true 4/5 during this
investigation.

The doubled frequencies stay under the ceiling, so the surfacing gate still passes
42/42 and nothing is visibly broken. User's decision (2026-07-26) is to leave it.
Recorded here because it silently understates exactly the quantity a stage-1
redesign would need to measure, so a redesign should fix it first.

## Open question for the redesign

`anchor_tokens` now carries three heuristics that interact: mid-sentence
capitalization as a proper-noun proxy, all-caps as a dateline, and multi-digit
numerals as story figures — plus two exclusions (sentence-initial, calendar
words). Each was added to fix a measured false-positive class, and the
sentence-initial rule is now demonstrably over-broad. Whether an
entity-recognition step belongs here at all, rather than a widening stack of
capitalization rules, is the question this withdrawal leaves open.
