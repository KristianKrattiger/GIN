# Design: Same-Story Stance Channel — Making the CONTRADICTS Branch Evidence-Based

**Date:** 2026-07-26
**Status:** IMPLEMENTED and measured 2026-07-26 (branch `stance-channel`). Component 2 amended during execution (None vs UNALIGNED); Component 3's union-to-intersection change WITHDRAWN. See Results.
**Sub-project:** E — the follow-on node5 was built to enable
**Predecessor:** `docs/superpowers/specs/2026-07-25-same-story-corpus-node5-design.md`
**Labels this rests on:** 24 curator labels over `corpus_node5.json`, committed at `ebceb46`

## Problem

Node5's spec shipped a corpus and declared the combined detector's degenerate
branch explicitly out of scope: *"fixing `combined.py`'s unconditional
`if same_story: return CONTRADICTS` branch. That is the follow-on this corpus
exists to enable, and it needs labels first."*

The labels now exist. 24 pairs, curator `kristian`, 2026-07-26: **12
`contradicts`, 5 `supersedes`, 5 `unrelated`, 2 `corroborates`**. Every one of
the 24 returns `same_story == True`. So the branch scores precision
**12/24 = 0.50** — **12/19** counting only the 19 within-event pairs, since 5 of
the 24 are cross-event pairs stage 1 should never have passed.

That single number decomposes into two independent defects, in two different
stages, plus a registration gap that keeps the labels from reaching anything.

| # | Defect | Location | Kind |
|---|--------|----------|------|
| A | `if same_story: return CONTRADICTS` types any same-story pair a conflict with no stance evidence | `gin/cartographer/combined.py:97-98` | structural |
| B | Weekday and month names are entity-grade, so a calendar word anchors a story | `gin/cartographer/relatedness.py:86` | bug |
| C | The anchor set is the **union** over both texts, so a proper noun in one licenses a coincidental common noun in the other | `gin/cartographer/relatedness.py:117` | bug |
| D | `CORPUS_NODES` stops at node4, so no `n5_doc_*` id resolves and all 24 labels drop out of B's dataset and C's calibration export | `gin/curator/text_index.py:28` | registration |

**Why one spec.** B and C both change `make_same_story`, which is a *global*
predicate consumed by the scan, the curator, the calibration sample generator
and the frames dataset. D changes the document-frequency corpus that predicate
is built over (236 → 274 docs, `_rare_df_ceiling` 7 → 9). All three therefore
land on the same frozen eval surfaces and share one re-measurement. Splitting
them means measuring the bar three times against three moving baselines.

## Measured evidence

Everything below was measured during design, not assumed.

### NLI cannot carry the branch

Real `all-MiniLM-L6-v2` + `nli-deberta-v3-xsmall` over the 24 labeled pairs, at
the shipped `contra_threshold = 0.686`:

| label | n | `p_contra` range | fire at threshold |
|---|---|---|---|
| contradicts | 12 | 0.008 – 0.960 | 2 |
| corroborates | 2 | 0.494 – **0.983** | 1 |
| supersedes | 5 | 0.033 – **0.980** | 1 |
| unrelated (cross-event) | 5 | 0.016 – 0.692 | 1 |

The two highest `p_contra` scores in the entire set are a **corroborates**
(`n5_doc_036↔037`, 0.983) and a **supersedes** (`n5_doc_007↔008`, 0.980) — above
every genuine conflict but two. So "delete the band branch and let the NLI
channel decide" yields 2 TP / 3 FP: precision **0.40**, recall **0.17**. Worse
than the 0.50 it would replace, on both axes. That option is closed by
measurement, not by preference.

### The discriminator is per-fact and structural

Reading all 19 within-event texts, the three kinds separate on structure, not on
whether numbers differ — **all 19 contain a numeric divergence**, so a naive
"numbers differ → contradicts" rule also scores 12/19 and changes nothing.

| kind | signature | example |
|---|---|---|
| conflict | same measure, same scope, different value | `34 people were evacuated` / `19 people were evacuated` |
| supersedes | revision marker or temporal progression **on that fact** | `initially reported at 8.5 … revised to 12`; `since Monday` → `as of Thursday` |
| corroborates | the numbers attach to **different** measures or scopes | `total capacity incl. standing-room 42,000` / `fixed seats in the bowl 36,500`; `34 hospital-wide` / `Ward 3 alone 21` |

Two of the 12 conflicts — `n5_doc_005↔006` and `n5_doc_017↔020` — carry revision
language *on a fact other than the conflicting one* (margin `6→9` revised while
turnout `47` vs `52` conflicts; winds `90→105` revised while shelters `65` vs
`40` conflict). A **pair-level** revision veto therefore costs two real
conflicts; a **fact-aligned** one does not. This is why the mechanism in §1 is
alignment-based.

### Both anchor defects, and neither needs a threshold moved

| variant | within-event same-story | cross-event FP |
|---|---|---|
| current (union anchors, calendar allowed) | 19/19 | **5/5** |
| drop calendar words only | 19/19 | 4/5 |
| require anchor entity-grade in **both** texts | 19/19 | 1/5 |
| **both fixes** | **19/19** | **0/5** |

Defect C is the larger one and was not in the handoff. `make_same_story` tests
`(anchor_tokens(a) | anchor_tokens(b)) & rare`. The union means an entity-grade
occurrence anywhere licenses the shared token, even where the sharing side's
occurrence carries no entity signal at all:

- `n5_doc_013` `Sable **Bridge**` anchors against `n5_doc_025` `bus shuttles to
  **bridge** the gap` — a verb.
- `n5_doc_024–026` `**Union** Yard` anchors against `n5_doc_023` `The **union**
  local said` — and that one collision is all three of the `023` false
  positives.

Both fixes are strictly better than the threshold repairs measured at `ebceb46`
(`story_floor 4` → 2 FP; `df_ceiling 4` → 0 FP but loses 2 real conflicts), and
neither is a tuning decision on n=24: each is a semantic correction with
standalone justification. `story_floor` and `df_ceiling` are **not** changed.

## Goal

Make the same-story CONTRADICTS branch require stance evidence, correct the two
anchor defects, register node5 so its labels reach their consumers, and
re-measure every global surface the predicate changes move — reporting each
number whichever way it goes.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Stage-2 mechanism | Fact-aligned quantity stance channel | The only option measured to move the number. Handles the two mixed-fact conflicts a pair-level veto forfeits. |
| Stage-2 fallback | Abstain (`RELATED_UNTYPED`) | A wrong CONTRADICTS edge costs a knowledge graph more than a missing one. |
| Stage-1 repair | Both anchor fixes; no threshold change | Both are bugs with independent justification; the thresholds are an n=24 tuning question that stays deferred. |
| Threshold question | Committed sweep script, no values written | The deferred decision gets a reproducible artifact instead of a paragraph in a commit message. |
| Recalibration | Re-measure only; `cartographer_thresholds.json` untouched | Recalibrating under a half-changed pipeline restates the change rather than evaluating it. Its own spec, once this one's number is known. |
| Backward compatibility | `stance=None` reproduces current behavior exactly | Keeps the baked 39-sample fixture, the 14-pair bar pin and every existing test valid without edits. |

## Architecture

```
                    gin/cartographer/quantity.py          (new, model-free)
                    extract -> align -> judge
                              │
                              │  StanceEvidence
                              ▼
gin/cartographer/combined.py  classify_relation(cos, p_contra, t, *,
                                                same_story, stance)
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
gin/cartographer/relatedness.py               gin/curator/text_index.py
  anchor_tokens   (calendar fix)                CORPUS_NODES += node5
  make_same_story (union -> intersection)       236 -> 274 docs, ceiling 7 -> 9
        │                                           │
        └─────────────────────┬─────────────────────┘
                              ▼
            frozen eval surfaces, re-measured after each commit
            45-pair eval set | 14-pair bar pin | held-out 40-pair score
```

### Component 1 — `gin/cartographer/quantity.py` (new)

Pure, model-free, no network, no corpus statistics. Three passes.

**Extract.** Quantity mentions as
`QuantityMention(value: float, unit_class: str, measure: frozenset[str], scope: frozenset[str], revised: bool, as_of: Optional[int], span: tuple[int, int])`.

- `unit_class` ∈ `{count, currency, percent, speed, area, date}`. Recognizes
  `$18 million`, `47 percent`, `42,000`, `105 mph`, `12 square kilometers`,
  `September 3`. Thousands separators and scale words (`million`) fold into
  `value`; a bare multi-digit numeral is `count`.
- `measure` — normalized content tokens governing the numeral: the noun phrase
  it modifies plus the governing verb, stopwords dropped, light stemming so
  `evacuated` / `evacuations` align. E.g. `{people, evacuat}`,
  `{turnout, percent, registered, voter}`, `{total, damage}`, `{capacity}`.
- `scope` — narrowing qualifiers from a closed vocabulary: `hospital-wide`,
  `ward`, `alone`, `citywide`, `downtown`, `total`, `fixed`, `permanent`,
  `standing-room`, `at the port itself`, `across the country`. The vocabulary is
  a module constant, reviewed as data, extended by editing one list.
- `revised` — the mention sits inside a revision construction:
  `initially reported at X … revised to Y`, `initially estimated`, `updated to`,
  `was revised to`.

  **A revision construction collapses to one mention, not two:** `initially
  reported at 90 mph, were revised to 105 mph` yields a single mention with
  `value = 105, revised = True`, not separate `90` and `105` mentions. Otherwise
  greedy one-to-one alignment can match the *stale* value against the other
  text's figure, score `agreement`, and hide the revision entirely — which on
  `019↔020` (cos 0.993) would produce a confident CORROBORATES for a `supersedes`
  pair. Collapsing is what makes the alignment result independent of greedy
  match order.
- `as_of` — the recency of the temporal marker attached to the mention's clause
  (`since Monday`, `as of Thursday`, `by Thursday`), as a weekday ordinal;
  `None` when the clause carries no marker. **This field is load-bearing for 3
  of the 5 `supersedes` pairs** (`007↔008`, `007↔010`, and the count arm of
  `019↔020`), which differ on an aligned fact with no explicit revision wording
  and are separated only by `since Monday` → `as of Thursday`. Without it those
  three are judged `conflict` and the channel scores worse than the branch it
  replaces on the negatives it was built to fix.

  Comparison is text-internal only: the detector receives two texts and nothing
  else, so the corpus's `published` metadata is not available to it and must not
  be consulted. Weekday ordinals are compared within a single week — the
  register these reports are written in — and an ambiguous comparison yields
  `None` rather than a guess.

  Note the deliberate asymmetry with Component 3: a weekday is excluded from
  *entity-grade anchor* status there because it is not a story entity, while
  here it is read as a *temporal* marker. Different uses of the same token, both
  correct.

**Align.** Pair mentions across the two texts on identical `unit_class` and
`Jaccard(measure) >= ALIGN_FLOOR`. Many-to-many, then greedily reduced to best
matches by descending Jaccard so one mention is not double-counted.

**Judge.** Per aligned mention pair:

| condition | evidence |
|---|---|
| values differ, neither side `revised`, scopes compatible, no later `as_of` | `conflict` |
| values differ, one side `revised` or carries a strictly later `as_of` | `revision` |
| scope token sets differ | `partial` (no conflict) |
| values equal | `agreement` |

Returns `StanceEvidence(conflicts, revisions, partials, agreements)`, each a
tuple of the aligned mention pairs, so a rationale can name the fact rather than
asserting a bare verdict.

The module exposes `stance_for(a_text, b_text) -> Optional[str]` collapsing the
evidence to a single verdict for `classify_relation`. **A pair routinely yields
more than one kind of evidence, so the precedence is fixed and explicit:**

```
conflict  >  revision  >  partial  >  agreement  >  None
```

`None` means no mentions aligned at all. This ordering is what the labels
require, and both directions matter:

- `017↔019` (contradicts) yields `agreement` on `210,000 customers` **and**
  `conflict` on `65` vs `40` shelters. Conflict must win, or a real conflict is
  swallowed by an incidental agreement elsewhere in the text.
- `019↔020` (supersedes) yields `revision` on the winds arm (one collapsed
  mention, `105 / revised`, against `90`) and `agreement` on the shelter and
  customer counts, and no `conflict` — so it abstains, correctly, without needing
  conflict/revision precedence at all.

`conflict > revision` is therefore only reachable when one aligned fact genuinely
conflicts while another is revised. No labeled pair exercises that combination;
the ordering is chosen so a genuine conflict is never suppressed, and the case is
called out here so a future counterexample is recognized as one rather than
absorbed silently.

### Component 2 — `classify_relation` and the proposer

`classify_relation` gains a keyword-only `stance: Optional[str] = None`.
**`stance=None` reproduces the current rule byte-for-byte.** That is the same
contract `same_story=None` already carries, and it is what keeps the baked
39-sample fixture, `tests/test_cartographer_eval_pairs.py` and the existing
combined-detector tests valid without edits.

With `stance` supplied, the `if same_story` arm becomes:

**AMENDED during implementation — see Results.**

```
same_story and stance == "conflict"                  -> CONTRADICTS, "stance"
same_story and stance == "agreement"
        and cos >= corroborate_ceiling               -> CORROBORATES, "band"
same_story and stance in {"revision", "partial",
                          "agreement", UNALIGNED}    -> RELATED_UNTYPED, "abstain"
```

The NLI channel keeps its current priority and its current story gate. It is
untouched: the measurement above says it is not the signal for this class, but it
remains correct for the legal/securities register it was calibrated on.

`CombinedRelationProposer` wires `quantity.stance_for` by default — it is
model-free, so there is no cost argument for leaving production unfixed — and
accepts an injected provider, or `stance_provider=None` to disable. Computed
only for pairs that reach the branch, after the cosine gate.

`Sample` and `EvalSample` in `gin/cartographer/calibration_samples.py` gain a
`stance: Optional[str] = None` field, and `SampleManifest` records the stance
provider identity, mirroring how `same_story_corpus_size` was added: defaulted so
the committed 39-sample fixture keeps loading, and gated so a sample file
measured under a different stance rule cannot silently calibrate the live
pipeline.

### Component 3 — `relatedness.py` fixes

`anchor_tokens` excludes a closed calendar vocabulary (seven weekday names,
twelve month names) from entity-grade status. The docstring already says
entity-grade means *proper noun* / *dateline* / *story figure*; a weekday is none
of those, and mid-sentence capitalization — the test the function actually
applies — is satisfied by every weekday and month in English prose.

Three month names are also ordinary English words: `may` (modal), `march`
(verb/noun), `august` (adjective). Excluding them costs the anchor signal in a
story genuinely named for one — a *March* on city hall, an *August* report. The
exclusion is accepted anyway, on two grounds: it only removes *anchor-grade*
status, not the token's rare-shared-token contribution, so such a pair can still
reach `story_floor` on its other entities; and the lowercase homographs are
common enough that their document frequency puts them above the rare ceiling in
any real corpus, so they were rarely anchoring anything to begin with. The
tradeoff is recorded here so a future story-anchor miss on a calendar-named
entity is diagnosable rather than mysterious.

`make_same_story` requires the anchor token be entity-grade in **both** texts:
`(anchor_tokens(a) & anchor_tokens(b)) & rare`. A token can only anchor a
*shared* story if it carries entity signal where it is shared.

`story_floor` and `df_ceiling` keep their current values and semantics.

### Component 4 — `scripts/sweep_same_story.py` (new, writes nothing)

Scores the cross product of `story_floor` × `df_ceiling` × anchor-mode
(`union` / `intersection`, calendar on/off) against the 24 labels, printing
within-event same-story kept and cross-event false positives per cell. Exists so
the deferred threshold decision has a reproducible artifact. It never writes
a threshold file and is not wired into any eval.

### Component 5 — node5 registration

`CORPUS_NODES = tuple(REPO_ROOT / f"corpus_node{i}.json" for i in (1, 2, 3, 4, 5))`.

Consequences, each measured rather than assumed:

- `default_text_index()` 236 → 274 documents; `_rare_df_ceiling` 7 → 9. This
  re-derives `make_same_story` wherever it is built over that index —
  `scripts/regen_calibration_samples.py` most importantly. `gin/cartographer/scan.py`
  builds the predicate over the chunks under scan, so registration does not
  affect it; the anchor fixes do.
- B's dataset (`gin/frames/dataset.py`) gains **7** rows: 2 AGREE + 5 UNRELATED.
  The 12 story contradicts still drop as `schema` — `_LABEL_MAP` has no
  `(CONTRADICTS, "story")` entry, because DIVERGENT is issue_frame-only by
  design (`gin/frames/labels.py`). This is correct, not a gap to close here.
- C's calibration export (`gin/curator/calibration_export.py`) gains **19** rows
  (`supersedes` is not a classifier output). Those 19 include 12 same-story
  contradicts — the precondition `scripts/recalibrate_cheap_pipeline.py` has been
  blocked on since 2026-07-25.
- Readiness is unaffected at 25/20: `touches_bar_text` skips unresolved ids, so
  the 24 already counted; after registration they resolve and still match no bar
  text.
- `bar_chunk_texts()` and `eval_pair_keys()` are unchanged — neither derives from
  `CORPUS_NODES` content that node5 touches.

### Component 6 — `scripts/eval_node5_stance.py` (new)

The reproducible 24-pair scorer. Reads the labels, wires the real proposer,
prints the confusion matrix, within-event and overall precision, and the
per-pair verdict with the aligned fact naming its evidence. Committed so the
number in §Results can be regenerated rather than trusted.

## Measurement plan

Three changes land on one frozen number, so they ship as three commits and the
held-out 40-pair score is measured after each. A regression is then attributable
instead of a mystery.

| state | commit | held-out 40-pair score |
|---|---|---|
| baseline | `ebceb46` | 0.700 (recorded, `c30f910`) |
| + node5 registration | 1 | measure |
| + anchor fixes + sweep | 2 | measure |
| + stance channel | 3 | measure |

Thresholds stay at the baked-39 values throughout, so the score moves only via
`same_story` and `stance` flips on those 40 pairs — which is exactly the
quantity of interest.

### The pre-registered "known likely miss" did NOT occur

This spec predicted `n5_doc_011↔012` (`September 3` vs `October 1`) would move to
abstention and cost one point of `R`, on the grounds that its governing phrases
share no content tokens. **That prediction was wrong.** The pair aligns and is
typed CONTRADICTS correctly, which is why `R` is 1.000 rather than 11/12.

It aligns on a measure Jaccard of **0.071** — a single shared stem, `repair`,
across `remain closed until … September 3` and `repairs … finished before
October 1`. At `ALIGN_FLOOR = 0.05` that clears.

The symmetry is worth stating plainly, because it is the same fact seen twice:
**the loose floor saved this real conflict and caused the one stance-arm false
positive.** `n5_doc_023↔024` aligns a dockworker headcount against a commuter
delay in minutes on the same kind of thin overlap. A floor tight enough to
reject that pair would also reject this one. On this corpus the trade is
favourable — one conflict kept for one false positive — but n=24 cannot say
whether it stays favourable, and that is the generalization question this work
leaves open rather than settles.

### Over-fitting control

Node5 is synthetic and its patterns were authored by the same person who labeled
it. A signal fitted to it will look good and may not generalize — the trap
node4's purpose-built corpus fell into when it turned out trivially separable at
22/22 on every encoder. Three controls:

1. **A named, pre-registered event split.** The 19 within-event pairs span 10 of
   node5's 12 events (`crosstown_line_suspension` and `district_enrollment_report`
   have no labeled within-event pair). The aligner is developed against the first
   **7** in manifest order and never run against the held-out **3** until it is
   final:

   | held out | pairs | kinds |
   |---|---|---|
   | `lakeshore_algae_bloom` | 2 | 2 supersedes |
   | `civic_bond_audit` | 2 | 2 contradicts |
   | `stadium_capacity_ruling` | 2 | 1 contradicts, 1 corroborates |

   6 held-out pairs covering all three kinds; 13 development pairs (9
   contradicts, 3 supersedes, 1 corroborates). Naming them here rather than
   deriving them later is the point — a rule chosen after the fact can be chosen
   to flatter. The development set carries only **one** corroborates pair, which
   is thin and is stated rather than smoothed over: it means the `partial`
   judgment is effectively validated on the held-out `stadium_capacity_ruling`
   pair, and a failure there is a genuine finding about the rule, not noise.
2. It is reported on node1–4 and the 45-pair eval set, where it must not regress
   anything.
3. The `scope`, revision and calendar vocabularies are module constants reviewed
   as data, so what the signal keys on is inspectable rather than buried in
   regexes.

## Success criteria

Pre-registered. Report each number whichever way it moves.

**The metric, stated exactly**, because `12/19` is a *precision* figure and a rule
that can abstain will otherwise look better simply by emitting fewer edges. Three
numbers, all reported, none tradeable against another:

| name | definition | value at `ebceb46` |
|---|---|---|
| `P` | of the within-event pairs typed CONTRADICTS, the fraction labeled `contradicts` | 12/19 = 0.632 |
| `R` | of the 12 labeled `contradicts`, the fraction typed CONTRADICTS | 12/12 = 1.000 |
| `P_all` | same as `P` over all 24 pairs, so stage-1 false positives count against stage 2 | 12/24 = 0.500 |

- **Stage 2:** `P` and `P_all` must both strictly improve, with `R >= 0.75`
  (at most 3 of the 12 conflicts lost). Trivially clearing `P` by abstaining on
  almost everything is excluded by the `R` floor; trivially holding `R` by typing
  everything CONTRADICTS is what the current branch does and is excluded by `P`.
  No target above those floors is pre-committed.
- **Four-way confusion is reported but not gated.** `036↔037` (`corroborates`,
  scopes differ) will be judged `partial` and abstain to RELATED_UNTYPED rather
  than CORROBORATES — an incorrect 4-way answer that is nonetheless the right
  CONTRADICTS decision. The gate is on the contradicts channel, which is what the
  defect is in; the full matrix is printed so the corroborates gap is visible
  rather than hidden by a headline number.
- **Stage 1:** cross-event false positives **0/5**, within-event same-story
  **19/19**, with `story_floor` and `df_ceiling` unchanged.
- **No regression on frozen surfaces:** the 45-pair eval set, the 14-pair bar
  pin (`tests/test_cartographer_eval_pairs.py`) and the scan gold eval all hold.
- **Held-out generalization:** stage-2 accuracy on the 6 pairs from the 3
  held-out events is reported alongside the 13 developed against, and the gap is
  stated.
- **Additivity:** `stance=None` reproduces current behavior exactly; the baked
  39-sample fixture loads unchanged.
- **Thresholds untouched:** `data/cartographer_thresholds.json` is byte-identical
  after this work.

Explicitly **not** success criteria: any recalibrated threshold value, any
encoder metric, and any improvement on node1–4 — those corpora exercise a
different phenomenon and the honest expectation there is *no change*.

`n5_doc_011↔012` (`September 3` vs `October 1` — date measures whose governing
phrases, `remain closed until` and `repairs … finished before`, share no content
tokens) is a **known likely miss**. It is currently scored correct by accident and
is expected to move to abstention, costing one point of `R`. The `R >= 0.75` floor
budgets for it and two more like it; papering over it by special-casing dates to
align on nothing but `unit_class` would re-introduce the naive rule that already
scores 12/19.

## Failure modes

| Condition | Handling |
|---|---|
| `P` or `P_all` fails to improve, or `R` falls below 0.75 | Report it and stop. The abstain fallback still removes wrong edges, which is a defensible partial outcome; do not tune the aligner against the labels to clear the bar. |
| The 3 held-out events score far below the 7 developed against | Report the gap explicitly as an over-fitting finding. That is a real result about the corpus, and it is what the split exists to detect. |
| A pair yields both `conflict` and `revision` evidence | Precedence says `conflict`. Record the pair — no labeled pair exercises this, so the first one that does is new information about the rule. |
| A `supersedes` pair reads `agreement` and so emits CORROBORATES | The `agreement` arm is the only place this channel makes a positive claim instead of abstaining, so this is worse than abstaining. Measured as **floor-dependent**: at a loose alignment floor the revised fact aligns and all five `supersedes` pairs read `revision`; at a tight one the revision is missed and the equal counts carry the pair to `agreement`. Pinned by a test over all five, so a later floor change cannot silently reintroduce it. |
| Held-out 40-pair score drops below 0.700 | Report which of the three commits moved it, and by how much. Do not compensate by writing thresholds — that is the next spec's decision, made with this number in hand. |
| An anchor fix breaks a node1–4 or gold pair | Investigate before reverting: the union anchor may have been carrying a pair for the wrong reason, which is a finding, not a regression to paper over. |
| The aligner needs an event-specific rule to work | Stop. That is fitting the corpus, and the constant vocabularies exist so this is visible when it happens. |
| Registration changes the readiness count | Investigate — the analysis says it should not, so a change means an assumption above is wrong. |

## Testing

Model-free throughout except the measurement runs.

- **`quantity.py`** — extraction unit tests per `unit_class`, including `as_of`
  weekday ordinals and the ambiguous-comparison → `None` case; alignment tests
  including the negative cases (different measure, different scope); judge tests
  for all four evidence kinds; a precedence test asserting
  `conflict > revision > partial > agreement > None` directly; the two mixed-fact
  conflicts (`005↔006`, `017↔020`) asserted as `conflict` specifically to pin
  that a pair-level veto has not crept back in; and `017↔019` asserted as
  `conflict` to pin that an incidental `agreement` cannot swallow it.
- **`classify_relation`** — `stance=None` reproduces the current truth table
  exactly (a table-driven test over the existing cases); each new arm asserted.
- **`relatedness.py`** — calendar words are not entity-grade; a proper noun in
  one text does not anchor a common noun in the other; the 19 within-event pairs
  still pass; all 5 cross-event pairs now fail.
- **`calibration_samples.py`** — the committed 39-sample fixture loads with
  `stance` defaulted; the manifest gate rejects a mismatched stance provider.
- **`text_index.py`** — node5 ids resolve; index size and derived ceiling
  asserted; `bar_chunk_texts()` unchanged.
- **Regression** — full suite (665 passed / 16 skipped at `ebceb46`), plus the
  bar pin and the layering check in both directions.

## Out of scope

- Recalibrating or writing `data/cartographer_thresholds.json`. The 19 new rows
  unblock it; that is the next spec.
- The `northgate` authoring question. `n5_doc_007↔008` and `007↔010` are labeled
  `supersedes` but authored as `conflict`; `test_update_pairs_are_ordered_in_time`
  correctly blocked flipping them to `update`, because CentralWire is the earlier
  report and that convention requires the reviser at `pair[0]`. The prose
  under-determines it. Left as `conflict` with the OPEN note; sharpening it is an
  authoring decision, not a code change, and it does not affect any number here —
  the labels are what this spec scores against, not the authored intent.
- node1–4 content, `gin/frames/` training, the escalation bar, any encoder work.
- Adding `(CONTRADICTS, "story")` to `_LABEL_MAP`. DIVERGENT is issue_frame-only
  deliberately; changing it is a frames-detector decision with its own rationale
  to overturn.

## Open questions

None blocking. Deferred with artifacts rather than prose:

- `story_floor` and `df_ceiling` values — deferred; `sweep_same_story.py` is the
  artifact the decision will be made from, at a corpus larger than n=24.
- Whether `supersedes` should become a labelable relation in the curator UI —
  inherited from node5's spec, still not a 4-way training class, and now
  additionally relevant because the stance channel distinguishes `revision`
  evidence explicitly.

## Results (measured 2026-07-26)

Implemented on branch `stance-channel`, commits `2490470..HEAD`. Two of this
spec's components changed during execution and both changes are recorded below
rather than quietly folded in.

### Stage 2 — the stance channel

Two figures, answering different questions. Reporting only one would either
credit the stance channel with the NLI channel's errors or blame it for them.

| | baseline (`ebceb46`) | stance arm isolated | end to end, real models |
|---|---|---|---|
| `P` within-event precision | 0.632 | **1.000** (tp 12, fp 0, fn 0) | **0.857** (tp 12, fp 2) |
| `R` recall | 1.000 | **1.000** | **1.000** |
| `P_all` incl. cross-event | 0.500 | **0.857** | **0.750** (tp 12, fp 4) |

**Pre-registered bar: PASS** on both — `P` and `P_all` each strictly improve and
`R` is 1.000, well above the 0.75 floor. Not one of the 12 real conflicts was
lost. Reproduce with `scripts/eval_node5_stance.py` (end to end) and
`tests/test_cartographer_stance_node5.py` (isolated).

**The attribution is the finding.** Of the four residual false positives end to
end, **three are the NLI channel, not the stance arm**:

| pair | channel | `p_contra` | gold | stance |
|---|---|---|---|---|
| `n5_doc_007↔008` | nli | 0.980 | supersedes | revision |
| `n5_doc_036↔037` | nli | 0.983 | corroborates | unaligned |
| `n5_doc_023↔026` | nli | 0.692 | unrelated | None |
| `n5_doc_023↔024` | stance | 0.615 | unrelated | conflict |

The first two are **exactly the pairs this spec's own §"NLI cannot carry the
branch" table identified as the two highest `p_contra` in the set**. The spec
concluded NLI is not the signal for this class, then deliberately left NLI's
priority over the band intact. The predicted cost materialised precisely: the
stance channel correctly says `revision` and `unaligned`, and NLI overrules it.
**Whether the stance channel should outrank NLI on same-story pairs is the
sharpest question this work leaves open**, and it is a one-line change with its
own eval to run — deliberately not made here, because the spec fixed NLI's
priority and moving it is a separate decision.

`n5_doc_023↔024` is the single stance-arm false positive, pre-registered: it
needs stage 1's union anchor *and* `ALIGN_FLOOR = 0.05` at once. It is the
concrete instance of the low-floor hazard this spec flagged.

**A sharper statement of the same four rows, because `n5_doc_023↔026` appears
twice above under different runs and that is easy to misread as double
counting:** of the four end-to-end false positives, **two are NLI-only**
(`007↔008`, `036↔037` — the band never gets a chance to fire because the NLI
channel's priority preempts it), **one is stance-only** (`023↔024`), and
**one is overdetermined** — `023↔026` fires through the NLI channel at
`p_contra` 0.692 with the real models (the row above), *and* it fires through
the `band` channel under `stance=None` when NLI is injected to abstain (the
stage-1-anchor-findings write-up's isolated measurement). Both are genuine,
independent routes to the same wrong edge on the same pair; it is attributable
to both channels at once, not to whichever one happens to be measured first.

### Over-fitting control

Development (13 pairs, 7 events) `P` **0.900**; held out (6 pairs, 3 events) `P`
**0.750**; gap **−0.150**, driven entirely by the `n5_doc_036↔037` NLI false
positive. Recall is 1.000 on both halves.

**Caveat, stated because it weakens the check:** the planning session's
exploratory sweep included the held-out events, so this is a weaker independent
check than the named split implies. `ALIGN_FLOOR` selection was still computed on
the development pairs alone.

### `ALIGN_FLOOR`

Settled at **0.05**, selected on the 13 development pairs only. Development
precision was **1.000 at every floor from 0.02 to 0.25**, so recall was the
binding constraint and the `P` half of the bar cleared trivially — recorded so
the headline `P` is not read as evidence the aligner is well constrained. At
0.05 alignment is close to "same unit class plus one shared stem"; the `scope`
and `revision` vetoes do the discriminating.

### Amendment to Component 2 — `None` versus `UNALIGNED`

Component 2 specified that `stance=None` should abstain. Implementation found
`None` was overloaded: it meant both "no provider wired" (which must reproduce
the pre-stance branch, or the baked 39-sample fixture and the 14-pair bar pin
break) and "the provider ran and aligned nothing".

Measurement then found that **three of the four gold `contradicts` pairs which
pass the story gate state no quantities at all** — `hf_af_staff↔hf_af_tenants`
and `hf_kc_inspection↔hf_kc_tenants` have zero mentions on both sides,
`disc_mer_pr↔disc_mer_complaint` has zero on one. Housing habitability disputes
and a securities PR versus a complaint contradict *qualitatively*. node5 is
quantity-dense by construction and hid this entirely.

Measured ad hoc during implementation (Task 9), before the rule was chosen; the
committed `scripts/eval_node5_stance.py` implements only the adopted rule, so the
two rejected rows are not reproducible from it:

| rule | node5 `P` | `P_all` | gold contradicts |
|---|---|---|---|
| `None` → CONTRADICTS | 0.923 | 0.706 | 4/4 |
| `None` → abstain (as specified) | 1.000 | 0.923 | **1/4** |
| **`UNALIGNED` sentinel (adopted)** | **1.000** | 0.857 | **4/4** |

The adopted rule: the channel may override the caller's default **only when it
had quantitative claims on both sides to compare**. `None` means nothing to
judge → defer to the pre-stance branch; `UNALIGNED` means it looked and found no
shared fact → abstain. `classify_relation` needed no logic change — `UNALIGNED`
already fell through to the abstain return.

**The cost of that choice is visible and recorded:** preserving the `None` path
also preserves the degenerate branch for quantity-free pairs, which is why
`n5_doc_023↔026` is still typed CONTRADICTS through the `band` channel. Pinned
by name in `tests/test_cartographer_stance_node5.py`.

### Stage 1 — WITHDRAWN

The `anchor_tokens` calendar-word exclusion shipped (`231d55d`). The
union → intersection change **did not**: it regresses two pre-registered gold
`contradicts` pairs, because `anchor_tokens` treats sentence-initial
capitalization as carrying no entity signal and `Northwind Systems reported…`
hides the entity that `The complaint alleges Northwind…` exposes. Full write-up
and the reproducible design space:
`docs/superpowers/specs/2026-07-26-stage1-anchor-findings.md` and
`scripts/sweep_same_story.py`.

Withdrawal was cheap because stage 2 absorbs most of stage 1's residue: four of
five cross-event pairs still pass the story gate, and the stance channel abstains
on two of them.

### Frozen surfaces and regression

- **14-pair escalation-bar pin** (`tests/test_cartographer_eval_pairs.py`): 7/7,
  **unedited throughout**.
- **45-pair eval set and the scan gold eval:** both hold, via the full suite
  below. Named explicitly because this spec's success criteria list them
  separately from the bar pin, and "the suite is green" is a weaker statement
  than "these two surfaces were checked".
- **Held-out 40-pair score, shipped thresholds:** 0.700 baseline → **0.725**
  after node5 registration → **0.725** after the stance channel.

  **The score is unchanged, but not for the reason a previous version of this
  document gave.** That version said stance did not move the number because
  "those pairs are largely not same-story, so the arm this work changed rarely
  fires on them" — which is false on both halves. 9 of the 40 held-out pairs
  ARE same-story, and stance is non-`None` on 5 of them, so the arm fires
  often, not rarely. The real reason the score holds: **all 9 same-story
  held-out pairs are gold `contradicts`**, and stance reads `conflict` on 5 of
  them and `None` on the other 4 — `conflict` types CONTRADICTS through the
  new stance channel, and `None` types CONTRADICTS through the pre-stance band
  fallback, so both paths give the same, correct answer on every one of the 9.
  That is a stronger claim than "the arm rarely fires," and it is worth stating
  plainly that the old wording would have read exactly the same — "unchanged"
  — even if the stance channel were badly broken, which is why it was worth
  checking directly rather than trusting the unchanged headline number.
  (This also required threading `stance` through `_score_held_out` in
  `scripts/recalibrate_cheap_pipeline.py`, which previously called
  `classify_relation` without it — so the number itself was correct by
  coincidence, but not honestly measured, until that call was fixed.)
- **`data/cartographer_thresholds.json`: byte-identical.** No task wrote it.
- **`story_floor` (2) and `_rare_df_ceiling`:** unchanged.
- Full suite **732 passed / 16 skipped / 0 failed** (was 665 at `ebceb46`).
- `data/calibration/samples.json` regenerated: 150 rows, manifest
  `stance_provider: quantity.stance_for`, corpus 274 docs, ceiling 9. Stance
  distribution across rows: None 130, conflict 16, unaligned 3, partial 1.

### Not measured, by design

Any recalibrated threshold value. The 19 new calibration rows unblock
`scripts/recalibrate_cheap_pipeline.py`, whose STATUS note now records that its
precondition is satisfied — that is the next spec's work, and recalibrating under
a just-changed pipeline would restate the change rather than evaluate it.
