# Real-Text Divergence Generalization (two-node corpus)

**Status**: CONFIRMED — measured in full DB eval
**Depends on**: [Phase 3 divergence correctness](nc_phase3_divergence_correctness.plan.md) (PROMOTED on synthetic corpus)
**Baseline**: two-node run `20260704T105554Z` — SEAR `divergence_fidelity` 0.333 (1 of 3 pairs)
**Result run**: `20260705T043114Z` — SEAR `divergence_fidelity` **1.000** (3 of 3 pairs)

Phase 3 promoted the divergent-decode mechanism (`compute_divergence_zones`,
`min_span_len` auto-close, EOS gating) against a synthetic corpus where every
contradicts pair shares a lede and forks at one sentence (e.g. "142" vs "98"
treated). This doc covers the gap Phase 3 didn't know existed: real editorial
divergence — an institutional statistic vs. a grassroots reframing — usually
shares **no** structure at all, and the Phase 3 machinery silently refused on
exactly that case.

---

## 1. Background: how the two-node corpus got here

A second corpus (`corpus_node1.json` / `corpus_node2.json`, institutional vs.
grassroots framing, real fetched text) was built to stress-test whether Phase
3's synthetic-corpus promotion generalizes. Three `contradicts` pairs
(emissions, wildfire, water) were wired via `data/corpus_edges.yaml` and
`data/eval/queryset_twonode.yaml`.

**Retrieval-side gap (prior session, already fixed — see
[[phase3-divergence-correctness]] follow-on).** The `_divergence_relevant`
lexical gate (max_sentence_score + matched_keyword_count) flipped real
minority-side chunks to convergent mode because the grassroots reframing uses
different vocabulary than the query. Fixed with an IDF-weighted relevance gate
(`corpus_idf`, `idf_weighted_relevance` in `gin/corpus/relevance.py`,
`DIVERGENCE_IDF_FLOOR=0.13`) so mode selection is no longer purely
keyword-overlap. This got all three pairs correctly routed to divergent mode
— but exposed the decode-side bug below.

## 2. Root cause: `compute_divergence_zones` returns empty for reframing pairs

**File**: `gin/corpus/divergence.py`

`compute_divergence_zones` marks a divergence point only where **index-aligned
sentences differ AND share >= 3 words** (`_word_overlap`, exact match, no
stemming). This assumes both sides of a contradicts pair share a lede and
fork at a specific sentence — true for synthetic "same story, different
number" pairs, false for real reframing pairs:

| pair | left (institutional) | right (grassroots) | shared words | zone? |
|---|---|---|---|---|
| emissions | "...cuts to predicted 2030 greenhouse gas emissions of roughly 28 percent..." | "...stopped or delayed greenhouse gas pollution equivalent to roughly one-quarter..." | greenhouse, gas, roughly, ... (>=3) | marked |
| wildfire | "56,580 wildfires burned 2,693,910 acres..." | "Elderly, immunocompromised...populations face...risk from wildfire smoke" | only "and" (`wildfire` != `wildfires`, no stemming) | **empty** |
| water | "...snowpack held a snow water equivalent of 61.1 inches..." | "Disadvantaged...communities...affected by water shortages..." | only "water" | **empty** |

With `divergence_starts` empty for a pair, the cascade is fatal
(`materialize_synthesis_bundle` in `gin/corpus/materialize.py` +
`sear/processor.py`):

1. The forbidden-tail net treats every doc-unique sentence as a forbidden
   "tail" (mayor-quote-style filler) **unless** it falls inside a divergence
   zone. With zero zones, this net swallows the pair's own anchor sentences.
2. `_position_start_permitted` (`sear/processor.py`) rejects every span start
   for both docs.
3. `_allowed()` returns `{}` in `BOUNDARY` mode while EOS is blocked
   (`block_eos_until_groups_satisfied`, required because `required_doc_groups`
   is non-empty for a divergent bundle with a contradicts pair) -> all logits
   masked to `-inf` -> decode refuses ("sources do not support an answer").

Emissions survived only because its aligned sentences happen to share >= 3
words. This was never a `min_span_len` or EOS-machinery bug — that
machinery (Phase 3's own fixes) is sound; the zone it depends on was never
being produced upstream.

## 3. Fixes

### Fix 1 — per-pair fallback zone for structurally-dissimilar pairs

**File**: `gin/corpus/divergence.py` (`compute_divergence_zones`)

When the index-aligned >= 3-word-overlap test marks nothing for a
`contradicts` pair, mark an anchor sentence per side as its own divergence
zone. Rationale: in a reframing pair there is no single "fork sentence" — the
divergence *is* the framing choice on each side. Only fires when the aligned
test found nothing, so synthetic/aligned pairs (emissions included) are
byte-for-byte unaffected.

**Anchor selection (guards against chunk-level tail bloat).**
`compute_divergence_zones` takes an optional `sentence_scorer`. With one
(materialize passes an IDF-weighted query-relevance closure — IDF so the
singular/plural fold catches `wildfire`~`wildfires`, matching the divergence
gate), the fallback marks only the single most query-relevant sentence per
side. Without one it marks every sentence start (backward-compatible). These
coincide for the current single-sentence corpus, but the distinction is
load-bearing the moment a real multi-paragraph chunk is ingested: "mark every
sentence" would turn filler/tail lines ("we thank our volunteers") into
divergence-steered starts — the exact forbidden-tail failure this fallback
exists to prevent, resurfacing one level down at chunk granularity. Guarded by
`tests/test_divergence.py::test_multi_sentence_fallback_anchors_on_relevant_sentence`.
(This closes former open question #2; real multi-paragraph validation still
pending — see §6.)

### Fix 2 — divergent `max_tokens` budget was tuned for synthetic sentence lengths

**File**: `gin/corpus/generate.py` (`_resolve_decode_params`)

`max_tokens = 40 + 25 * len(required_doc_groups)` gave 65 tokens for one
group. Real institutional sentences run ~50-55 tokens; the first extract
consumed the budget and the second (grassroots) side truncated to a fragment
("Elderly, immunocompromised" instead of the full sentence). Raised to
`40 + 90 * len(required_doc_groups)` (130 for one group) — a ceiling, not a
fixed cost: `stop_when_groups_satisfied` still fires EOS the instant both
sides are quoted, so aligned/synthetic pairs that finish early are unaffected.
**(Update — §6 #5 resolved.)** The `90` was later put on a measured basis: the
worst-case full divergent decode across all pairs is 97 tokens, so the
coefficient dropped to `80` (120 for one group, 24% headroom). See §6 #5 for the
per-pair token table.

## 4. Verification (three layers)

1. **Deterministic repro** (no llama.cpp): whitespace-tokenizer rebuild of the
   real wildfire/water chunk pairs through `compute_divergence_zones` directly.
   Pre-fix: empty zone, both gold anchors `(0,0)`/`(1,0)` forbidden. Post-fix:
   two-sided zone, anchors un-forbidden.
2. **Real Mistral-7B dry run** (hand-built `SynthesisBundle`, no DB): drove the
   actual `ExtractiveCopyConstraint` + `BiasedGINLogitsProcessor` through
   `llm.create_completion` on the wildfire/water pairs. Pre-Fix-2: both sides
   quoted but grassroots side truncated to a fragment. Post-Fix-2: both
   sentences render in full, `quoted_docs=[0,1]`, `groups_satisfied=True`.
3. **Full DB eval** (real Postgres + real retrieval + real model,
   `scripts/eval_run.py --queryset data/eval/queryset_twonode.yaml`): see
   results below.

Unit regression:
`tests/test_divergence.py::test_divergence_fallback_for_structurally_dissimilar_pair`
and `::test_multi_sentence_fallback_anchors_on_relevant_sentence` (both new).
Full logic suite: 155 passed. (The 3 DB-dependent tests in `test_ingest.py` /
`test_retrieve.py` skip when no Postgres is reachable and error with
`type "vector" does not exist` when a non-pgvector Postgres is — both expected;
they are not part of the logic suite.)

## 5. Measured results

Two-node eval (`data/eval/queryset_twonode.yaml`, Mistral-7B-Instruct-v0.3-Q6_K,
overlap verifier):

| Metric | Before (`20260704T105554Z`) | After (`20260705T043114Z`) |
|---|---|---|
| divergence_fidelity (SEAR) | 0.333 | **1.000** |
| fabrication_rate (SEAR) | 0.000 | 0.000 |
| fabrication_rate (RAG) | 0.438 | 0.412 |
| gold_chunk_coverage (SEAR) | 0.600 | 1.000 |
| cross_node_within_ratio (SEAR) | 1.000 | 1.000 |
| wall-clock per query | 34.3s | 51.8s |

All three contradicts pairs (emissions, wildfire, water) now surface both
sides; fabrication stays at zero against RAG's ~41%. Wall-clock roughly +50%
for divergent queries, tracking the larger `max_tokens` ceiling — decode still
terminates on `stop_when_groups_satisfied`, so this is a ceiling increase, not
a per-query fixed cost.

## 6. Open questions / next steps (not yet done)

Ranked by what would most change confidence in this result:

1. **Generalize past 3 pairs.** The whole real-text divergence claim rests on
   three hand-picked pairs from a 19-document corpus. Add more contradicts
   pairs across more topic domains (ideally with a third framing style, not
   just institutional-vs-grassroots) to check the fallback zone and IDF gate
   aren't overfit to this specific corpus's vocabulary distribution.
2. **[GUARD LANDED, decision resolved] Multi-sentence fallback.** The decision
   point — "is whole-chunk actually wrong, or does the forbidden-tail net catch
   the noise downstream?" — is settled empirically. On a realistic 4-sentence-
   per-side wildfire pair (lede + supporting detail + org-credit + tail stat),
   whole-chunk marking leaves **3 irrelevant sentences per side in the zone, and
   all 3 are immune to the forbidden-tail net** — because that net skips any
   sentence already in the divergence zone (`if key in all_divergence:
   continue`). So the net structurally *cannot* filter in-zone noise; whole-
   chunk is unsafe by construction, not just untidy. The IDF-anchored fallback
   (§3 Fix 1) narrows to the one relevant sentence per side (grassroots filler
   scores 0.0 under IDF, anchor 0.262), and the filler then falls back under the
   forbidden net. Guarded by the realistic-prose unit test
   (`test_multi_sentence_fallback_anchors_on_relevant_sentence`), which asserts
   both the immunity failure mode and the narrowed fix.
   **[RESOLVED] Validated end-to-end.** `data/fixtures/wildfire_multipara.yaml`
   ingests two *actual* multi-sentence chunks (one paragraph per side: an
   on-topic anchor sentence plus supporting detail and an org-credit/tail line,
   with the distinctive "Cascade Ridge fire" entity only on the anchors) wired
   by a `contradicts` edge; `data/eval/queryset_multipara.yaml` drives the
   query. Run `20260705T065559Z` (real DB + real Mistral) decoded the **anchor
   sentence on each side, not filler/tail**:
   - institutional: *"The Cascade Ridge fire burned 214,000 acres…"* (sentence 0),
     not the suppression/dispatch tail;
   - grassroots: *"Elderly, immunocompromised, and low-income residents downwind
     of the Cascade Ridge fire face the heaviest burden from wildfire smoke
     exposure."* (sentence 1), **not** the "Neighborhood clinics" filler lede,
     the "Community groups distributed air purifiers" line, or the "The coalition
     thanks the volunteers" tail.

   `divergence_fidelity` 1.000, `fabrication_rate` 0.000, `gold_chunk_coverage`
   1.000. The IDF anchor scorer picks the anchor and the forbidden-tail net
   catches the filler exactly as the unit test predicted, now confirmed on a
   real decode. This is the "do this **before** scaling the corpus" gate — now
   closed.
3. **Generalize past 3 pairs (couples with the Cartographer — see §7).** The
   real-text claim rests on three hand-picked pairs from a 19-document corpus,
   and all three share a hidden trait: both sides respond to the *same
   underlying event*, weighting it differently. The IDF floor (0.13) is
   already thin — measured minority-side relevance was emissions 0.328, water
   0.445, but **wildfire 0.165, only ~0.035 above the floor**. A sparser-
   overlap domain would likely dip under it. Expansion should break assumptions
   the current three don't test, **one variable per round** so a floor failure
   is attributable:
   - *New framing style* (round 1): adversarial/legal register, not advocacy-
     vs-institution — e.g. corporate press release vs. SEC filing / complaint on
     one event. Different vocabulary distribution than either current node;
     tests whether 0.13 is a climate-corpus artifact.
   - *New topic domain* (round 2): a field where the two sides share almost no
     surface vocabulary — e.g. housing (zoning-board technical vs. tenant-
     organizing language). Harder on the IDF gate than climate, where
     "greenhouse"/"wildfire"/"water" hand you accidental overlap.
   Hand-curating edges does not scale; this is fundamentally the Cartographer's
   job (§7).
4. **[RESOLVED] `supported_irrelevance_rate` 0.0 -> 0.200 (SEAR).** Root-caused
   and fixed; re-run `20260705T061539Z` restores it to **0.000** with
   `divergence_fidelity` 1.000 and `fabrication_rate` 0.000 intact (bonus:
   `failure_recall`/`failure_precision` now 1.0 — the out-of-scope probe is
   correctly classified as a refusal).

   **Root cause.** `tn_out_of_scope_referendum` ("By how many votes did the
   harbor district referendum pass?") retrieves the *synthetic* election chunks
   `election_centralwire:0` / `election_metrodaily:0` (the two-node eval runs on
   a mixed DB), which carry a `contradicts` edge on *turnout* (61% vs 58%). The
   query is query-relevant to those chunks (both are about the referendum), so
   the IDF divergence gate correctly routes to divergent mode and decodes both
   turnout sides. But the emitted turnout claims don't answer a *vote-margin*
   question, so SEAR should refuse — and didn't. The refusal gate
   `_claims_query_relevant` (`gin/eval/arms.py`) let them through: its
   fallback branch checked whether the query keywords appeared *anywhere in the
   cited chunk* (which is on-topic — it contains "harbor district referendum")
   and whether the claim was a substring of it. A multi-sentence chunk that is
   on-topic elsewhere thus vouched for an off-topic extracted sentence.

   **Fix.** `_claims_query_relevant` now judges the **claim's own text**, never
   the cited chunk: a claim qualifies on direct token overlap OR by sharing a
   *normalized* query keyword (`shares_query_keyword` in `relevance.py`, the
   same singular/plural fold — `wildfire`~`wildfires` — the divergence gate
   uses). Every legitimate divergence claim shares such a keyword within its own
   text (`wildfire`, `water`, `greenhouse`/`gas`), while both turnout claims
   share none -> refuse. Regression:
   `tests/test_eval_arms.py::test_claims_query_relevant_refuses_out_of_scope_with_contradicts_edge`.

   **Corpus nuance worth recording.** The task framing assumed "the chunks only
   give turnout %", but `election_centralwire:0` in fact also contains the
   convergent lede *"The harbor district referendum passed by 842 votes"* — the
   literal answer. The divergent decode never surfaced it because that sentence
   is *shared* (non-divergent), so the divergence machinery quoted the diverging
   turnout sentences instead. Refusing is the correct conservative behavior for
   the answer SEAR actually produced (off-topic turnout spans); a
   convergent-mode "842 votes" extraction would be a separate sentence-selection
   change, out of scope here and in tension with the queryset's
   `expectation: out_of_scope` label. The refusal fix is a post-decode guard: a
   pre-decode retrieval-relevance refusal cannot fire here because the retrieved
   chunks genuinely ARE query-relevant.
5. **[RESOLVED] `max_tokens` ceiling now measured.** Tokenized every two-node +
   synthetic divergence pair with the Mistral tokenizer (per-side extracted
   sentences and the actual full decodes from run `20260705T043114Z`):

   | pair | left tok | right tok | full decode |
   |---|---|---|---|
   | tn_water (worst) | 55 | 36 | **97** |
   | tn_emissions | 53 | 34 | 93 |
   | tn_wildfire | 51 | 26 | 83 |
   | syn_incident_treatment | 15 | 14 | 35 |
   | syn_incident_arrests | 12 | 12 | 30 |
   | syn_election_turnout | 11 | 11 | 28 |

   Overhead for `" | [1] "` (delimiter + cite marker) is 6 tokens. **Worst-case
   full divergent decode = 97 tokens; longest single extracted sentence = 55.**
   The old `40 + 90*n` = 130 left 33 tokens (34%) unused on the worst case —
   confirming the "picked to clear one truncation" suspicion. Lowered to
   `40 + 80*n` = **120** for one group: 24% headroom over the measured 97, and
   the per-group 80 + shared 40 base bounds the (currently unobserved) two-pair
   case at 200 ≥ 2·97. Still a ceiling, not a fixed cost — `stop_when_groups_satisfied`
   fires EOS the instant both sides are quoted. `test_generate.py::test_divergent_with_groups_blocks_eos`
   pins the new value; re-run `20260705T064533Z` confirms all three pairs still
   render in full (`divergence_fidelity` 1.000, `fabrication_rate` 0.000,
   `supported_irrelevance_rate` 0.000 — the water pair, the 55-token worst case,
   renders both sides complete with both cites, no fragment truncation).
6. **Cross-model check.** Verified only on Mistral-7B-Instruct-v0.3 (Q6_K, CPU,
   temperature 0, greedy). No evidence yet the fix's behavior (or the
   IDF-floor / word-overlap constants) generalizes across model families.
7. **Revisit output fluency.** Divergent answers are two extracted sentences
   joined by a bare `|` delimiter — grounded and citable but not natural
   prose. Inherent to the extractive design (Flagged Generation / Mode 2 is
   the registered-but-unimplemented path for this), not a regression, but
   worth tracking as a product-quality gap separate from correctness.

## 7. Forward: does this unlock Bookkeeper + Cartographer planning?

Per [GIN_Session_Synthesis_v1.md](GIN_Session_Synthesis_v1.md), the two-node
divergence demo was "the empirical keystone… until that number exists, this is
architecture; after it, a record." That number now exists (§5), so the gate to
Phase 2 (Bookkeeper separation) / automated Cartographer discovery is
genuinely open. But the sequencing matters:

- **Reasoning-layer robustness gates both.** This whole result runs on
  hand-curated, fully-trusted edges. The moment a Cartographer proposes edges
  automatically, edge **quality** (precision/recall) becomes a live variable
  the reasoning layer has never been stress-tested against — today it assumes
  every `contradicts` edge is real and query-relevant. Open questions #2/#4
  are the foundational cracks; close them before adding a layer whose value is
  feeding the reasoning layer *more and noisier* edges. The architecture's
  falsifiability-by-layer only holds if each layer is independently sound
  first.
- **Cartographer before Bookkeeper.** The Bookkeeper adjudicates Cartographer
  proposals; there is nothing to gate until proposals exist. A minimal
  Cartographer (the cheap relatedness gate + a `contradicts` proposer over the
  existing corpus) is also the only thing that makes open question #3
  (more pairs / domains) tractable without hand-curation. Plan the Cartographer
  first, with the Bookkeeper's admission interface (anchor verification, DAG
  invariants, provenance stamp) sketched alongside.
- **Recommended order:** (a) close #2 real-corpus validation + #4 root cause;
  (b) minimal Cartographer to generate edges at a larger corpus scale;
  (c) Bookkeeper as the admission gate once proposals exist to gate;
  (d) Phase 3 federation. Corpus expansion and the Cartographer are coupled —
  but reasoning robustness gates the whole chain.

### 7.1 Two things to confirm *before* the corpus grows (verified this session)

- **Edges are chunk-granular; the anchor sentence is not stored.** SEAR's own
  attribution is fine — `_close_span` (`sear/processor.py`) records sources as
  `(chunk_index, token_start, token_end)`, i.e. token-granular within a chunk,
  so provenance already resolves below the sentence. **But the graph layer is
  chunk-granular**: an `EdgeRecord` links `src_chunk_id`/`dst_chunk_id`, and a
  warm chunk id is `<doc_id>:<index>`. While chunks are single-sentence, edge
  == sentence and nothing is lost. The moment chunks are multi-sentence, *which
  sentence the `contradicts` edge is actually about* is **not** in the graph —
  the reasoning layer re-derives it every query via `compute_divergence_zones` +
  the IDF anchor scorer (§3). That makes the #2 fallback-anchor problem a
  **Bookkeeper/Cartographer design decision**, not only a decode concern:
  either (a) keep divergence-participating chunks ~sentence-sized, or (b) have
  the Cartographer propose and the Bookkeeper stamp **sentence-level anchors**
  (token offsets) on `contradicts` edges, so the anchor is admitted graph
  state, not re-derived by a heuristic on every read. Recommend deciding this
  before multi-sentence ingest, because retrofitting anchor offsets onto an
  existing edge table is migration pain.
- **The same IDF signal is now load-bearing in two reasoning paths.**
  `idf_weighted_relevance` is used by the retrieval-side divergence *gate*
  (`retrieve.py`: `_divergence_relevant` / `_pair_divergence_ok`, floor 0.13)
  **and** the decode-side fallback anchor scorer (`materialize.py`). A
  keyword/IDF-based Cartographer relation-finder would be a **third** consumer
  of the same signal — so the "is 0.13 a climate-corpus artifact" overfitting
  risk (#3) would show up in *relation detection* too, not just gating and
  anchoring. If/when the Cartographer is built, it should be tested for edge
  precision/recall on the new framing styles (§6 #3) *independently*, so a
  shared-IDF blind spot can't hide behind "just noisy data" once corpus size
  grows. No shared code path exists yet (Cartographer unbuilt); this is a
  constraint to carry into its design, not a current bug.

## 8. Out of scope (this doc)

- RAG arm changes, Flagged Generation arm implementation
- Verifier changes
- Retrieval-side IDF gate implementation detail (see prior-session notes;
  summarized in section 1 only as context for why pairs reach divergent mode
  at all)
