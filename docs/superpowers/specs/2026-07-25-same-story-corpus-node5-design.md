# Design: Same-Story Corpus (node5) — Propositional Conflict and Its Negatives

**Date:** 2026-07-25
**Status:** Approved (design), pending implementation plan
**Sub-project:** D — re-aims curation at the phenomenon the existing pipeline detects

## Problem

Four independent lines of evidence now say the same thing: **the curation effort
and the automated machinery have drifted apart in what they are about.**

| finding | source |
|---|---|
| Label gate closed at 102 rows; `issue_frame_recall` still 0.00 | sub-project B |
| AGREE and RELATED_UNTYPED symmetrically confused (12 vs 13, near chance) | sub-project B |
| Corpus and cheap pipeline target different phenomena | sub-project C |
| No frozen encoder recovers framing — invariant to capacity *and* objective | encoder sweep `22a16a9` |
| The 22 `same_story=False` contradicts are genuinely cross-story | adjudication `74b252f` |

The cheap pipeline detects **same-story propositional conflict**: two reports of
one event disagreeing on a fact. Curation has grown toward **cross-document
framing divergence** — node4's contested-policy pro/con, which the encoder sweep
confirms no frozen encoder represents at all. The user's decision (2026-07-25) is
to re-aim curation at the phenomenon the machinery already handles well: node4
policy opposition scores **22/22 on every encoder tested**, and the story-gated
contradicts channel is the part of the pipeline that works.

The substrate for that phenomenon is currently almost empty. `news_corpus.yaml`
holds 21 chunks yielding **4** genuine divergence pairs (3 incident + 1
election). Only **20 of 178** labeled store pairs pass `make_same_story` at all.

## Goal

Build a synthetic same-story corpus at scale, surface it to the curator, and
verify it arrives — so the story class can be labeled at the size the readiness
gauge asks for.

This work explicitly does **not** label the pairs. Human labels with rationales
are what the gauge counts and what any downstream detector work trains on. That
is the boundary node4's spec drew and it holds here.

**Not in scope:** fixing `combined.py`'s unconditional
`if same_story: return CONTRADICTS` branch. That is the follow-on this corpus
exists to enable, and it needs labels first. Also out of scope: node1–4, the
escalation bar, `gin/frames/`, and any encoder work.

## Decisions (from brainstorming)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Direction | Re-aim curation at same-story conflict | User decision. The machinery already detects this class well; the alternative (build a cross-story channel) means encoder fine-tuning, which the sweep says is the only remaining frozen-path option. |
| Sourcing | Synthetic controlled divergence | Total control over divergence type and difficulty; cheap; scales; no copyright exposure; and it is the pattern the 4 working pairs already use. Real multi-outlet disagreement is hard to source — most real divergence is corrections over time, not conflicting simultaneous reports. |
| Composition | Divergences **and** same-story negatives | A corpus of pure conflicts would confirm the degenerate branch rather than test it — a detector that always answers CONTRADICTS on same-story would score perfectly. That is exactly the node4 trap, where a purpose-built corpus turned out trivially separable (22/22 everywhere). |
| Structure | One event, several outlets, mixed pairs | Both classes fall out of the same authoring effort, and it mirrors real reporting: outlets agree on most facts and differ on a few. |
| Labels in the manifest | Divergence **intent** recorded as metadata, not as labels | Node4's precedent: `stance` was additive, ignored by `load_corpus_chunks`, present for build/verify and human reading. Encoding the intended answer would defeat the curation this project exists to do. |

## The composition target

The negatives are the point. `gin/cartographer/combined.py` types **any**
same-story pair as CONTRADICTS on story membership alone, with no stance
evidence — on the current corpus that is 11 rows and 11 wrong, the degenerate
branch sub-project C's final review identified. A corpus that can falsify it
needs same-story pairs that are **not** contradictions:

| pair kind | expected relation | why it is hard |
|---|---|---|
| **conflict** — outlets report different values for one fact | `contradicts` / `story` | the positive class |
| **corroboration** — same fact, different wording | `corroborates` | high cosine + same story, but no conflict |
| **update** — a later report revises an earlier figure | `supersedes` | looks identical to a conflict without the timestamp |
| **compatible partial detail** — "23 arrests downtown" vs "31 citywide" | `corroborates` or `related_untyped` | **the sharpest case**: a naive numeric-conflict detector calls this a contradiction and it is not |

## Scale

~12 events × 3–4 outlet reports each. Within an event, pairs are a mix of the
four kinds above, yielding roughly 40–50 labeled-ready pairs — comfortably over
the gauge's 20/class for the story class while also supplying corroborates and
related_untyped.

## Architecture

Two-stage, matching node4's precedent exactly. The authored artifact is
reviewable; the corpus is regenerable from it forever.

```
data/curator/node5_events.yaml        (authored by hand, reviewed before build)
        │
        │  scripts/build_node5.py — pure, deterministic, network-free
        ▼
corpus_node5.json                     (identical schema to node1–4)
        │
        ▼
SameStoryCandidateSource              (new — the residue source cannot serve this)
        │
        ▼
curator UI  →  data/curator/labels.jsonl  →  readiness (story target)
```

### Component 1 — event manifest (`data/curator/node5_events.yaml`)

One entry per event, each carrying its outlet reports:

```yaml
- event: riverport_warehouse_fire
  domain: incident
  shared_lede: "RIVERPORT — Fire crews responded to a warehouse blaze Tuesday evening."
  reports:
    - outlet: CentralWire
      published: "2026-03-04T21:10Z"
      chunks:
        - "Fire crews responded to a warehouse blaze Tuesday evening. Officials confirmed 34 people were evacuated."
    - outlet: MetroDaily
      published: "2026-03-04T21:40Z"
      chunks:
        - "Fire crews responded to a warehouse blaze Tuesday evening. Officials confirmed 19 people were evacuated."
  intent:
    - pair: [CentralWire, MetroDaily]
      kind: conflict
      varied_fact: evacuee_count
```

`intent` drives the build and the surfacing gate. It is **never** written into
`corpus_node5.json` as a relation label and never reaches the curator UI.

**Review gate:** the event list and its intent matrix are approved before the
chunk text is authored, mirroring node4's approved-source-list gate.

### Component 2 — builder (`scripts/build_node5.py`)

Pure, network-free, deterministic. Manifest → `corpus_node5.json`:

- `node_id`: `"node_5_samestory"`.
- Documents ordered by event then outlet: `n5_doc_001`, `n5_doc_002`, …
- `global_id = "gid_" + sha256(f"{source}|{outlet}|{published}").hexdigest()[:16]`,
  matching node4's derivation shape.
- `chunk_id` = `n5_doc_00X_c000…`; `position` stringified index.
- `metadata`: `{outlet, published, event, domain}` — additive, ignored by
  `load_corpus_chunks`, present for verification and human reading.
- Output schema identical to node1–4 so every existing consumer works unchanged.

### Component 3 — candidate source (`gin/curator/same_story.py`)

`EscalationResidueCandidateSource` cannot serve this corpus, for two independent
reasons: it filters **to** the anchor-less residue — `escalation_candidates()`
returns "pairs the cheap path cannot type: **not same-story**" — and it ranks
mid-band cosine first, so same-story pairs (high cosine from a shared lede) would
rank last even if they survived the filter.

`SameStoryCandidateSource` selects pairs where `make_same_story` fires and ranks
NLI-contradiction-descending, so genuine conflicts surface ahead of the
negatives. The same-story predicate is injectable so tests run model-free.

### Component 4 — surfacing gate (`scripts/verify_node5_surfacing.py`)

Hard gate, following `verify_node4_surfacing.py`. Every authored pair in the
`intent` matrix — **conflicts and negatives alike** — must appear in the
curator-reachable backlog.

The negatives matter more than the conflicts here. If only the conflicts
surface, the curator never labels a same-story non-contradiction, and the corpus
cannot break the degenerate branch it was built to break. The gate fails on a
missing negative exactly as loudly as on a missing conflict.

### Component 5 — readiness

Add `story: int = 20` to `ReadinessTarget` and `new_story` to `ReadinessReport`,
counted as `CONTRADICTS` + `relation_class == "story"`, excluding bar pairs and
bar-text aliases exactly as the other classes already are.

## Included correction

`hf_af_staff:0 ↔ hf_af_tenants:0` and `hf_kc_inspection:0 ↔ hf_kc_tenants:0` are
labeled `issue_frame` in the store. That is wrong on two counts: the project's
own labeling guide, corrected against `labels.jsonl` on 2026-07-20 (`b9e0079`),
lists rezoning and habitability as **`story`** examples, and the `gold_edges`
seed labels the same content `story` under its long-form ids
(`hf_alderflats_*`, `hf_kestrel_*`). The `issue_frame` label came from
sub-project B's Task 1 backfill and contradicts both.

Both pairs pass `make_same_story`, so the cheap pipeline already types them
CONTRADICTS correctly — which is what `story` means. Correct them via superseding
records, the same append-only mechanism the original backfill used.

Downstream effects, none of which flip a published verdict:

- Sub-project B's training set drops 102 → 100 rows; DIVERGENT 24 → 22, all node4.
- The encoder sweep's "framing" bucket was **entirely** these two rows. Corrected,
  it is n=0 — which strengthens rather than weakens that sweep's finding that
  B's suggested probe was not well-posed. The sweep's verdict read off the 4
  held-out bar pairs and is unaffected.

## Success criteria

- **Surfacing (hard gate):** 100% of authored pairs, conflicts and negatives
  alike, reach the curator backlog.
- **Composition:** the built corpus yields at least 20 conflict pairs and at
  least 20 same-story negative pairs, verified by the builder against the intent
  matrix.
- **Reproducibility:** `build_node5.py` is deterministic — a rebuild produces a
  byte-identical `corpus_node5.json`.
- **Additivity:** node1–4, the escalation bar, `gin/frames/`, and every existing
  eval surface are untouched. The bar pin test still passes.

Explicitly **not** a success criterion: any detector metric. This spec ships a
corpus and the means to label it. Whether the pipeline can then distinguish
same-story conflict from same-story agreement is the follow-on's question, and
pre-judging it here would repeat the mistake sub-project C made when it assumed
its calibration corpus exercised the mechanism it was calibrating.

## Failure modes

| Condition | Handling |
|-----------|----------|
| An authored pair does not surface | Surfacing gate fails, naming the pair and kind |
| Fewer than 20 of either class | Builder hard-errors against the intent matrix |
| `make_same_story` does not fire on an authored conflict | Gate failure — the shared lede is too thin; strengthen it rather than loosening the predicate |
| Manifest intent references an unknown outlet | Builder hard-errors |
| Rebuild is not byte-identical | Determinism test fails |

## Testing

- **Builder** — determinism, schema match against node1–4, `global_id`
  derivation, hard errors on malformed manifest entries.
- **Corpus regression guard** — asserted pair counts per kind.
- **`SameStoryCandidateSource`** — model-free via an injected same-story
  predicate; asserts negatives are included, not filtered out.
- **Readiness** — `story` counted, bar pairs and text-aliases excluded.
- **Surfacing gate** — every intent pair asserted present.
- Everything model-free except one real-model surfacing smoke run.

## Open questions

None blocking. Deferred by design: whether `supersedes` should be surfaced as a
labelable relation in the curator UI (it currently is not a 4-way training
class), which only matters once the update-kind pairs are being labeled.
