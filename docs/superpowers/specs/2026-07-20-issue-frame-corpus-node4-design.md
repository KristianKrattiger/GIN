# Design: issue_frame Corpus (node4) — Contested-Policy Polar Pairs

**Date:** 2026-07-20
**Status:** Approved (design), pending implementation plan
**Sub-project:** A (curator) → feeds B (issue_frame bi-encoder)

## Problem

The curator UI over the current corpus (node1 climate science, node2 climate
justice/advocacy, node3 monetary policy) yields almost only `corroborates` and
`related_untyped` labels. The `contradicts / issue_frame` class — the entire
point of the curation pass, and the readiness gauge's gating target of 20 new
labels — never surfaces.

Root cause is the corpus, not the pipeline. `issue_frame` is a subclass of
`CONTRADICTS`: two sources taking **opposed positions on the same proposition**
(`app.py:96-101`). The corpus contains topic *silos* and *complementary*
framings, not *opposed* ones:

- node1 × node1 → consensus science, mutually corroborating.
- node1 × node2 → different frame (metrics vs. justice) but complementary, not
  contradictory; reads as `related_untyped` / `corroborates`.
- anything × node3 → different topic entirely → `unrelated`.

The residue candidate source (`EscalationResidueCandidateSource`) faithfully
surfaces the moderate-cosine, not-same-story band and re-ranks it
mid-band-first, but it cannot manufacture opposed pairs that do not exist in the
corpus.

## Goal

Make genuine `issue_frame` pairs **exist** in the corpus and **surface** to the
top of the curator backlog, then let the human curator (kristian) label them.
This work explicitly does **not** label the pairs — human labels with rationales
and anchor-stamps are what the readiness gauge counts and what sub-project B
trains on.

## Decisions (from brainstorming)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Data role | Training/dev data | The escalation bar (14 pairs) is already the held-out eval the detector "will be measured on" (`readiness.py`); the new pairs train, the bar measures. |
| Realism bar | Bar-like difficulty | Because they train and the bar tests, pairs must resemble the bar's moderate-cosine, different-story, genuinely-opposed shape — not trivially-separable op-eds (cf. the 1.6%-at-scale precision failure). |
| Sourcing | Real fetched citations | Matches existing corpus provenance (`global_id` from `source\|author\|date`); most authentic; best generalization. |
| Domain scope | Climate + energy + fiscal | Adjacent domains keep cosine realistic while giving the detector more than one shape of framing conflict. |
| Approach | Topic-anchored pro/con documents | Real provenance + predictable yield + genuine human-in-the-loop labeling. Not claim-manifest pre-pairing (half-defeats the curator); not fetch-broad-and-hope (reproduces the current failure). |
| Topic selection | Proposition-level | Each topic is one claim with an opposite verdict → shared vocabulary (high cosine) + opposite verdict (NLI fires) = the issue_frame fingerprint. |
| Source review gate | Approve source list first | Tightest control over what is cited, before extraction. |
| Surfacing verification | Hard gate | Every topic's thesis pair must PASS surfacing before the corpus is done. |

## Topic set (10, fixed)

Each topic = one proposition with a real source per side (pro / con).

**Climate**
1. Carbon tax — most effective lever / regressive & ineffective
2. Nuclear power — essential to decarbonize / too slow, costly, dangerous
3. Carbon offsets — legitimate mitigation / greenwashing
4. Degrowth vs green growth — growth must shrink / growth can decouple
5. Fossil-fuel divestment — moves capital & pressure / performative, counterproductive
6. Solar geoengineering — prudent to research/deploy / reckless moral hazard

**Energy**
7. 100% renewables grid — reliable without fossil baseload / needs firm gas/nuclear
8. Natural gas as bridge fuel — necessary transition / locks in emissions

**Fiscal / economic**
9. Large-scale climate spending (GND-style) — fiscally sound investment / inflationary & unaffordable
10. Border carbon adjustments — level the field & cut leakage / protectionist & harmful

**Yield:** 10 topics × 2 sources = ~20 docs; ~3–5 chunks/doc; ~2–3 clean opposed
pairs/topic → ~20–30 `issue_frame` candidates, clearing the 20-label target with
margin for curator rejects.

## Architecture

Two-stage, reproducible pipeline. The fragile network step happens once by hand
into a reviewable manifest; the corpus is regenerable from the manifest forever.

```
Stage 1 (by hand, during impl)        Stage 2 (deterministic)         Wiring + gate
search → fetch → read → extract   →   build_node4.py                  launcher default
        │                              (manifest → corpus)            + verify surfacing
        ▼                                    ▼                              ▼
data/curator/node4_sources.yaml  →    corpus_node4.json          curator UI shows pairs
   (APPROVED before extraction)       (real global_ids)          → kristian labels
```

### Component 1 — Source manifest (`data/curator/node4_sources.yaml`)

Human-readable, the reviewable artifact. One entry per topic-stance (20 total):

```yaml
- topic: carbon_tax          # slug, shared by the pro/con pair
  stance: pro                # pro | con
  source: "<real article title>"
  author: "<real author/org>"
  date: "2023-11"            # YYYY-MM or YYYY-MM-DD
  url: "<real url>"
  domain: fiscal_policy      # climate_policy | energy_policy | fiscal_policy
  type: opinion              # opinion | advocacy | analysis
  chunks:
    - "<extracted claim sentence 1>"   # one claim, ~20-30 words, real text
    - "<extracted claim sentence 2>"
```

Copyright: chunks are single attributed claim sentences (matches existing
corpus); no long passages reproduced.

**Review gate:** the source list (title/author/url/stance per topic) is approved
by the user *before* sentence extraction and build.

### Component 2 — Builder (`scripts/build_node4.py`)

Pure, network-free, deterministic. Manifest → `corpus_node4.json`:

- `node_id`: `"node_4_contested"`.
- Documents ordered as topic-pairs, pro then con: `n4_doc_001` = topic 1 pro,
  `n4_doc_002` = topic 1 con, `003`/`004` = topic 2 pro/con, … (pro = odd id,
  con = even id).
- `global_id = "gid_" + sha256(f"{source}|{author}|{date}").hexdigest()[:16]`.
- `chunk_id` = `n4_doc_00X_c000…`; `position` = stringified index (`"0"`).
- `metadata`: `{author, date, domain, type, category, stance, topic}`.
  `stance` and `topic` are additive and ignored by `load_corpus_chunks` (which
  reads only `doc_id`/`position`/`text`); they exist for build/verify + human
  reading.
- Output schema identical to node1–3 so `load_corpus_chunks` and everything
  downstream works unchanged.

### Component 3 — Wiring

Add `corpus_node4.json` to the launcher's default `--corpus` list
(`scripts/curator_serve.py:39-41`). One-line change; the `escalation-residue`
source then includes node4.

### Component 4 — Surfacing verifier (`scripts/verify_node4_surfacing.py`)

The acceptance test for the corpus. Steps:

1. Load node1–4; build the real `EscalationResidueCandidateSource`
   (SentenceTransformer + HF NLI — runs on Windows Python, no llama_cpp/WSL).
2. Use `stance`/`topic` metadata to identify each topic's intended
   **pro-thesis × con-thesis** pair.
3. For each intended pair report: passes residue filter (not-same-story,
   cosine ≥ floor)? rank in surfaced backlog? signals (cosine, `p_contra`,
   `same_story`, `informativeness` tier)?
4. Print a per-topic PASS/SINK table.

**Hard gate:** all 10 thesis pairs must PASS before the corpus is done. A SINK is
fixed at the *source* level — during Stage 1, pick sharper, more directly
contradictory claim sentences (raising cosine + NLI contra) and rebuild. The
residue/candidate ranking logic is never modified; we adapt sources to the
ranker.

## Data flow

1. Research → `node4_sources.yaml` (source list approved, then chunks extracted).
2. `build_node4.py` → `corpus_node4.json`.
3. `verify_node4_surfacing.py` → PASS/SINK table (hard gate; loop back to step 1
   on any SINK).
4. Launcher includes node4 → curator UI surfaces the pairs.
5. kristian labels → readiness gauge `issue_frame` count climbs toward 20.

## Error handling / edge cases

- **Weak-NLI sink:** a genuinely-opposed pair whose NLI `p_contra < 0.5` scores
  `informativeness = 0.0` and sinks. Caught by the verifier; fixed by sharper
  source sentences.
- **Too-high cosine:** near-duplicate vocabulary can push a pair above the
  corroborate ceiling; the NLI-disagreement tier (`informativeness = 2.0`,
  `candidates.py:45`) is designed to rescue exactly high-cosine + contradict
  pairs, which is the issue_frame signature — so this is expected to help, not
  hurt, and the verifier confirms per pair.
- **Malformed manifest:** builder validates required fields
  (topic/stance/source/author/date/url/chunks) and fails loudly.
- **Provenance collision:** if two docs produce the same `global_id`
  (identical source|author|date), the builder errors rather than silently
  dropping (mirrors the loader's dedupe-by-id behavior).

## Testing

All DB-free, no llama_cpp:

- **Builder unit tests:** manifest→corpus mapping; `global_id` asserted against
  the known `gid_f5842fdb72d6327a` example; `chunk_id`/`position` format;
  pro/con adjacency; metadata fields incl. `stance`/`topic`.
- **Loader compatibility:** `load_corpus_chunks(["corpus_node4.json"])` succeeds
  and normalizes ids to `{doc_id}:{position}`.
- **Verifier logic test:** with a **fake proposer**, checks the stance/topic →
  intended-pair mapping without loading HF models. The real-model surfacing run
  is a script-driven gate, not CI.
- **Regression:** existing curator suite stays green.

## Scope boundaries (YAGNI)

Explicitly **not** in this work:

- No pre-labeling of pairs — the curator labels them.
- Not the header-refresh bug (parked, tracked separately as issue B).
- Not the same-issue candidate guard (parked, issue C).
- Not modifying residue/candidate ranking logic — sources adapt to the ranker.
- Not the bi-encoder, retraining loop, or eval harness.

## Success criteria

1. `node4_sources.yaml` — 20 real sources, source list approved before
   extraction.
2. `corpus_node4.json` — 20 docs, valid schema, real computed `global_id`s, real
   url/author/date, `stance`+`topic` metadata, chunks ~3–5 × ~20–30 words.
3. `build_node4.py` — deterministic, unit-tested.
4. `verify_node4_surfacing.py` — all 10 thesis pairs PASS (hard gate).
5. Launcher default `--corpus` includes node4.
6. Downstream: kristian can reach ≥20 `issue_frame` labels, flipping the
   readiness gauge (outcome, not a step this work performs).
