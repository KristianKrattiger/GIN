# Task 4 review-fix report

## Findings addressed
- **A (Important, measured):** NLI ran on the full uncapped residue with no cache (~420s),
  hanging /curator/next on first load. Fixed: `nli_rank_limit` (default 400) bounds NLI to the
  highest-cosine residue pairs; everything past it is cosine-ranked and never pays a
  cross-encoder call. Added a real `_p_contra_cache` (unordered-pair keyed) to
  CombinedRelationProposer. Corrected the false "(cached)" comment.
- **B (Important):** ranking used `candidates.CONTRA_THRESHOLD` (0.5) instead of the calibrated
  system threshold. Now uses `proposer.thresholds.contra_threshold` (0.686 from
  data/cartographer_thresholds.json). Unused import dropped.
- **C (document only):** residue docstring now records the tension that residue pairs are
  by construction not-same-story — the population `classify_relation` story-gates NLI off for —
  and why it is acceptable here (ranking hint for a human adjudicator, not automated typing),
  plus the consequence that signals.py reports nli_p_contra: None for those same pairs.

## Verification
- `pytest tests/test_curator_residue.py tests/test_curator_node4_verify.py tests/test_curator_candidates.py tests/test_curator_node4_build.py tests/test_curator_node4_corpus.py -q`
  -> **22 passed in 0.17s** (sub-second proves no real model pulled into unit tests)
- `pytest tests/test_cartographer_combined.py -q` -> **7 passed in 0.07s**
- `python scripts/verify_node4_surfacing.py` -> **10/10 thesis pairs surfaced / HARD GATE PASSED**
- Gate wall-clock: **~420s before -> 220s after**

## New tests
- `test_pairs_bounds_nli_to_rank_limit` — injected scorer RAISES if NLI is consulted beyond the
  limit; asserts pure cosine ordering past it.
- `test_nli_p_contra_is_memoized_on_unordered_pair` — asserts no re-invocation, incl. swapped args.

## Tradeoff introduced (honest note)
Bounding NLI by top-400 cosine means contradictions whose cosine falls outside that window lose
their float. Observed rank shifts: nuclear 1056->156 and carbon_offsets 1505->147 (better — high
cosine), but degrowth 23->1132 and renewables 714->1865 (worse — strong NLI contradiction below
the cosine window is never consulted). Gate is presence-based so all 10 still PASS. Net effect on
curator triage is favourable: 5 thesis pairs now sit within the top 156, including the framing-type
hard cases. Raising `nli_rank_limit` trades runtime for restoring the deep-contradiction float.
