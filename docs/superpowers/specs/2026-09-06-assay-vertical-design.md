# Design — `GIN_14_Assay`: the portable constraint vertical

*Status: approved design | 2026-09-06 | conceptual register*

## Purpose

Add a third vertical to the GIN series. GIN and Receipts already exist as two
instances of one thesis — **grounded confidence and loud refusal**. GIN is that
thesis scaled: federation, an Epistemic Council, SEAR constrained decoding, tiered
corpora. Receipts is that thesis shipped thin: live cloud browsers fan across a
vendor's claims and independent web writing, an LLM proposes, a deterministic gate
re-derives every quote from fetched bytes and discards what it cannot find.

The **Assay** is the same thesis with everything situational removed: hand it a set
of documents and a query, and it returns a cited answer, a divergence report, or a
refusal — and never a fluent claim it cannot anchor. GIN is the assay office at
scale; Receipts is a field assay of the live web; the Assay is the portable kit
both are built from.

The deliverable of this design is **one conceptual-register document**,
`docs/GIN_14_Assay.md`, plus index and backlink wiring. No code. Any request/response
shape in the document is illustrative, not normative; it crosses to a real
specification only when something is built.

## The name

`Assay`. To assay is to test a sample against a question — "how much gold is in
this?" — and return a measured result *or reject the sample as unassayable*. That
is exactly `documents + query → grounded reading | refusal`. It is portable by
nature (an assay kit), agnostic to the material, and it fits GIN's apophatic
discipline: the assay that comes back empty is a real result, not a failure. It
reads well in the series next to SEAR and MOCAP, and it avoids leaning on "API"
for a document that is conceptual, not a spec.

Rejected alternatives: `Agnostic Grounded API / Layer` (weak lead word — agnostic
to *what* is not visible in the name; "API/Layer" generic; acronym collisions);
`Touchstone` (metaphor is exact but the word is worn smooth by business use);
`Grounded Constraint Kernel` (systems-register; pairs oddly with Receipts'
plain-noun style).

## The spine (shared thesis)

**Grounded confidence + loud refusal.** Every emitted claim traces to a span that
provably exists in the supplied corpus. Where nothing grounds an answer, the system
marks the silence instead of filling it. Everything else in all three verticals is
downstream of this.

## Decisions locked during brainstorming

1. **Deliverable form.** A GIN-series concept/positioning document. No code, no
   software spec in this cycle.

2. **Register.** Conceptual. The API shape appears as an *illustrative sketch* to
   make the boundary concrete (as GIN_04 shows the friction dial without specifying
   bytes), explicitly marked non-normative.

3. **Document scope.** One document, `GIN_14_Assay`, that does both jobs: opens with
   the three-vertical map (the "tie it together" work) and then goes deep on the
   Assay itself so the map does not undersell it.

4. **Three outcomes, not two.** `corpus + query → {grounded answer | divergence
   report | refusal}`. Averaging two contradictory grounded spans into one
   confident answer is exactly the confabulation the thesis exists to prevent, so a
   conflict is surfaced with both sides cited rather than collapsed or refused.
   This is what makes the Assay genuinely the core Receipts could be rebuilt on.

5. **Graded refusal.** Every response carries a confidence. "Refusal" is the band
   below a **caller-set threshold**, not a separate code path. Near-miss spans stay
   attached to a refusal — a refusal that shows the three spans it rejected and why
   is more useful than a bare "no", and consistent with Receipts publishing its
   denial counts as the thing that makes the guarantee checkable.

6. **Refusal reason codes** (first-draft set for the doc):
   - `NO_GROUNDING` — no span in the corpus addresses the query
   - `BELOW_THRESHOLD` — spans exist but none clears the confidence bar
   - `CONFLICTING_UNRESOLVABLE` — grounded spans contradict and the caller opted
     into convergent-only (see decision 4; default is the divergence report)
   - `QUERY_UNGROUNDABLE` — the question is not the kind a corpus can answer
     (opinion, prediction, counterfactual with no basis in the documents)
   - `CORPUS_INSUFFICIENT` — corpus empty, or too thin to answer *or* refuse
     honestly (the "looks like a result but isn't" guard, ported from Receipts'
     single-role refusal)

7. **Headline agnosticisms: model, source, task.**
   - *Model* — post-hoc verification needs no control over decoding, so the Assay
     runs on any LLM. This is what separates it from SEAR.
   - *Source* — the Assay never fetches; the caller supplies bytes. This removes
     Receipts' cloud-browser dependency.
   - *Task* — Q&A, claim-check, extraction, and cited summary are the same
     operation: query + corpus → grounded spans or refusal.
   - *Domain* — stated as inherited from the spine (GIN's "no per-vendor code",
     Receipts' "JSON not code"), not a fourth marquee axis.

8. **Argument structure.** Subtractive framing for the map section (GIN and Receipts
   as the spine plus situational layers; the Assay as the invariant residue),
   mechanism-paper treatment for the deep section, comparative-case material folded
   into a single "who is this for" section.

## Document outline (`GIN_14_Assay.md`)

Frontmatter: `tags: [GIN, research, assay, grounding, refusal, portable]`,
`updated: 2026-09-06`, `version: 0.4-preliminary`, `status: working draft`,
`register: conceptual`.

Thesis blockquote: the "assay office at scale / field assay / portable kit"
sentence from Purpose above.

**§1 — The three verticals (subtractive map).** Start from the spine. Remove
things. GIN = spine + federation + Epistemic Council + SEAR + corpus tiers.
Receipts = spine + live web + cloud browsers + per-run fetching. The Assay = the
invariant residue when both situational layers are stripped. The point: all three
are one idea seen at three scales, not three tools.

**§2 — The operation.** The three outcomes. Graded confidence on every response;
refusal as the sub-threshold band; caller-set threshold; near-miss spans attached.
An audit line (proposed / admitted / denied with reasons) echoing Receipts — the
denial count is what makes the guarantee checkable.

**§3 — The three agnosticisms.** Model, source, task as in decision 7, each with the
one sentence that says what it buys against the other two verticals. Domain-
agnosticism named as inherited, not argued.

**§4 — The loud refusal.** The graded mechanism from §2 plus the reason codes from
decision 6.

**§5 — Divergence as a first-class outcome.** Why it is outcome #3 and not a
refusal (decision 4). Default: report both sides with citations. Convergent-only is
a caller option that turns unresolved conflict into `CONFLICTING_UNRESOLVABLE`.

**§6 — What the Assay is not.** No fetching, no federation, no Council, no
promotion gates, no persistence, no divergence *governance* — it detects conflict;
it does not rule on which side is legitimate (that is GIN_07's work). Governance is
deliberately out of scope, and that exclusion is what "portable" means.

**§7 — Who it is for.** Third parties who want grounding but will never run a
federation or a cloud browser; Receipts rebuildable on it; an agentic tool that
needs cite-or-refuse without buying the whole GIN stack.

**§8 — Relationship to SEAR.** SEAR and post-hoc exact-substring verification are
two implementations of one contract: grounding by construction. The Assay ships the
post-hoc one (model-agnostic, enforced after generation); GIN uses SEAR (needs open
weights, enforced during decoding). Same guarantee, different enforcement point —
so the Assay is not "SEAR-lite", it is the contract SEAR also satisfies.

**§9 — Open seams.** Honest section matching the other docs: what counts as "a
span" when an answer is synthesised rather than extracted; confidence calibration
across models; whether divergence *detection* needs a model call (and if so, that
step is not model-agnostic); the `CORPUS_INSUFFICIENT` threshold being a non-neutral
judgment.

Related line: `[[GIN_00_Reader]] · [[GIN_02_Productive_Divergence]] ·
[[GIN_04_SEAR]] · [[GIN_07_Governance_Validity]] · [[GIN_09_Agentic_Layer]] ·
[[GIN_STRAT_00_Strategy_Register]]`, then `## Back to Vault` → `[[HOME]]`.

## Ripple edits

- **`docs/GIN_00_Reader.md`** — add a row to the document-set table:
  `| [[GIN_14_Assay]] | The portable grounding core; corpus+query → answer /
  divergence / refusal; model / source / task agnostic | conceptual |`. Add one
  sentence to "How to read this set" placing it after GIN_04 as the portable
  distillation of the same constraint.
- **`docs/GIN_04_SEAR.md`** — add `[[GIN_14_Assay]]` to the Related line.
- **`docs/GIN_07_Governance_Validity.md`** — add `[[GIN_14_Assay]]` to the Related
  line.
- **Receipts `README.md`** — one line in the Lineage section: Receipts as a field
  instance of the Assay contract, cross-referencing `GIN_14_Assay`.

## Out of scope

- Any software spec, package layout, or API type definitions for a buildable Assay.
- Changes to the engineering or strategy registers.
- A new register.
- Ruling on divergence legitimacy (belongs to GIN_07).

## Self-review

- **Placeholders:** none. All five refusal codes, all three agnosticisms, and all
  nine sections are specified.
- **Internal consistency:** decision 4 (always report divergence) and the
  `CONFLICTING_UNRESOLVABLE` code (decision 6) are reconciled explicitly — the code
  fires only when a caller opts into convergent-only; the default is the report.
- **Scope:** one document plus four small wiring edits. Single implementation plan.
- **Ambiguity:** "illustrative, not normative" is stated three times (register
  decision, purpose, §2 framing) so the API sketch cannot be mistaken for a spec.
