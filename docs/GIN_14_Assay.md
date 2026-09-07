---
tags: [GIN, research, assay, grounding, refusal, portable]
updated: 2026-09-06
version: 0.4-preliminary
status: working draft
register: conceptual
---

# GIN 14 — Assay

> The Assay is grounded confidence and loud refusal with everything situational
> removed. Hand it a set of documents and a query; it returns a cited answer, a
> divergence report, or a refusal — and never a fluent claim it cannot anchor.
> GIN is the assay office at scale; Receipts is a field assay of the live web;
> the Assay is the portable kit both are built from.

---

## 1 — Three verticals, one constraint

The GIN series has, until now, described one system. This document adds a second
and a third vertical that are the same system at a different scale, and names what
all three share.

The shared claim is small: **every emitted claim traces to a span that provably
exists in the supplied corpus, and where nothing grounds an answer the system
marks the silence instead of filling it.** Grounded confidence, and loud refusal.
Everything else in the series is downstream of that sentence.

GIN is that claim scaled. Around the constraint it wraps federation
([[GIN_03_Node_Identity]]), a governing institution ([[GIN_10_Epistemic_Council]]),
constrained decoding that enforces grounding at generation time ([[GIN_04_SEAR]]),
and a tiered corpus from cold content-addressed blobs upward. The wrapping is most
of the engineering and all of the politics.

Receipts is that claim shipped thin (a working tool; hosted ledgers at
kristiankrattiger.github.io/receipts). It points a fan of cloud browsers at one
vendor's marketing and at independent writing about it, lets a model propose which
claims corroborate or contradict which, and then runs a deterministic gate that
re-derives every quoted span from the bytes that were actually fetched and
discards anything it cannot find. No federation, no council, no local model — a
live-web claim ledger and nothing more.

Strip the situational layers from either one — the federation and governance from
GIN, the browsers and the live web from Receipts — and the same object is left in
both hands: a function from a corpus and a query to a grounded answer, a surfaced
disagreement, or a refusal. That object is the **Assay**. GIN is the assay office;
Receipts is a field assay; this document is about the kit.

## 2 — The operation

An assay takes a sample and a question and returns a reading or a rejection. The
Assay takes a set of documents and a query and returns one of three things:

- a **grounded answer** — a claim with every load-bearing span in it citing a
  location in the supplied corpus;
- a **divergence report** — two or more grounded claims that the corpus supports
  and that contradict each other, presented together with their citations rather
  than reconciled;
- a **refusal** — a statement that the corpus does not support an answer, carrying
  a reason and the spans that came closest.

Every response carries a **confidence**. Refusal is not a fourth outcome bolted
on; it is the band below a threshold the caller sets — the reason codes of §4 say
*why* a response landed there, and one of them (`CONFLICTING_UNRESOLVABLE`) is a
caller-mode outcome rather than a low score. A caller who wants only near-certain
answers sets the bar high and receives more refusals; a caller triaging leads
sets it low. The spans that were rejected for falling under the bar stay attached
to the refusal, because a refusal that shows its work — here are the three
passages we considered and the score each earned — is worth more than a bare no,
and it is what lets the guarantee be checked rather than trusted.

The response also carries an **audit line**: how many candidate claims were
proposed, how many admitted, how many denied and under which reason. Receipts
publishes exactly this line on every ledger, and it is the single thing that
converts "we verified our quotes" from an assurance into something a reader can
recount.

Sketched concretely — to fix the boundary, not to specify bytes:

- **in** — a set of documents (content and an identifier each), a query, a
  confidence threshold, and whether an unresolved conflict should be reported or
  refused.
- **out, grounded answer** — the claim, and for each load-bearing phrase the
  document and the character offsets a reader can slice to check it; plus the
  confidence and the audit line.
- **out, divergence report** — the same shape, but two or more claims, each with
  its citations, marked as in conflict.
- **out, refusal** — a reason code from §4, the confidence, and the near-miss
  spans with the score each earned.

That sketch and the reason vocabulary of §4 are *illustrative, not normative*.
This document is in the conceptual register; it draws the boundary, it does not
specify a wire format. A normative specification waits for an implementation.

## 3 — Agnostic to model, source, and task

The name's adjective was "agnostic" before it was "Assay", and three axes are what
it meant.

**Model.** The grounding check is post-hoc: generate a claim by whatever means,
then verify each span against the corpus by exact match. Nothing in that loop
needs control over decoding, so the Assay runs on a frontier API model, a local
open-weights model, or a model not yet released. This is the clean line between the
Assay and [[GIN_04_SEAR]], which enforces the same guarantee *during* generation
and therefore needs a model it can open up.

**Source.** The Assay never fetches. The caller hands it bytes — from a web crawl,
a database, an upload, a git repository, a disclosure packet — and the Assay treats
all of them the same way. Receipts owns cloud browsers because reading hostile
pages is its whole problem; the Assay has no opinion about where a document was
before it arrived.

**Task.** Question answering, claim checking, extraction, and cited summarisation
are one operation seen from different sides: a query and a corpus resolve to
grounded spans or to a refusal. The Assay does not have four modes for these; it
has one.

Domain-agnosticism is not a fourth axis because it is already a house rule.
Receipts adds a domain with a JSON file and no engine change; GIN's nodes are
architecturally uniform and differ in corpus, not in code
([[GIN_03_Node_Identity]]). The Assay inherits that, and would be remarked on only
if it broke it.

## 4 — The loud refusal

Half the thesis is the refusal, so it has a vocabulary. A refusal carries one of:

- `NO_GROUNDING` — nothing in the corpus speaks to the query at all;
- `BELOW_THRESHOLD` — the corpus speaks to it, but no candidate span clears the
  caller's confidence bar;
- `QUERY_UNGROUNDABLE` — the question is not the kind a corpus can answer: a
  prediction, an opinion, a counterfactual with no basis in the documents;
- `CORPUS_INSUFFICIENT` — the corpus is empty, or so thin that neither an answer
  nor an honest refusal can be built on it. This is the guard against a result
  that looks like a finding and is not; Receipts refuses a run before it spends
  money when a plan names only one side, for the same reason;
- `CONFLICTING_UNRESOLVABLE` — the grounded spans contradict each other and the
  caller asked for a single answer. By default this is not a refusal at all (see
  §5); it becomes one only when the caller has opted out of divergence reports.

The reason codes are a draft. What is not a draft is that a refusal names its
reason: an unexplained refusal is as opaque as an ungrounded answer, and opacity
on either side is the failure the Assay exists to prevent.

## 5 — Divergence is an outcome, not a failure

When two passages in the corpus are each well grounded and say opposite things,
three moves are available: pick one, average them, or refuse. All three destroy
information. Picking one hides a real citation; averaging invents a claim that no
passage makes; refusing throws away two findings that are individually sound.

The Assay takes the fourth option and returns both, cited, marked as in conflict.
This is [[GIN_02_Productive_Divergence|productive divergence]] at the smallest
possible scale — not a federation of nodes disagreeing across a network, just two
spans in one bundle of documents — and it is the reason the Assay is a faithful
core for Receipts rather than a cousin of it. Receipts' entire divergent section
is this outcome.

A caller who genuinely needs one answer can ask for convergent-only behaviour and
receive `CONFLICTING_UNRESOLVABLE` instead. That is a legitimate request — some
consumers cannot act on a pair — but it is the exception, and the default is to
surface the disagreement.

What the Assay does not do is judge the disagreement. Whether a contradiction is a
legitimate difference of situated perspective or a sign that one source is simply
wrong is the validity question, and it belongs to [[GIN_07_Governance_Validity]]
and the institution that holds it. The Assay reports that the corpus disagrees
with itself; it does not rule on the merits.

## 6 — What the Assay is not

The Assay does not fetch, crawl, or render. It does not federate, sync, or hold a
network identity. It has no council, no promotion gate, no quarantine register, no
notion of a claim graduating from provisional to established. It does not persist
anything between calls. It does not govern divergence.

Every one of those is deliberate, and together they are the definition. GIN is the
Assay plus the machinery that makes grounded reasoning survivable at global scale
and accountable to someone. Receipts is the Assay plus the machinery for reading
sources that fight back. The Assay is what is common to both once the machinery is
removed — and something that portable is something a third party can pick up
without adopting a federation or a browser fleet.

One consequence is worth stating plainly: the Assay is faithful to the corpus it
is handed and has no opinion about how that corpus was assembled. GIN's honest
limit — structural fidelity reproduces the capture, it does not vouch for it
([[GIN_04_SEAR]], [[GIN_07_Governance_Validity]]) — applies here in full, with
none of the governance that answers it. Portability and that exposure are the
same property.

## 7 — Who it is for

Three readers.

An **agent developer** who wants their tool to cite or decline, and has no
interest in running nodes. Today the options are a full retrieval stack they must
trust or a constrained-decoding setup they must host. The Assay is the guarantee
without either.

**Receipts itself**, rebuilt. The propose–admit–render pipeline in Receipts is the
Assay with a browser fan bolted to its front and a claim-ledger renderer bolted to
its back. Factoring the middle out and depending on it would shrink Receipts to
the parts that are actually about the live web.

A **GIN node**, for the paths where constrained decoding is too heavy or the base
model is not one SEAR can open. The Assay and SEAR enforce the same contract; a
node can hold both and choose per query.

## 8 — SEAR and the Assay are one contract

[[GIN_04_SEAR]] constrains a model so that it can only emit spans that occur in the
corpus; grounding is true by the time a token exists. The Assay lets the model
emit anything and then deletes what does not verify; grounding is true by the time
a response is returned. The enforcement point is different — during decoding
versus after it — and so is the robustness: the engineering register grades
decode-time enforcement hard and post-hoc enforcement soft
([[GIN_ENG_01_SEAR_PoC_Spec]]). What the reader may rely on is the same in both:
no claim without a span.

So the Assay is not a lightweight SEAR. It is the contract stated independently of
how it is kept, with SEAR as the enforcement that needs an open model and post-hoc
verification as the enforcement that does not. Naming the contract on its own is
what lets the two implementations be compared instead of conflated.

## 9 — Open seams

Held open, in the series' habit:

- **What is a span, when the answer is synthesised.** Exact-substring
  verification is clean for extractive answers and unclear for a sentence that
  fairly paraphrases three passages. Receipts sidesteps this by only ever quoting;
  a general Assay has to decide whether paraphrase is admissible and, if so, how it
  is checked.
- **Confidence across models.** A caller-set threshold assumes the confidence
  number means the same thing from one model to the next. It does not, without
  calibration, and calibration is work the conceptual register cannot wave away.
- **Divergence detection may need a model.** Deciding that two spans contradict is
  itself an inference. If it takes a model call, the detection step is not
  model-agnostic even though the verification step is, and the doc should not claim
  otherwise.
- **`CORPUS_INSUFFICIENT` is a judgement.** The line between "thin but answerable"
  and "too thin to be honest" is not a measurement. Receipts draws it with a
  single-role guard; a general Assay needs a defensible rule, and any rule it picks
  is a small act of governance in a thing that claims to have none.

## Related

[[GIN_00_Reader]] · [[GIN_02_Productive_Divergence]] · [[GIN_04_SEAR]] · [[GIN_07_Governance_Validity]] · [[GIN_09_Agentic_Layer]] · [[GIN_STRAT_00_Strategy_Register]]

## Back to Vault

[[HOME]]
