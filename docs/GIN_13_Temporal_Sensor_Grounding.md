---
tags: [GIN, research, architecture, temporal, sensors, grounding]
updated: 2026-07-11
version: 0.4-preliminary
status: working draft (scoping — many phases out, not yet sequenced)
register: conceptual
---

# GIN 13 — Temporal & Sensor Grounding

> Extending SEAR's grounding guarantee from editorial text to **temporal sensor data**: ground truth at a specific resolution that no node can fabricate and no frame can wish away. This is the strongest available stress test of the `fabrication_rate` guarantee ([[GIN_ENG_02_Eval_Baseline_v1]]) — because a sensor value can be checked against the world, not only against corpus-internal consistency — and the case in which productive divergence ([[GIN_02_Productive_Divergence]]) becomes structurally honest rather than a coordination failure to be smoothed over.

---

## Why sensor data belongs in GIN

The divergence thesis so far runs on editorial corpora — institutional statistic vs. grassroots reframing, one outlet against another. A sensor introduces a different *kind* of node. It is not another interpretation of a contested fact; it is an **instrument**. When a buoy, a strain gauge, a plant-physiology probe, or a blood-glucose stream is federated alongside models and narratives operating at other resolutions or epistemic frames, the disagreement between them is not noise to be averaged away — it is signal.

The pattern generalises across domains that share GIN's founding structural problem: **multiple sources of ground truth, at different scales and resolutions, that should not be forced into consensus.** The sensor does not care about anyone's frame. It measures what it measures, and in doing so it forces every other node to either explain the disagreement or concede a gap in its model. That is exactly the pressure GIN exists to keep legible rather than dissolve.

Origin note: this direction was sparked by in-field plant-physiology sensors (e.g. the ECAL/Vivent "Vita" device) — instruments producing plot-level physiological ground truth that no regional climate model or generational tacit knowledge reduces to.

---

## Sensor data is a stricter grounding substrate, not just a harder one

Text is generative by nature: a model can always produce a fluent, plausible-sounding fabricated claim. A timestamped sensor value cannot be fabricated the same way — it either exists or it does not. An interpolated or invented reading is a **more structurally detectable** failure mode than an invented sentence.

This strengthens SEAR's core guarantee rather than merely complicating it. The current editorial evaluation measures `fabrication_rate` against corpus-internal consistency. Sensor data tests the same guarantee against something *outside* the corpus — a falsifiable physical substrate that is harder to integrate but harder to fake. Medicine and infrastructure, where false consensus carries direct patient-safety and structural-integrity cost, are the strongest candidates for that stress test.

---

## The structural difference: context-grounding *before* divergence

With editorial corpora an edge is **semantic** — two claims relate because they are about the same entity, event, or contested fact. A raw reading has no semantic content on its own. `soil_moisture: 14.2%` at a given timestamp is meaningless without, at minimum:

- the sensor's calibration state,
- the crop / growth-stage — or patient, structure, shipment — it is attached to,
- what "normal" is for that variable, at that time, in that context,
- what other variables were doing simultaneously.

So the edge is not node-to-node semantic similarity first. It is **node-to-context grounding first**, and only then node-to-node divergence comparison. A reference/baseline layer is needed *underneath* Relation-Finder, to which raw readings are anchored before they are eligible for comparison at all. This is the load-bearing architectural consequence of taking sensor data seriously: a new layer, not a new corpus.

---

## Domain applications

Each pairing below produces meaningful divergence signal precisely because the nodes occupy genuinely different epistemic vantage points.

| Domain | Nodes | Why divergence is signal |
|--------|-------|--------------------------|
| **Medical / diagnostic** | biometric streams (HR, glucose, EEG) + clinical notes + labs + protocols + patient-reported symptoms | Three vantage points — sensor sees physiology, patient experiences symptom, clinician sees population pattern — that need not converge; when they diverge it is often clinically meaningful. Highest stakes. |
| **Infrastructure / structural health** | accelerometers, strain gauges, moisture on bridges/buildings + maintenance logs + load models + inspection reports | A monitoring system disagreeing with a theoretical load model is often the *earliest* signal of a real problem — actionable before it is visible. |
| **Supply chain / logistics** | GPS/temp/humidity on shipments + customs data + supplier scores + demand forecasts | Divergence between what the sensor says happened and what the paperwork says happened *is* the investigative signal — fraud, spoilage, delay, misreporting. |
| **Climate / oceanography** | buoys, weather stations, satellite retrievals + IPCC/regional models + Indigenous/local ecological knowledge | Adjacent to the current institutional-vs-grassroots pair, but adds a live sensor layer: not different *interpretations* of climate, different *instruments* measuring it. |
| **Cultural / music production** | streaming behaviour (plays, skips, duration by demographic) + artist/label reporting + criticism | Interpretive divergence grounded in a measurable signal — what people actually listen to, moment to moment, vs. the narrative of what is said to matter. |

Ag-environmental federation — live plant-physiology sensors + regional/institutional climate data (NOAA, USDA, extension offices) + local/tacit farmer and Indigenous knowledge — is the founding concrete case: three epistemic vantage points at three physical scales (plot-level, regional-model, generational-tacit) that do not reduce to one another.

---

## Calibration and trust: metadata as a declared claim

Sensor data is treated as ground truth, but sensors can be wrong — calibration drift, environmental noise, systematic bias, deferred maintenance. Without accounting for this, SEAR could flag a broken instrument's readings as legitimate divergence when the disagreement is really just noise from a bad sensor.

This is a **Bookkeeper problem** — Bookkeeper already owns calibration and audit, and this extends that role into the sensor domain rather than requiring a new subsystem. The design principle is GIN's throughout: **required metadata, not inferred trust.** Institutions publish sensor metadata (calibration/maintenance date, expected drift parameters, model and known error margins, historical validation records) as a *condition of participation*. That metadata becomes part of the node's epistemic signature in the knowledge graph — searchable, auditable, comparable — and Bookkeeper consults it before comparing readings across nodes. A sensor calibrated six months ago with 2%/month drift carries roughly ±12% built-in uncertainty; that uncertainty becomes explicit context for how much weight the signal carries, not a disqualifier.

This turns "is this sensor trustworthy?" from an inference problem into a **declared, checkable claim** — and false metadata is itself an auditable claim, subject to the same governance enforcement described in [[GIN_10_Epistemic_Council]]. The engineering schema for the registry is an issue below, not a spec.

---

## The honest limit, stated up front

**A poisoned or miscalibrated sensor, faithfully reproduced, produces a grounded distortion.** This is the [[GIN_04_SEAR]] corpus-integrity limit in a new substrate: every technical guarantee floats on the integrity of what the node holds — here, sensor integrity — which the architecture can *govern* (via required metadata) but cannot technically enforce. Metadata publication mitigates the failure and makes it discoverable; it does not eliminate it. That boundary is not a weakness to hide; it is what defines the kind of system GIN is.

---

## Engineering issues (not specs)

Status: **many phases out** from the current Cartographer/Bookkeeper sequencing ([Real-text divergence generalization](nc_real_text_divergence_generalization.plan.md) §7); structural scoping, not an implementation plan. All items below are unmeasured — see [[GIN_ENG_00_Engineering_Register]].

- **Architectural fork.** *Derived-claim conversion* (preprocess raw series into text-like claims before Relation-Finder — low effort, reuses the text pipeline, but discards the raw ground-truth quality and reintroduces the fabrication surface) vs. *native temporal nodes* (time-series objects with their own edge semantics — correlation, lag, anomaly co-occurrence — as a parallel reasoning pathway). **Current lean: native temporal nodes** — honest to what sensor data is, at the cost of a real second reasoning pathway.
- **Baseline/reference layer.** Per-crop, per-region, per-sensor-model, or domain-specific — undetermined.
- **Calibration metadata registry.** Schema, and whether it is council-hosted shared infrastructure or self-published-and-council-audited (open; leaning shared).
- **Divergence scoring mechanics** for time-series vs. time-series and time-series vs. text claim — likely different comparison machinery than text vs. text.
- **First node pair / dataset** — not yet identified.

Non-goals for now: real-time ingestion architecture; selecting an actual first data-source pair.

## Related

[[GIN_00_Reader]] · [[GIN_02_Productive_Divergence]] · [[GIN_04_SEAR]] · [[GIN_07_Governance_Validity]] · [[GIN_10_Epistemic_Council]] · [[GIN_ENG_00_Engineering_Register]] · [[GIN_STRAT_00_Strategy_Register]]

## Back to Vault

[[HOME]]
