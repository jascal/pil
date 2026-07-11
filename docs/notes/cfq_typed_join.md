# CFQ typed-schema join assembler (modality-gap probe 2 — final CFQ probe)

**Status:** measured, decisive negative under a pre-registered rule (2026-07-11). The
second and **last** probe into the modality gap on CFQ: after
[the join headroom diagnostic](cfq_join_headroom.md) ruled out majority-vote (A=0.33 IID)
and lexical (B≈0.02) join predictors, this slice measured the remaining simple candidate —
a deterministic **type-driven assembler** over mined `(predicate, slot) → required arg-type`
tables. `pil/cfq_edges.py` (`parse_sparql_typed_slots`) +
`experiments/campaign_cfq_typed_join.py`. Measurement only, oracle-predicate regime, same
harness/splits/leak-guards as probe 1.

## Pre-registered decision rule

Recorded **before any assembler numbers were observed** (implementation was in flight,
no scores reported): the typed assembler is a live path **iff mean `join_f1` C ≥ 0.7 on
the IID `random` split** (schema-determinism territory; probe 1's A=0.33 is the floor).
Either way, the CFQ probe chain stops here — no probe 3.

## Method

- **Mining (fit only):** per `(pred, slot)`: `var_rate` (fraction of occurrences holding a
  variable; ≥0.5 ⇒ var slot at assembly), `req_type` (majority *explicit* `a`-type over
  typed var occurrences; never-typed slots get non-merging sentinels), `label_maj`
  (majority surface label, usually `VAR`, for the signature's first element — decoupled
  from the merge key because **64–68% of gold join vars carry no `a`-type**, so surface
  labels cannot drive merging).
- **Assembler (input = oracle predicate multiset only):** maximal merge — group var-slot
  occurrences by exact `req_type` equality (no type-compatibility classes in v1,
  deliberately, so the ceiling diagnostic stays readable); groups with ≥2 incidences emit
  signatures in the probe-1 representation. Known limitation kept visible (unit-tested,
  not patched): a cycle of same-typed vars over-merges into one star.
- **Decomposition:** C (full pipeline) vs A (probe-1 majority baseline, identical test
  sample); `C_inc` = **quarantined diagnostic** ceiling (gold slot occupancy, var identity
  dropped, same merge rule) — isolates the merge family from occupancy prediction; it
  informs no positive verdict. Unseen-key A-vs-C comparison restricted to queries with ≥1
  gold join var (`join_f1` scores empty-vs-empty as 1.0, so unrestricted unseen-key means
  are inflated for A).

## Result — oracle-predicate regime, 3000 test queries/split

| split | A | A_seen | **C** | C_seen | C_unsJF | A_unsJF | merge-ambig | unseen key | C_inc (diag.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mcd1 | 0.191 | 0.257 | 0.168 | 0.159 | 0.196 | 0.000 | 0.020 | 0.540 | 0.485 |
| mcd2 | 0.090 | 0.114 | 0.149 | 0.133 | 0.181 | 0.000 | 0.014 | 0.417 | 0.402 |
| mcd3 | 0.152 | 0.178 | 0.228 | 0.138 | 0.428 | 0.000 | 0.011 | 0.359 | 0.503 |
| **random (IID)** | 0.332 | 0.455 | **0.215** | 0.196 | 0.266 | 0.000 | 0.137 | 0.276 | 0.477 |

## Finding (decisive; pre-registered rule applied)

**C = 0.215 on `random` — far below the 0.7 live-path threshold, and below the majority
baseline A = 0.332.** The mined-type assembler is not a live path.

- The assembler *does* generalize compositionally where A cannot (C_unsJF 0.18–0.43 vs
  A_unsJF = 0.00 on unseen predicate multisets — it beats A outright on mcd2/mcd3), but its
  absolute level is nowhere near assembly-grade.
- **The failure is the merge family, not the mining:** with *oracle* slot occupancy,
  `C_inc` is only 0.40–0.50. Since same-type collisions within a query are rare
  (merge-ambig 0.01–0.14), over-merge explains only part of the deficit; the rest is
  consistent with **fragmentation** — a single gold join var whose slots mine to
  *different* required types is split across groups (not directly decomposed here;
  interpretation, not a measured claim). Type-compatibility classes (union-find) would
  trade fragmentation for worse over-merge — a tension intrinsic to exact-type merging,
  which is why v1 stopped here per the pre-registration.

## The record

**CFQ join structure is beyond the current induction paradigm at this feature
granularity.** Three predictor families are now measured out: exact-multiset majority
vote, lexical word-votes (probe 1), and mined-type maximal merge incl. its
oracle-occupancy ceiling (this probe). This is a **recipe plateau** across those families —
**achievability stays OPEN** (a richer join model — real Freebase ontology, finer typed
features, or an incidence-structured proposer — remains untested on CFQ and is *not*
claimed impossible). The CFQ probe chain is closed; the falsifiable successor is a
**synthetic join battery** with planted ground-truth join topology (stars/chains/cycles,
controllable type-ambiguity), where "can any intrinsic proposer induce joins that are
genuinely there?" is answerable — CFQ cannot answer it (join structure underdetermined at
this granularity; MCD confounds fit-coverage).

## Tags

| Claim | Tag |
|---|---|
| typed-slot parser matches probe-1 parsing rules; sentinel/merge/label conventions | **empirical** (unit-tested, 13 tests) |
| C = 0.215 on random IID (< 0.7 pre-registered threshold, < A = 0.332) | **empirical** (mcd1/2/3 + random) |
| type-merge family ≤ ~0.5 even with oracle slot occupancy (C_inc) | **empirical** (diagnostic; quarantined from positive verdicts) |
| fragmentation (mixed mined types on one gold var) drives the C_inc deficit | interpretation — **open** (not decomposed) |
| a richer join model reaches CFQ joins | **open** — recipe plateau across three families; NOT unbridgeable |

## Honesty notes

- Tables are mined from CFQ train, **not** the real Freebase ontology; a real-ontology
  assembler is untested (open), though it would face the same C_inc ceiling mechanism.
- Any schema-derived join assembler is `template_fixed`/schema-derived and must **not**
  count toward `frac_induced` ([provenance audit](residual_templates.md)) — moot for this
  negative, binding for any future positive.
