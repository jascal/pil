# CFQ relational-join headroom (modality-gap probe 1)

**Status:** measured, decisive negative (2026-07-11). The first probe into the
**modality gap** — the induction machinery is a `(x, marker) → tgt` *transduction*
learner, and CFQ's exact SPARQL stays ~0 because the relational **join** (which
predicates share a variable) is invisible to it. This probe asks, *before building any
mechanism*: is the join **schema-deterministic**, **question-predictable**, or neither?
Measurement only, oracle-predicate regime. `pil/cfq_edges.py` (`parse_sparql_joins`,
per-var join signatures, `join_f1`) + `experiments/campaign_cfq_join_headroom.py`.

## Why the join is the gap
Bag-of-predicates (`sparql_ns`) and the role-typed edges from
[the edge diagnostic](cfq_edge_headroom.md) both **discard variable identity** —
`_normalize_role` (`cfq_edges.py`) replaces `?x0` with its *type*, so "these two triples
share `?x0`" (the join) is destroyed. This probe builds the first var-identity-preserving
representation.

## Representation (isomorphism-invariant)
`parse_sparql_joins` keeps raw `?x` identity. A **join var** = a variable with ≥2
incidences (appears in ≥2 non-type triples). Its **signature** =
`(type_or_VAR, sorted multiset of (pred, slot))` — **label-free**, so a 3-star (one var in
3 predicates) and a 3-cycle (3 distinct vars) do **not** collide (unit-tested `star != cycle`;
pairwise-pair representations would collide them and manufacture false determinism). A
query's join object = the multiset of its join-var signatures; `join_f1` = multiset F1.

## Result — oracle-predicate regime, mcd1/2/3 + IID `random` cross-check

| split | A (majority-join prior) | B (lexical→join) | unseen-key rate | frac w/ join |
|---|---:|---:|---:|---:|
| mcd1 | 0.191 | 0.015 | 0.54 | 0.82 |
| mcd2 | 0.090 | 0.010 | 0.42 | 0.88 |
| mcd3 | 0.152 | 0.023 | 0.36 | 0.86 |
| **random (IID)** | **0.332** | 0.019 | 0.28 | — |

- **A** — majority join-signature conditioned on the *exact gold predicate multiset*
  (the schema-determinism test). Fit-unseen key → **empty** prediction (no gold-join
  fallback — that would leak gold structure and manufacture determinism).
- **B** — lexical `content_word → signature` votes (the question-predictability test).
- Ceiling = 1.0 by construction (reading `gold.signatures` scores 1.0; not a predictor).

## Finding (decisive)
Neither predictor comes near determining CFQ join topology. On the MCD splits, **A** is
depressed by a 36–54% unseen-key rate — maximum-compound-divergence deliberately puts half
the test predicate-multisets outside fit (a confound, not a join fact). The IID `random`
cross-check removes most of it (unseen → 0.28) and A rises to **0.332** — real, but still
far below schema-determinism (≥0.9). Lexical features add **no** signal (B ~0.01–0.02,
*below* A on every split).

So, at this granularity — *exact-predicate-multiset majority* + *lexical word-votes* — the
CFQ join is **not predictable**. This rules out the two obvious first mechanisms:
- a **majority-vote schema assembler** (A) — the predicate multiset does not force the join;
- a **lexical feature → join inducer** (B) — question words don't surface it.

## Tags
| Claim | Tag |
|---|---|
| var-identity join representation; isomorphism-invariant signatures (star ≠ cycle) | **empirical** (unit-tested) |
| CFQ join not determined by exact-multiset majority vote (A ≤ 0.33 incl. IID) | **empirical** (mcd1/2/3 + random) |
| CFQ join not predictable from lexical word-votes (B ≪ A) | **empirical** |
| a richer model (typed schema constraints / stronger features) reaches the join | **open** — recipe plateau; **NOT** unbridgeable |

## What it means for the modality gap
The two simplest join mechanisms are ruled out. A genuine join model would need **typed
schema constraints** — Freebase `predicate → required arg types → forced shared vars`
(e.g. `person.spouse` joins two person-typed vars) — a *different, richer* mechanism than
the transduction inducers, or a stronger question-feature family than lexical votes. That
is a real decision point for the research lead: **build the typed-schema join model, or
record CFQ joins as beyond the current induction paradigm.** Either way, this probe cost a
parser extension + one diagnostic and produced decisive guidance — no mechanism built on a
false premise.
