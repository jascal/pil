# CFQ join-structure headroom (role-typed edge diagnostic)

**Status:** measurement landed (2026-07-11). `pil/cfq_edges.py` (SPARQL→edge parser
+ multiset `edge_f1` + `role_typed`) and `experiments/campaign_cfq_edge_headroom.py`
measure how much CFQ argument-role structure is predictable *beyond* the predicate bag.
Two commits: parser + metric (step A), diagnostic (step B). No decoder is built — this
note decides whether one is worth building.

**Why.** Bag-of-`ns:`-paths set-F1 (`campaign_cfq_residual.py`) plateaus ~0.14–0.17 near
a frequency prior and is blind to argument roles / direction (`influenced` vs
`influenced_by`; `M2 directed_by M3` vs `…M4`) and the shared-variable join. This note
introduces a *finer* structure metric and measures whether that finer structure is
learnable per question — before committing to a modeling slice.

## Metric — role-typed edge-F1
SPARQL `WHERE` body → **multiset** of normalized edges `(s_role, pred, o_role)`:
- compound property paths (`ns:a/ns:b|ns:c`) are ONE predicate (never split — splitting
  reifies a nonexistent intermediate node);
- `rdf:type` triples (`?x a ns:T`) are consumed into a var→type map, never scored;
- variables → their recorded type else `VAR`; `FILTER` clauses excluded (counted);
- multiset, so `directed_by M3` / `directed_by M4` are two edges, not one.

Then `role_typed` abstracts concrete entities (M-mentions + grounded `ns:m.*`/`ns:g.*`
MIDs) to `ENT`: M-ids are **query-local** and not globally predictable, so the metric
scores argument-slot **kind** + directionality + multiplicity — **not** entity identity
(a separate entity-linking problem). Strictly finer than the predicate bag; the bag
number is kept only as an *incommensurable reference* (it re-extracts split `ns:`
fragments, this uses full-path predicates).

Stratify **entity-anchored** (≥1 `ENT`) vs **var-join** (both non-`ENT`). Real mcd1:
4.0 edges/query, anchored fraction 0.74.

## Diagnostic (measurement, no promised lift)
Per split, role-typed edge-F1 of:
- **oracle-pred + majority-role** — gold predicates *with multiplicity*, each assigned its
  *global majority* `(s_role, o_role)` learned on fit (predicates unseen in fit fall back
  to the true gold role, which only **inflates** the oracle → headroom is conservative).
  This isolates role assignment from predicate recall: the ceiling a global majority-role
  prior can reach with perfect predicates.
- **role-prior (realistic)** — a mined word→edge-predicate decode + majority role;
  predicate-recall-limited in full compound-path space.

## Result — empirical, 3 MCD splits

| split | oracle overall | oracle anchored | oracle var-join | (bag ref) |
|---|---|---|---|---|
| mcd1 | 0.427 | 0.446 | 0.799 | 0.166 |
| mcd2 | 0.342 | 0.363 | 0.746 | 0.145 |
| mcd3 | 0.377 | 0.403 | 0.739 | 0.139 |

**Finding (empirical, robust across splits).** Given *perfect* predicate recall, a global
majority-role prior recovers only **0.34–0.43** of role-typed edge-F1. Argument-role
structure is therefore **not** a deterministic function of the predicate — **~0.57–0.66
of it varies per question**, and that variation is concentrated in **entity-anchored**
edges (oracle 0.36–0.45) far more than **var-join** edges (oracle 0.74–0.80, i.e. var-join
roles are largely predicate-stereotyped). This is the first CFQ structure signal beyond
predicate frequency that is real, finer than the bag, and *not* already captured by a
frequency-style prior — measured headroom for a question-conditioned role decoder over
anchored edges.

**What it does NOT show.** The realistic mined decode (`role_prior`) scores 0.05–0.07 —
predicate-recall-limited in full-path space (2681 edge-predicates vs the bag's fewer
`ns:` fragments). Its var-join stratum (0.000) is a `min_support=2` mining **artifact**:
exactly 3 rare film-crew predicates (`ns:film.film.film_art_direction_by`,
`ns:film.film.cinematography`, `ns:film.film.costume_design_by`) are the only var-join
majority predicates and each is spuriously predicted in all 3000 test queries, crushing
precision. It is reported for the predicate-recall-cost gap (`oracle − role_prior ≈
+0.36`) only, **not** as a role-prior measurement.

## Tag discipline
| Claim | Tag |
|---|---|
| role-typed edge parser + multiset edge-F1 | **empirical** (golden fixtures unit-tested) |
| per-question argument-role headroom on anchored edges (oracle < 1, 3 splits) | **empirical** (mcd1/2/3) |
| var-join roles largely predicate-stereotyped (oracle 0.74–0.80) | **empirical** |
| realistic mined role-prior number (esp. var-join 0.000) | **artifact** — `min_support=2`, not a clean signal |
| a question-conditioned decoder *achieves* this headroom | **open** (no decoder built) |
| exact SPARQL generation from edges | **open** (~0) |

## Next
The measurement says a question-conditioned role decoder for **anchored** edges is worth
building — the headroom exists, is stable across MCD splits, and is not predicate-
determined. Var-join roles sit near the predicate-stereotype ceiling (less to gain there).
Either path also needs better full-path edge-predicate recall (edge-predicate schemas, or
IDF-reweighted mining) before role-conditioning pays off end-to-end. The parity-verified
set-F1 selector from the [residual→schema bridge](residual_as_schema.md) is the natural
oracle for a later differentiable edge decoder.
