# Query-shaped judging + stratified admission

The third-sighting lesson (window-shaped val hides deployment value) gets its machinery — and the
first results reshape it.

## Four judge configurations, one benchmark (elements, full 708)

| judge | rules admitted | bench |
|---|---|---|
| window (fold-stable) | 6 | **0.753** |
| query-only, first-token | **1** (collapse) | 0.739 |
| query+window BLEND, chain-scored | 5 | 0.751 |
| blend + **STRATA** (user-proposed) | 5 + **4 stratum-2** | 0.751* |

*the emitted package does not yet carry stratum 2 — see below.

## Findings

1. **Small query sets collapse admissions**: 138 queries are covered by one k=6 tier; every other
   candidate shows zero query marginal and dies. Deployment-shaped ≠ sufficient — the query set
   must be large and diverse, or blended with window signal (the blend restored 5 admissions and
   benchmark parity).
2. **Chain scoring is necessary**: first-token scoring is blind to generation-time value (the
   mass lesson); `query_agree` now greedy-generates the full answer and requires every token.
3. **Stratified admission (user-proposed) works as designed**: calibrated non-winners
   (fired-accuracy ≥ 0.5) enter a labeled stratum-2 pool instead of oblivion — the first
   population caught exactly the rules the per-sleep races kept killing (tpointer 0.52,
   three moveloc gates 0.67–0.79). Serving falls through to stratum 2 where stratum-1 confidence
   < τ (0.35). C10 applies per stratum; the fall-through is lexicographic (stratum, confidence).
   The architecture also gives claymore its graceful-degradation story: answers carry their
   stratum — certified / supported / tentative — extending provenance instead of breaking the
   refusal contract.
4. **Honest residuals**: mass (0.017) is structural (chain-vs-corpus tokenization divergence),
   unmoved by any judge; stratum-2 rules are in-learner only — the package schema needs a
   `stratum` field + runtime fall-through (the relation-kind playbook) before serving sees them.
