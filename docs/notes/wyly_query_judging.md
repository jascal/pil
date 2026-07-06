# Query-shaped judging + stratified admission

The third-sighting lesson (window-shaped val hides deployment value) gets its machinery — and the
first results reshape it.

## Four judge configurations, one benchmark (elements, full 708)

| judge | rules admitted | bench |
|---|---|---|
| window (fold-stable) | 6 | **0.753** |
| query-only, first-token | **1** (collapse) | 0.739 |
| query+window BLEND (first-token*) | 5 | 0.751 |
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

## Correction and the bAbI verdict

**Correction**: the elements "blend" run above scored queries by FIRST TOKEN only — the
chain-scoring edit had silently rolled back (an edit script asserting mid-way rolls back ALL its
in-memory edits; verify with grep after any multi-part edit). The blend alone fixed the
collapse. Chain scoring landed with the bAbI work below.

**bAbI, 1000 valid-split chain-scored judge queries + strata, package on 1000 unseen test:**

| config | package |
|---|---|
| window-judged (previous best) | 0.527 |
| + strata qualified on WINDOW fired-acc | **0.222** |
| + strata qualified on QUERY fired-acc | **0.527** |

The 0.222 crash is the sharpest strata lesson: window fired-accuracy admitted the stale
pointer (0.71 on windows) to stratum 2, and the fall-through zone — low stratum-1 confidence on
unseen stories — is exactly where it is wrong. **The fall-through pool must be calibrated on
the deployment distribution** (C10's optimality premise, applied per stratum). Query-qualified,
the pointer is correctly excluded (query fired-acc < 0.5), stratum 2 keeps kgram k=10 (0.56),
and the package returns to parity.

**The movement rules got their fair hearing and lost it fairly**: with 1000 deployment-shaped
chain-scored queries in the judge, pointer/moveloc still don't clear admission — their query
value is genuinely insufficient as built. The bAbI residual (0.527 vs Qwen 0.782) is now
attributed to the RULE LIBRARY, not the judge. Strata are servable end to end (schema stratum
field; rosetta PR #40, sgiandubh PR #25).