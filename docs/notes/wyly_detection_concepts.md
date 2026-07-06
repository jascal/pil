# Rule detection & concept formation: the overnight batch (items 7–15)

One protocol (the full stack: mined library, cover judge, sw arbitration, concepts, pointers),
pythia-70m rung, focused ablations. Flags: `WYLY_DX` (7+8+11), `WYLY_CX` (9+15v1),
`WYLY_GROW` (13+14). All empirical, single seed; logs in the review artifacts dir.

| # | item | status | result |
|---|---|---|---|
| 7 | anti-unification | **implemented, active** | 5–11 anchor pairs/sleep feed ConceptSpace as frame-tier evidence; cores flat — the pairs largely re-derive merges the bigram rows already found |
| 8 | CEGIS retreat | **implemented** | hygiene-neutral on clean cells (the val-variance-era pre-gates already did this work); protective machinery for noisier settings |
| 9 | cluster-scoped mining | **implemented, contributes** | 1–3 extra frames/sleep from concept-space error clusters; part of the CX wins below |
| 10 | analysis-route proposer | **v1 implemented** | `wyly_probe_proposer.py`: pythia-70m's equality-attention concentrates **entirely in layer 2** (all 8 heads, 3 above z=1) → proposals = relation/induction/pointer — the families the judge already admits: **two-instrument confirmation at the mechanism level**. Named gap: static score ignores rotary phases; offset-specific proposals need a forward probe (fieldrun-side) |
| 11 | MDL admission | **implemented** | penalties negligible on healthy tables (1e-8/entry — a 50k-entry table must beat double threshold); working as designed: a brake, not a tax |
| 12 | concept induction | **done previously (PR #16)** + multi-tier evidence channel via #7 |
| 13+14 | append-only C growth + compounds | **implemented; clean negative** | 200–270 high-PMI compounds mined and trained (C frozen, decode untouched — C9-compatible); code student −0.006, de flat: compound delta-vectors don't pay at this scale |
| 15 | relational concepts | **v1 implemented, ADMITTED** | the mate gate (innermost unclosed opener via prefix-depth + reverse-cummin — a derived predicate, gated on exactly) **admitted on isabelle** (+0.0013 cover marginal); declined on code (+0.0003, brackets already claimed by pointer/frames). The first two-layer rule in the library. Full two-layer Datalog packages remain the named future |

## Headline numbers (ablation B: CX = cluster mining + mate gate, on top of the full stack)

- **isabelle: core_sw 0.573 → 0.586 (+0.013, new arc best)** — mate gate admitted; student
  0.573 → 0.580. Proof text rewards the first structural (non-memoization) rule immediately.
- **wikitext: core_sw 0.329 → 0.338 (+0.009, the largest wikitext gain of the entire arc)** —
  cluster-scoped mining finds frames the offset grid missed (2 per late sleep from ~3.7k
  errors), on the corpus where every previous family had saturated.
- code: 0.610 flat (mate gate declined — its structure was already claimed).

## Read

The detection items (7, 8, 11) are **protective**, not additive, on already-clean cells — their
value is exactly what the val-variance episodes paid for manually. The formation items split:
growth into the *soft* space (13/14) doesn't pay; growth into the *rule* space — derived
predicates (15) and cluster-scoped frames (9) — sets new bests on the two corpora that had
plateaued differently (isabelle via structure, wikitext via scope). The path past the prose
plateau continues to run through rules-over-derived-features, not more memory.
