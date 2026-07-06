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

## 15-full: shipped — the two-layer package is real

The package schema now speaks both layers (rosetta PR #33, sgiandubh PR #16, pil this branch):

- **`derived`**: feature extractors shipped extensionally (bracket-mate: openers/closers as token
  sets, feature = innermost unclosed opener). The extractor's semantics is **Soufflé-certified**
  against the tensor mirror (`wyly_mate_certify.py`: recursive prefix-depth + stratified negation
  + max aggregate ≡ prefix-cummin tensor, **256/256 PROVED** on isabelle windows).
- **`dgate`**: rules gating on (derived feature, last token) — guards that are computed ROLES.
  Nesting semantics unit-tested in both runtimes (inner opener wins; closed pair abstains; outer
  becomes innermost after close).
- **`pointer`** kind + **`concepts`** map: the two deferred emissions, completing package
  fidelity for the modern library.
- **End-to-end**: the isabelle package (5,525 rules: pointer + 2 induction + 88 gates + 5,434
  ngrams + 189 concept groups) serves at **exact parity with the arc best — 0.586 @ 99.9%**
  (12k windows, python; C++ spoke 200/200 over HTTP).

Named honestly: the mate gate's admission is variance-fragile (+0.0010–0.0013 against a 5e-4
threshold, racing other candidates per sleep — admitted in the recorded B-ablation run, edged out
in the emit re-run by GPU-atomics nondeterminism), so this isabelle manifest carries the pointer
and concept layers but no dgate; the dgate path is verified by certificate + unit test. Multi-run
admission (admit if it wins any of k seeds) is the obvious stabilizer. Also fixed here: the
STATE-suffix chain bug (stack runs were overwriting the base-suffix state file; all reported
ablation numbers came from run logs and are unaffected).