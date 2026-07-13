# Non-deterministic beam validation — the energy ranks the right guess (substantive FIRES; signature splits)

**Status:** measured + registered, cross-vendor (2026-07-13). The substantive test the #101 reproduction
couldn't give — the beam engine validated where it actually BRANCHES. Pre-registered in
`PIL_NONDET_VALIDATION_PREREG.md` (SIGNED before numbers; **one CORRECTION after the run** — a confounded
signature, user-approved). `experiments/campaign_nondet_validation.py` (grok) +
`experiments/nondet_validation_codex.py` (codex), independent lanes (raced, independence audited clean).
**Verdict: the substantive result FIRES cross-vendor; the strict rising-signature SPLITS (grok fires,
codex boundary) on a single-cell pre-branching noise dip.**

## Why this pilot exists
The #101 reproduction validated the engine in the DEGENERATE regime: on propagation-solvable sudoku the
beam NEVER branched (`prune_bite = 0`). This pilot tests the energy as a *ranker* on GUESS-REQUIRING
puzzles (propagation stalls → the beam must branch, prune contradictions, rank survivors), on BRANCH cells
(`propagation_depth is None`), vs the M=1 det_rank baseline.

## Result
In the branching regime (`prune_bite > 0`, M≥6), the pinned energy commits gold on ~63–71% of branch
cells at M=8 vs a ~0.33 per-token baseline — the substantive FIRES conditions hold in BOTH lanes at M=7,8:

| lane | baseline | beam_acc @ M8 (population) | SEP @ M8 | prune_bite @ M8 | rising signature | verdict |
|---|---|---|---|---|---|---|
| grok  | 0.33 | 0.705 | **+0.37** | 0.41  | monotonic → holds | **FIRES** |
| codex | 0.34 | 0.626 | **+0.29** | 0.049 | 1-cell dip M=2→3 → fails strict | **BOUNDARY** |

The **substantive signal is cross-vendor robust**: both lanes clear SEP≥0.15, beam_acc≥0.50, prune_bite>0
at M=7,8, and both show a dramatic overall rise (grok 0.33→0.71, codex 0.34→0.63 as M:1→8). The **split is
a signature technicality**: codex's population `beam_acc` correct-counts are `[344,344,343,355,380,449,566,
633]/1011` — one cell flips correct→incorrect at M=2→3 (0.1 pp, in the pre-branching flat region), which
breaks the *strict* "monotonic non-decreasing across M=1..8" reading → BOUNDARY. It is a noise dip in the
region where the beam hasn't started branching (prune_bite=0), NOT a substantive disagreement. Grok's random
puzzle set happened to have no such dip.

## The corrections (a confound, then honest limits — no p-hacking)
1. **Confound (caught cross-vendor BEFORE any verdict).** Both lanes first hit a `r==M`-diagonal signature
   that FAILS because `r` counts GUESSES but the beam's `M` counts PROPAGATION passes + guesses
   (`prune_bite≈0` until M≥6 proves the beam propagates ~5 passes before branching → it forces an `r`-cell
   at M≈6+, not M=r; the `r==M` diagonal measured the pre-branching beam). Same class as gate-(b) run 1.
   **User-approved fix:** the rising signature is the fixed branch-cell **population** `beam_acc(M)` rising
   with M, not the `r==M` diagonal.
2. **Under that corrected signature the result SPLITS** — grok clean-monotonic (FIRES), codex one-cell dip
   (BOUNDARY). Rather than invent a further restriction to force "FIRES both" (M≥prune-onset only, a
   tolerance, etc. — each un-registered tuning), this is recorded HONESTLY as the split it is. The strict
   "monotonic across all M" criterion is brittle to pre-branching noise; that brittleness is the honest
   limit, disclosed rather than papered over.

## Honest scope + tags
| Claim | Tag |
|---|---|
| the pinned energy commits gold on ~63–71% of branch cells vs a ~0.33 per-token baseline (SEP +0.29–0.37), cross-vendor, in the genuinely-branching regime | **empirical**, cross-vendor — the energy is a better RANKER of the right guess than confidence when the beam must guess |
| the strict population-monotonic rising signature holds | **grok yes / codex no** — a 1/1011 pre-branching noise dip; the substantive rise is unambiguous in both, so this is a signature-brittleness split, not a substantive one |
| the FIRES holds on SHALLOW-branch cells (`r ≤ 3`) | **scope** — the signed uniform-45-hole generator yields only ~0.18% stall+unique puzzles, almost all shallow; DEEPER search (`r > 3`) is untested (codex's insight). A harder / lower-clue generator is the registered next lever. |
| the signal needs DEEP lookahead (M≥7; M=8 = the beam's k·M≈64 budget) | **scope** |
| this generalizes to real text | **NOT shown** — sudoku ground-truth only; the real-text payoff is the lead's separate wikitext test (next) |

## Process (RACE + two honest catches)
Raced grok + codex from scratch (independent generators, refutation-depth, metric; codex independence
audited clean). BOTH hit the identical confound + substantive signal. The lanes' machinery categorized the
final signature-block differently (grok's data is clean-monotonic → fires; codex's has the 1-cell dip →
boundary), agreeing field-for-field on the DATA. NO data changed under the signature correction (the
confound was in the CRITERION). Both catches — the `r==M` confound and the population-monotonicity split —
were surfaced before recording, not tuned away. See [[lane-balancing-rule]], [[sequence-energy-direction]],
[[beam-arc-followups]], [[grok-cli-permission-recipe]] (a grok CLI edit-cancel/fabrication was caught via
independent mtime/diff verification this slice).
