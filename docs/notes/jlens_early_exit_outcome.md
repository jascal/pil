# J-lens Early Exit — Outcome: HALTED

Pre-reg: [`jlens_early_exit_prereg.md`](./jlens_early_exit_prereg.md) (SIGNED 2026-09-25). Follows
[`certified_early_exit_outcome.md`](./certified_early_exit_outcome.md) (#126). Same dumps, same radius code; the
J-lens is the fit shipped in the bundle (23 fitted layers). Numbers: `results/jlens_early_exit.{txt,json}`.

## Verdict: HALTED `[empirical]`
Reading the prefix through the J-lens and bounding only the unpredicted leftover still certifies **0/320** positions
on both EVAL splits, at every exit layer and at every shrinkage `λ ∈ {0.25, 0.5, 1}`, even with the *actual*
leftover as the bound (oracle). `λ*` (chosen on CAL) = 0.25 by the tie rule: every `λ` scored 0 on CAL. By the
signed rule, the averaged Jacobian does not predict the late write at this scale, and the J-lens is ruled out for
early exit on this model.

## Sanity gate: PASS
- `λ = 0` reproduces #126: identical argmax at every (position, exit), max relative radius difference `4.6e-12`, suffix
  norms identical.
- The oracle has 0 violations at every `λ`. The J-read certificate's soundness given an exact leftover bound is
  covered by `tests/test_early_exit.py`.

## Why: the J-lens predicts only about a quarter of the late write
Fraction of the late write the J-lens does **not** predict, `‖e_k‖ / ‖suffix_k‖` (median; 1.00 = predicts nothing):

| `λ` | k = 12 | k = 20 | k = 21 | k = 22 (one layer left) |
|---|---|---|---|---|
| 0 (plain prefix, #126) | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.25 | 0.98–0.99 | 0.92–0.93 | 0.89–0.91 | 0.86–0.88 |
| 0.5 | 0.97 | 0.87–0.88 | 0.81–0.84 | 0.76–0.80 |
| 1.0 | 0.96 | 0.86 | 0.79 | 0.76–0.78 |

In #126 the late write was **50–84×** larger than the certified radius at `k = 22`. To certify, the leftover would
have to shrink by roughly that factor. The J-lens shrinks it by about **1.3×** at best (to 76%), so the gap stays
around 40–60×. Direction was the missing ingredient, but this J-lens supplies only a small part of it.

## Headroom (uncertified): about half a layer better, as expected
Mean layers skippable if you exit where the J-read first locks to the final answer: prose **1.27 → 1.68** (`λ = 0.5`),
code **1.08 → 1.44**. That's about +0.4 layer, consistent with pil's earlier `jlens_correction_sweep` (Δresolve
−0.02…−0.03 of depth). One check costs 9.18 layer-equivalents, so the economics are unchanged: P-single net −9.18
everywhere.

## Predictions vs outcome
| prediction | outcome |
|---|---|
| FIRES-EMPIRICAL effectively unreachable | ✓ |
| `‖e‖/‖suffix‖` well below 1 at `k = 22` | ✗ partly — only down to 0.76 |
| still large at mid-depth | ✓ (0.96–0.99 at `k = 12`) |
| most likely **WEAK** | ✗ — **HALTED** (0 coverage at every `λ`) |

## What this does and does not show
- **Shows:** on Qwen2.5-0.5B, the shipped averaged Jacobian explains only about 24% of the late write. So neither a
  size-only bound (#126) nor a J-lens-predicted bound gets within an order of magnitude of certifying early exit.
- **Does not separate two causes:** (a) the late write is genuinely **input-specific**, not predictable by any single
  averaged linear map; or (b) this particular fit is too **noisy or too shifted**. It was fit on 300 lines of
  technical documentation with 5 probes, and fieldrun's own notes say the fit is `σ√d`-noisy and context-fragile at
  this scale.
- **A cheap way to separate them (not run; would need its own prereg):** fit the best linear predictor
  `y_final ≈ A_k y_k` directly on CAL (ridge, since CAL has 640 < 896 samples) and measure its leftover on EVAL. If
  even the in-distribution best linear map leaves the late write mostly unexplained, then early exit on this model
  needs per-input (nonlinear) information, and no averaged lens can supply it.

## Consequences
1. **J-lens early exit: do not build** on this model.
2. Across #126 and this study, early exit on Qwen2.5-0.5B is closed three ways: norm-only bounds (0%), J-lens
   directional bounds (0%), and headroom (about 1–1.7 layers against a 9-layer check).
3. If early exit is revisited, do it on a **larger model** (more headroom, and a cheaper check relative to a layer:
   about 2.3 layer-equivalents at 7B), and only after the ridge diagnostic above says an averaged predictor can work
   at all.

## Scope
Qwen2.5-0.5B-Instruct int8 bundle, CPU; the bundle's shipped J fit; 640 EVAL positions (wikitext-2 prose, C/C++
code). Measurement only.
