# Certified Early Exit — Outcome: HALTED

Pre-reg: [`certified_early_exit_prereg.md`](./certified_early_exit_prereg.md) (SIGNED 2026-09-25, Addendum A pre-numbers).
Data: Qwen2.5-0.5B-Instruct fieldrun bundle, `--source-dump` at fieldrun `d17438c` (CAL 640 / EVAL-PROSE 320 /
EVAL-CODE 320 positions). Numbers: `results/certified_early_exit.{txt,json}`. Weight bound committed before any dump:
`results/certified_early_exit_weight_bound.json`.

## Verdict: HALTED `[empirical]`
The oracle bound (the *actual* norm of what the remaining layers write) certifies **0/320** positions on both EVAL
splits at **every** exit layer, including `k = 22`, where only the last layer remains. The weight-derived and
calibrated bounds certify 0 as well. By the signed rule: even perfect knowledge of the suffix **norm** never certifies
on this model, so norm-bounded early exit is dead here, and any early-exit certificate needs directional information
about the suffix.

## Controls and soundness gate: all pass
- `s`-recovery residual max `1.2e-5` (bar `1e-3`); full-vocabulary reconstruction `1.000` on all three splits.
- Violations: 0 for every bound (trivially, since nothing certified). Harness soundness and tightness are covered by
  `tests/test_early_exit.py`, which includes a worst-case suffix just past the radius that does flip the decision.

## The numbers that decide it
| exit `k` | median radius `R_k` (y coords) | median suffix norm | median suffix / `R` (where `t_k` = decision) |
|---|---|---|---|
| 0 | 0.013 | 29.7 | ~800 |
| 12 | 0.035 | 29.3 | ~2000 |
| 20 | 0.11–0.12 | ~28 | 60–77 |
| 22 (one layer left) | 0.21–0.23 | ~29 | **84 (prose) / 50 (code)** |

The last layer alone writes a vector about as large as the whole final residual (`‖y_final‖ = √896 ≈ 30` by
construction), while the margin that decides the token tolerates only about 1% of that. Nearly all of the last
layer's write is orthogonal to the decision directions `w_t − w_v`.

## Mechanism read (exploratory, post-hoc — not a decision variable)
- About 40% of the last layer's write (in y coordinates) sits in dimension 490, where the final-norm gain is
  `θ_f = −0.011` (median `|θ_f|` 6.9). That is the massive-activation channel the final norm suppresses.
- It is **not** the whole story, and a different norm does not rescue the certificate. The gain-weighted variant
  (`‖θ_f ⊙ suffix‖` against distances in `U` coordinates, equally sound) still certifies **0/64** prose positions at
  `k = 22`, with the suffix at a median **122×** the radius.
- Norm-only bounds fail because the late write is large and mostly decision-irrelevant, not because of one
  coordinate choice.

## Headroom was small anyway (the uncertified reference)
`k★` = the earliest exit after which the prefix argmax stays equal to the final decision:
- prose: median 23, exitable at all 46%, **mean skippable 1.27 layers**;
- code: median 22, exitable 52%, **mean skippable 1.08 layers**.

Even an oracle that *knew* the answer could skip only about one layer on average. One full-vocabulary check costs
**9.13 layer-equivalents** on this model. So on Qwen2.5-0.5B, early exit cannot pay for a full-vocabulary check,
certified or not. This agrees with the deep-computation finding in `pic` §8 (convergence depth `k★/L ≥ 0.8`).

## Predictions vs outcome
| prediction | outcome |
|---|---|
| `B_w` vacuous (< 1%) | ✓ (0%) |
| `B_or` certifies mostly in the last few layers | ✗ — 0% even with one layer left |
| code covers more than prose | not testable (both 0) |
| `B_cal` between the two | degenerate (all 0) |
| net saving negative under P-every | ✓ |
| most likely IN-BETWEEN | ✗ — **HALTED** |

## Consequences
1. **Norm-bounded early exit: do not build** on this model, certified or calibrated.
2. **If early exit is revisited, it needs direction:** bound the suffix's *projection* onto the few decision
   directions `w_t − w_v` that matter (the near rivals), for example calibrated per-direction quantiles, or read
   through `--jlens-export` averaged Jacobians. Only rivals with a small radius need bounding, which also offers a
   way to avoid the full-vocabulary pass.
3. **Scale first:** with about one layer of headroom at 0.5B, the payoff ceiling is tiny whatever the certificate.
   Pythia's convergence depth falls with scale (`pic` §8, Fig. B), so a larger model is where to look for headroom
   before building anything directional.
4. For `pic`, this is an `[empirical]` negative for norm-only pairwise early exit on Qwen2.5-0.5B. It does not show
   that certified early exit is impossible.

## Scope
One model (Qwen2.5-0.5B-Instruct, int8 bundle, CPU), one seed, 640 EVAL positions from wikitext-2 prose and C/C++
code. RMSNorm path only.
