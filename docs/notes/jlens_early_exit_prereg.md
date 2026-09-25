# J-lens Early Exit — Pre-Registration

**Status: DRAFT — awaiting signature. Decision rules are fixed BEFORE numbers. No J-lens quantity has been computed
on the early-exit dumps.** A follow-up to [`certified_early_exit_outcome.md`](./certified_early_exit_outcome.md) (#126,
HALTED). That outcome's diagnosis: the remaining layers write a vector about as large as the whole residual, mostly
orthogonal to the decision directions, so a **size-only** bound can never certify. Its stated next step: add
**directional** information. The J-lens is the directional model fieldrun already has for this exact model.

## 1. The move
Replace "bound everything the remaining layers write" with "predict it, and bound only the unpredicted leftover". At
exit `k`, predict the final residual as `ŷ_k = J'_k · y_k` (J-lens, shrinkage `J' = (1−λ)I + λJ`). The leftover is
`e_k = y_final − ŷ_k`. Since `y_final = ŷ_k + e_k` exactly, the same Cauchy–Schwarz argument as before gives: the
J-read argmax `t̂_k = argmax_v ⟨ŷ_k, w_v⟩` is the model's decision if

> `∀ v ≠ t̂_k:  ⟨ŷ_k, w_t − w_v⟩ > B · ‖w_t − w_v‖`,   with `B ≥ ‖e_k‖`.

`λ = 0` is exactly the #126 test. **This can never be a certificate:** `J` is a context-averaged fit, so no sound
input-independent bound on `‖e_k‖` exists. The best outcome is a statistical guarantee (conformal), tagged
`empirical`.

## 2. PINS
- **PIN A — model, data, dumps: unchanged from #126.** Qwen2.5-0.5B-Instruct bundle; the same CAL (640) / EVAL-PROSE
  (320) / EVAL-CODE (320) source dumps (fieldrun `d17438c`), the same `y = d̃/θ_f` coordinates and `s`-recovery,
  and the same full-vocabulary radius code (`pil/early_exit.py`, `radii_batch`).
- **PIN B — the J-lens.** `bundles/Qwen2.5-0.5B-Instruct/Qwen2.5-0.5B-Instruct.jlens` (file dated 2026-07-08),
  exported with `fieldrun --jlens-export` to `J [24, 896, 896]`, layers 0–22 fitted, `J[23] = I`. The fit corpus
  (`fieldrun/experiments/jlens/fit_corpus.txt`, 300 lines of technical prose) shares **0/300** prompts with our
  splits (checked). `J` maps the post-block residual of layer `k` to the final pre-norm residual. That map is linear,
  so it applies unchanged in `y = s·x` coordinates.
- **PIN C — shrinkage.** `λ ∈ {0, 0.25, 0.5, 1.0}`. The decision uses `λ*`, chosen on **CAL** as the `λ > 0` with the
  highest CAL coverage of the calibrated bound (ties → smaller `λ`). All `λ` are reported descriptively.
- **PIN D — bounds on `‖e_k‖`.**
  1. **Oracle** `B_or(k) = ‖e_k‖`, the actual leftover. The coverage ceiling for this J-lens.
  2. **Calibrated** `B_cal(k) = q̂_k · ‖y_k‖`, with `q̂_k` the conformal `(1−α)` quantile over CAL of
     `‖e_k‖/‖y_k‖`, `α = 0.05`, rank `⌈(n+1)(1−α)⌉`. The same construction as #126.
- **PIN E — cost and policies:** as in #126 (`c_chk = 9.13` layer-equivalents), plus `d²/layer ≈ 0.054` for applying
  `J`. P-every and P-single (`k₀` chosen on CAL).

## 3. Metrics (per EVAL split)
Oracle and calibrated coverage at `λ*` (any exit `k ≤ 22`, and with ≥ 2 layers skipped); `k_c` distribution;
violations; net saving (P-every, P-single).

**Mechanism metric:** median `‖e_k‖ / ‖suffix_k‖`, the fraction of the late write that `J` does *not* predict, at
`k ∈ {12, 20, 21, 22}`. **Headroom:** `k★_J`, the earliest exit after which the J-read argmax stays equal to the
decision, compared with #126's `k★` (mean skippable 1.27 prose / 1.08 code).

## 4. Decision rules (FIXED BEFORE NUMBERS)
- **FIRES-EMPIRICAL**: at `λ*`, calibrated coverage ≥ 25% on either EVAL split, violation rate ≤ 2α of certified,
  **and** P-single net saving > 0. → A statistically guaranteed early exit, worth building.
- **IN-BETWEEN**: at `λ*`, oracle coverage ≥ 25% on either split, but FIRES-EMPIRICAL is not met. → Direction was the
  missing piece: the J-lens captures enough of the late write to make the pairwise test work in principle. It is not
  deployable here, and it motivates the same test on a larger model.
- **WEAK**: best oracle coverage (any `λ > 0`, either split) is ≥ 1% but < 25%. → The J-lens explains some of the
  late write, but not most of it.
- **HALTED**: oracle coverage < 1% on both splits for every `λ > 0`. → The averaged Jacobian does not predict the
  late write at this scale. The J-lens is ruled out for early exit on this model.
- **SANITY GATE**: `λ = 0` must reproduce #126 exactly (oracle coverage 0/320 on both splits, identical radii). The
  oracle must show 0 violations at every `λ`. Otherwise the run is void.

## 5. Architect's predictions (recorded, not decision variables)
- **FIRES-EMPIRICAL is effectively unreachable.** P-single net > 0 needs more than 9.13 layers skipped on average
  among certified positions, and prior measurements put J-lens headroom at only about 0.5 layer better than the raw
  read (pil `jlens_correction_sweep`: Δresolve −0.02…−0.03 of depth on this model). Stated up front so a negative
  economic result is not read as news.
- `‖e_k‖/‖suffix_k‖` drops well below 1 at `k = 22` (the fit is closest to the identity there, and the massive-
  activation write is systematic), but stays large at mid-depth, where the fit is noisy (`σ√d`) and was context-
  fragile in fieldrun's evaluation.
- Most likely outcome: **WEAK**, with any coverage concentrated at `k = 21–22`.

## 6. Controls
- The `λ = 0` reproduction (sanity gate) and the #126 controls (s-fit, reconstruction), reused.
- `λ*` and `q̂_k` fitted on CAL only; EVAL read once.
- Property test: on synthetic data, the J-read certificate with `B = ‖e‖` never certifies a wrong answer.
- pil `ruff` + `pytest`.

## 7. Risks / open
- **Fit-corpus shift.** `J` was fit on technical documentation prose, not wikitext or code. A poor transfer shows up
  as a large `‖e_k‖`, and would be a fact about this fit, not about J-lenses in general.
- **Multiple looks** (P-every): the per-`k` conformal guarantee, as in #126.
- **Fit quality at 0.5B.** fieldrun's own evaluation found the J-space effect does not reproduce cleanly at this scale
  (JLENS.md, "Status & honest findings"). A HALTED here inherits that caveat.

## 8. Scope fences
Measurement only, on existing dumps and the existing `J` fit: no refit, no new dumps. Qwen2.5-0.5B-Instruct, RMSNorm.
