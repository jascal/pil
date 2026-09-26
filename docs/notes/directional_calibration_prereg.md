# Calibrating the Directional Early-Exit Bound — Pre-Registration (dev → confirm)

**Status: SIGNED 2026-09-26 14:29 UTC, before any EVAL2 decision was built or dumped. The selection and
confirmation rules below are fixed before EVAL2 numbers. The development data below has already been seen.**
A follow-up to [`restricted_early_exit_scale_outcome.md`](./restricted_early_exit_scale_outcome.md) (#130). There,
Oracle-D certified 89% of 3B decisions (median exit layer 9 of 36), but the baseline Calibrated-D certified only near
the end (median exit 34, net +0.39 < 1 layer). That outcome named **calibration** as the bottleneck.

## 1. Why two stages
The #129/#130 CAL and EVAL decisions (450 per model) have been analysed in detail, so tuning a calibrator on them and
reporting it there would be optimistic. This study therefore:
1. **Develops** on the seen data: a closed menu of calibrators and one selection rule.
2. **Confirms** the single selected calibrator **once**, on **fresh** decisions (EVAL2) that nobody has looked at.

Only the confirmation carries a verdict.

## 2. The calibrators (the full menu, fixed now)
At exit `k`, the observable quantities are `R_k` (the restricted radius) and `‖y_k‖`. The unobservable quantity is
the adverse push `a_k` (#130 Arm D), and a calibrator's job is to bound it. A decision is certified at `k` iff
`R_k > B_k`.
- **C0 — baseline** (#130): `B_k = q̂_k · ‖y_k‖`, with `q̂_k` the conformal `(1−α)` quantile of `a_k/‖y_k‖`.
- **C1 — margin-stratified, 3 bins:** bin each decision at each `k` by `m_k = R_k/‖y_k‖` into terciles
  (bin edges = CAL terciles of `m_k` in development), then use C0's quantile **within each (k, bin) group**. This
  tests whether separate thresholds let high-margin decisions certify earlier.
- **C2 — margin-stratified, 5 bins:** C1 with quintiles.

All use `α = 0.05`, the quantile rank `⌈(n+1)(1−α)⌉`, and P-every (check after every layer). A group
with rank `> n` gets `B = ∞` and never certifies. Ties at a bin edge go in the lower bin. If `‖y_k‖ = 0`,
exclude that observation from ratio fitting and use `B_k = ∞` at that look. C1/C2 are empirical stratified
calibrators here: bin edges and quantiles use the same seen data, so the usual fixed-bin conditional conformal
coverage claim does not apply. Moreover,
P-every selects among looks, so none of C0–C2 has a simultaneous per-decision error guarantee.
Use NumPy's `quantile(..., method="linear")` at probabilities `i/b` for `i=1,…,b−1` and `b=3` or `5`; assign
edge ties to the lower bin. Exclude the full-model index from fitting and checking.

The proposed Bonferroni candidate is excluded as infeasible: at 3B, 35 looks give `α/35`, requiring at least 699
calibration points **per bin** for a finite quantile. At 0.5B, 23 looks require at least 459 per bin. Even the
450 seen decisions per model fall short before binning. A simultaneous guarantee would need a separate study with
enough calibration data or a different pre-registered simultaneous score.

## 3. Stage 1 — development (seen data), per model (0.5B, 3B)
- Calibrate on the #129/#130 **CAL** (150) and evaluate on the #129/#130 **EVAL** (300), exactly as #130.
- **Selection rule (one choice, fixed before EVAL2):** among C0–C2, take the calibrator with the highest mean net
  saving on **3B** dev EVAL, **subject to** at least one certified decision and a dev violation rate ≤ α (5%) of
  certified. Mean net saving includes all 300 EVAL decisions and uses #130's P-every check cost. Exact ties use
  the fixed priority C1 > C2 > C0. If none qualifies, the study stops: **NO CANDIDATE**.
- Dev numbers are reported in full but carry **no verdict**.

## 4. Stage 2 — confirmation (fresh data)
- **EVAL2:** the same construction as #129 (parallel-decisions @ `45820320`, the `location` field, 6 options,
  canonical order). The contexts are the **next** items of each bAbI task's seed-0 permutation, indices
  `150 … 199`: 50 per task, **150 decisions**, never used before. Dumped with fieldrun `--tail 1` (`master`) on
  0.5B and 3B.
- **Calibration for confirmation:** refit the selected calibrator's bin edges (if any) and quantiles on **all seen
  data** (CAL + EVAL = 450 per model); none of it is EVAL2. The candidate was selected using EVAL, and C1/C2
  also fit edges on their quantile data. Thus the conformal rank is a reproducible threshold rule here, not a
  finite-sample coverage guarantee. EVAL2 tests the complete selected procedure empirically.
- **Decision rules (3B primary; 0.5B reported alongside, same rules):**
  - **FAILED:** coverage < 10%, including zero certifications, **or** violation rate > 2α (10%) of certified.
  - **CONFIRMED (D-FIRES):** otherwise, EVAL2 coverage ≥ 25% **and** mean net saving ≥ 1 layer. This is a
    pre-registered empirical success on fresh decisions, worth further engineering evaluation.
  - **PARTIAL:** all remaining outcomes (violation rate ≤ 10%, coverage ≥ 10%, but coverage < 25% or mean net
    saving < 1 layer).
  Apply these rules in the stated order. The violation rate is undefined when nothing certifies; such a run is
  FAILED by the coverage rule. Mean net saving includes all 150 EVAL2 decisions and #130's P-every check cost.
- **Soundness gate:** Oracle-D on EVAL2 has 0 violations (it is exact given `a_k`). Otherwise the run is void.
- Tagged `empirical` by construction: neither a sound input-independent projection bound nor a simultaneous
  statistical guarantee is established. Wilson intervals describe EVAL2 uncertainty; crossing the point-estimate
  violation threshold does not prove a population violation rate ≤ 10%.

## 5. Architect's predictions (recorded, not decision variables)
- **Dev:** C1 or C2 is selected. Margin stratification lets the high-margin third certify well before layer 34.
- **3B confirmation:** **PARTIAL.** Net saving rises above C0's +0.39, perhaps into the 1–4 layer range, but with
  n = 150 and per-bin calibration from 450, the violation rate and coverage are uncertain. CONFIRMED is plausible,
  not expected.
- **0.5B:** PARTIAL or FAILED (collapsed answers, lower Oracle-D).

## 6. Controls
- `s`-recovery and reconstruction.
- EVAL2 contexts are disjoint from CAL/EVAL (index ranges), checked in code.
- Oracle-D soundness on EVAL2.
- Library fp16 vs fieldrun int8 agreement on EVAL2 at 3B (reported, not a gate).
- Unit tests for bin assignment, ties, zero norm, and the quantile rank on synthetic data.
- pil `ruff` + `pytest`.

## 7. Risks / open
- **Small confirmation set** (150). CIs on coverage and violation rate are reported (Wilson 95%).
- **Bin edges are fitted on CAL in development and on all 450 seen decisions for confirmation**, then held fixed
  on EVAL2. Using the same data for edges and quantiles, selecting the candidate on EVAL, and checking every
  layer preclude a formal conformal guarantee. Distribution shift between bAbI tasks is an additional risk.
- **Compute:** 3B EVAL2 is about 4.5 CPU-hours at #130's measured 106 s per decision; 0.5B is about 6 minutes.
- **7B** stays deferred (no RAM here). A CONFIRMED 3B result would make it the obvious next rung.

## 8. Scope fences
Measurement only. The menu (C0–C2) and the selection rule are closed. Any further calibrator idea needs a new
prereg and fresh data.
