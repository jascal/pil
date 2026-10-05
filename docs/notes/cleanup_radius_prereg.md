# How much exact clean-up does the directional radius certify? — pre-registration

**Status: written 2026-10-05, committed and pushed before any code for this study existed.**

**Context.**
- jascal/pil#139: the T6(b) clean-up certificate covered no context. Both radii bound everywhere, yet clean-up
  recovered the exact structure on 81–84% of SVO `d_F=32` contexts (`mse`, `t6`) at `‖n‖ ≈ 80ρ`. The worst-case
  clean-up radius `ρ = min_s γ_s/(2K‖w_s‖)`, with `K = 1/σ_min(W)`, is therefore loose.
- jascal/i-orca#32 (`PIC_Cleanup.thy`, kernel-checked) proves the exact one. The clean-up region is an
  intersection of half-spaces, one per (role `s`, rival filler `a`): `⟨M_s n, d_a⟩ < ‖d_a‖²/2` (dual readouts, so
  no crosstalk). Here `d_a = f_a − f_σ(s)` and `M_s n = unbind(P n, w_s)`. Its inradius is
  **`ρ_dir(σ) = min_{s∈D(σ)} min_{a≠σ(s)} (‖d_a‖²/2) / ‖M_sᵀ d_a‖`**.
  - It is exact per half-space (`directional_radius_tight`) and never below `ρ` (`worst_case_implies_directional`).
  - `‖n‖ < ρ_dir` ⟹ clean-up returns `σ` exactly (`role_cleanup_directional`, `cleanup_certified_directional(1)`).

**The question.** Of the contexts where clean-up is exact, what share does `ρ_dir` certify? How much larger is
`ρ_dir` than `ρ`?

## 1. Model, tasks, substitutes, data

- Exactly #139's setup: GPT-2 small's decode input; LIST and SVO; the TPR family in the three settings with a left
  inverse (LIST `d_F=8`, SVO `d_F=8`, SVO `d_F=32`); objectives `mse`, `cert`, `t6`; `P = W⁺`; dual role readouts;
  `F_s` from train.
- Reused **unchanged** from `experiments/cleanup_certificate.py`: data handling, training, `readout`, `cleanup`,
  `beta_and_decision`, the flags.
- **Fresh seeds:** stimulus **51**, split **52**, fit seeds **{0, 1, 2}**. 12,000 contexts per task, split 60 / 10 / 30.

## 2. The directional radius, computed per test context

- `P` reshaped to `(d_F, n_role, 768)`; `M_s = Σ_j (w_s)_j · P[:, j, :]`, a `d_F × 768` matrix.
- For each role `s` and each pair of distinct fillers `b, a ∈ F_s`: `r_s(b, a) = (‖f_a − f_b‖²/2) / ‖M_sᵀ (f_a − f_b)‖`.
  It depends only on `(s, b, a)`, so it is tabulated once per fit.
- `ρ_dir(σ) = min_{s∈D(σ)} min_{a∈F_s, a≠σ(s)} r_s(σ(s), a)`.
- `ρ` (worst case) and `β` exactly as in #139.

**Verdicts** are Soufflé queries over fixed-point facts (scale 2³⁰; `‖n‖` rounded up, radii rounded down; slack
10⁻⁶), with a Python twin:
- **`dir`**: `‖n‖ < ρ_dir` (clean-up certified);
- **`wc`**: `‖n‖ < ρ` (clean-up certified, worst case);
- **`full`**: `‖n‖ < min(ρ_dir, β)` (the full directional T6(b) certificate).

**Soundness checks (any failure aborts the run).**
- Every `dir`- or `wc`-certified context must have exact clean-up.
- Every `full`-certified context must also agree with the host.
- `ρ_dir ≥ ρ·(1 − 10⁻⁹)` on every context (the theorem).

## 3. Metrics (test split; seed mean and range per cell)

- **E**: the exact clean-up rate (clean-up actually run).
- **C_dir** and **C_wc**: the shares of all test contexts certified by `dir` and `wc`.
- **Primary: S = C_dir / E**, the share of exact recoveries that `ρ_dir` certifies. Undefined when E = 0.
- **Gain**: the median over test contexts of `ρ_dir / ρ`.
- **Secondary:** `full` coverage; the share of uncertified contexts with `‖n‖ ≥ ρ_dir` and with `‖n‖ ≥ β`.

## 4. Decision rules (fixed now)

- **H1: the directional radius explains exact clean-up.** Among cells with seed-mean **E ≥ 0.2**, H1 passes if
  seed-mean **S ≥ 0.5** in at least half of them. If no cell has E ≥ 0.2, H1 is **untestable** (reported, not
  failed).
- **H2: the tightening is substantial.** For each setting, take the median over its three objective cells of the
  seed-mean gain. H2 passes if that is **≥ 10** in ≥ 2 of 3 settings.
- **Flagged fits** (no left inverse, no dual readout) count as C_dir = 0 and are reported, never re-fitted.

**No expected outcome is stated.** #139's E values are prior, published context, not a prediction of S.

## 5. Not claimed

- A `dir` certificate says clean-up recovers `σ` for every residual in the ball. It says nothing about the host's
  decision; that needs `β` (the `full` certificate).
- `ρ_dir` is tight per half-space. Exact recovery outside the ball is possible, because the ball is the largest one
  centred on `x(σ)`, not the whole region.
- GPT-2 small, two templated tasks, the TPR family only.

## 6. Protocol and artifacts

- Script `experiments/cleanup_radius.py`, committed (frozen) before the real run.
- `--smoke`: stimulus seed 999, 600 contexts, 30 steps. Smoke numbers are not results.
- Any change after a smoke run goes in a dated addendum, before the real run, disclosing what the smoke run showed.
- Outcome: `docs/notes/cleanup_radius_outcome.md` plus the summary JSON. Neither is edited after the first real run.
