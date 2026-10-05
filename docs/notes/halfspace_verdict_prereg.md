# Per-context half-space verdicts and certified neighbourhoods — pre-registration

**Status: written 2026-10-05, committed and pushed before any code for this study existed.**

**Context.**
- jascal/pil#139 / #140: ball certificates centred on the code point `x(σ)` cover nothing, even with the tight
  directional radius `ρ_dir`. Yet clean-up is exact on 80–82% of SVO `d_F=32` contexts.
- jascal/i-orca#32: the exact clean-up condition is a set of half-spaces. Role `s` snaps to `σ(s)` **iff**
  `⟨c_s + M_s n, d_a⟩ < ‖d_a‖²/2` for every rival filler `a` (`nearest_iff_halfspace`), where `n = u − x(σ)`,
  `d_a = f_a − f_σ(s)`, `M_s n = unbind(P n, w_s)`, and `c_s = 0` for dual readouts.
- jascal/pil#135 / i-orca#28: the host decides the code point's decision `t` **iff**
  `⟨u, U_t − U_v⟩ > 0` for every `v ≠ t` (bias-free; the pairwise iff with the roles of `r` and `r̂` swapped).

**The questions.**
1. **Verification.** Does a per-context Soufflé verdict built from these iffs certify exactly the contexts where
   clean-up is exact and the host agrees with the code point? It must, by the theorems; this run checks that the
   implementation does.
2. **Neighbourhoods (the new measurement).** Around each certified **observed** residual `u₀`, how large is the ball
   in which every residual is still certified?
   - Clean-up slack: `δ_c(u₀) = min_{s,a} (‖d_a‖²/2 − ⟨M_s n₀, d_a⟩) / ‖M_sᵀ d_a‖` (`directional_snap` with the
     offset `c = M_s n₀`).
   - Host slack: `δ_h(u₀) = min_{v≠t} ⟨u₀, U_t − U_v⟩ / ‖U_t − U_v‖` (`agreement_ball` centred on `u₀`).
   - Local radius: `ρ_loc(u₀) = min(δ_c, δ_h)`. Every residual within `ρ_loc` of `u₀` gets exact clean-up and the
     code point's decision.
   - This is the first certificate in this line that covers residuals beyond the evaluated points.

**Kernel backing.** Every step is kernel-checked: `nearest_iff_halfspace`, `directional_snap`,
`role_readout_decomp`, `agreement_ball`, `substitution_pairwise_iff`. The **composed** local statement ("every
residual within `ρ_loc(u₀)` …") is not yet a single i-orca theorem. **It will be added to i-orca and merged before the
real run**; if that proof fails, the neighbourhood half of this study is withdrawn by addendum.

## 1. Model, tasks, substitutes, data

- #139's setup, reusing `experiments/cleanup_certificate.py` unchanged: GPT-2 small's decode input; LIST and SVO; TPR
  in LIST `d_F=8`, SVO `d_F=8`, SVO `d_F=32`; objectives `mse`, `cert`, `t6`; `P = W⁺`; dual role readouts; `F_s` from
  train.
- **Fresh seeds:** stimulus **61**, split **62**, fit seeds **{0, 1, 2}**. 12,000 contexts per task, split 60 / 10 / 30.
- **The run is wrapped in `systemd-inhibit --what=sleep:idle`** (#140's first run was stalled by a system suspend).

## 2. Verdicts (Soufflé over fixed-point facts, scale 2³⁰, slack τ = 10⁻⁶; Python twin)

- **`clean(c)`:** for every bound role `s` and rival `a ∈ F_s \ {σ(s)}`, the fact `slack(c, s, a)` =
  `‖d_a‖²/2 − ⟨M_s n, d_a⟩`, rounded down, exceeds τ.
- **`agree(c)`:** `hmin(c)` = `min_{v≠t} ⟨u, U_t − U_v⟩` over all 50,257 tokens (float64; rounded down) exceeds τ,
  where `t` is the code point's decision.
- **`full(c) = clean(c) ∧ agree(c)`.**

**Soundness and iff checks (any failure aborts the run).**
- **(i) The iff, implemented:** `clean(c)` equals actual clean-up exactness, and `agree(c)` equals
  `argmax(u) = argmax(x(σ))`, on every context, except contexts whose deciding slack lies within 10⁻⁶ of 0. Those
  are counted and reported as ties.
- **(ii) Neighbourhoods, sampled:** for every `full` context, 8 random directions at radius `0.999·ρ_loc`, plus the
  two worst-case directions (the minimising rival's `q` and the minimising host difference). All must keep exact
  clean-up and the code point's decision.

## 3. Metrics (test split; seed mean and range per cell)

- `E` (exact clean-up), `A` (host agrees with the code point), and `F_cov` (the `full` coverage).
- Among `full` contexts, medians of:
  - `ρ_loc`, `δ_c`, `δ_h`;
  - `ρ_loc / ‖n‖` (the neighbourhood relative to the residual's distance from its code point);
  - `ρ_loc / ρ_dir(σ)` (relative to the code-centred radius).
- **Bottleneck share:** the fraction of `full` contexts with `δ_h < δ_c` (the host's own decision boundary is the
  closer one).

## 4. Decision rules (fixed now)

- **H1 (verification).** Over all 27 fits, `clean ≡ exact` and `agree ≡ (argmax(u) = argmax(x))`, ties excepted.
  Passes iff no mismatch. A mismatch is a bug: the run aborts (§2) and nothing is reported as a result.
- **H2 (per-context certificates are substantial).** `F_cov ≥ 0.5` (seed mean) in at least one cell.
- **H3 (the neighbourhoods are not negligible).** In every cell with seed-mean `F_cov ≥ 0.05`, the median of
  `ρ_loc / ‖n‖` among `full` contexts is **≥ 0.01**. If no cell has `F_cov ≥ 0.05`, H3 is **untestable**.
- **Bottleneck label (descriptive), per cell:** **host-bound** if the bottleneck share is ≥ 0.8; **clean-up-bound**
  if it is ≤ 0.2; **mixed** otherwise.

**No expected outcome is stated.** The E and agreement rates from #139/#140 are prior, published context.

## 5. Not claimed

- Per-context certificates need the observed residual `u₀`. The neighbourhoods certify perturbations of observed
  residuals (robustness), not unseen contexts as such. Whether an unseen context's residual lands in some certified
  neighbourhood is a separate question.
- GPT-2 small, two templated tasks, the TPR family only; LIST `d_F=32` is out of scope.

## 6. Protocol and artifacts

- The i-orca composed theorem is merged before the real run (see Kernel backing).
- Script `experiments/halfspace_verdict.py`, frozen before the real run. `--smoke`: stimulus 999, 600 contexts,
  30 steps; smoke numbers are not results.
- Any change after smoke goes in a dated addendum, before the real run, disclosing what the smoke run showed.
- Outcome: `docs/notes/halfspace_verdict_outcome.md` plus the summary JSON. Neither is edited after the first real run.

---

## Addendum A (2026-10-05, after a smoke run, before any real run): disclosure only, no design change

- **What was run:** the `--smoke` pipeline on stimulus seed 999 (600 contexts, 30 steps), for all 27 fits, under
  `systemd-inhibit`. Seeds 61/62 are untouched.
- **Result:** it completed, and the iff checks (§2 (i)) held on every fit, with 0 mismatches and 0 ties.
  - E = 0 and `F_cov` = 0 everywhere, since 30-step fits are undertrained (as in #139/#140). So the neighbourhood
    checks (§2 (ii)) were not exercised on smoke data; the unit tests exercise them, including tightness.
  - Host agreement `A` was 0–0.39 per fit, and the `agree` verdict matched it exactly.
- **The design is unchanged. No expected outcome is stated.**
