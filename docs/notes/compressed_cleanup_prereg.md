# Compressed binding (T4(c)): clean-up certificates without a left inverse — pre-registration

**Status: written 2026-10-05, committed and pushed before any code for this study existed.**

**Context.**
- Every clean-up study since jascal/pil#139 excluded **LIST `d_F=32`**: its 960-dimensional tensor maps into 768
  residual dimensions, so `W` has no left inverse.
- jascal/i-orca#36 (`PIC_Cleanup.thy`, kernel-checked; **open, awaiting the maintainer's merge**) proves T4(c) in
  decoder form.
  - For any linear decoder `P`, `P(u − b₀) = T(σ) + E(σ) + P·n`, where `E(σ) = (P·W − I)T(σ)` is a fixed per-σ
    compression error.
  - If `‖E(σ)‖ ≤ ε‖T(σ)‖`, every half-space slack loses at most `ε‖T(σ)‖‖w_s‖‖d_a‖`
    (`compression_offset_bound`).
  - That gives the degraded radius `ρ_ε = min_{s,a} (‖d_a‖²/2 − ε‖T‖‖w_s‖‖d_a‖)/‖M_sᵀd_a‖`
    (`compressed_cleanup_certified`). With dual readouts, `c_s = 0`.
- Per context, `E(σ)` is computable, so exact clean-up is still decided **exactly** by the half-space slacks of the
  decoded tensor (`nearest_iff_halfspace`, via `role_cleanup_offset`). This needs no `ε`.

**The question.** Without a left inverse, do per-context certificates still cover a meaningful share of LIST
`d_F=32` contexts? How large is the compression error `ε`, and how much does it shrink the clean-up radius?

## 1. Model, tasks, substitutes, data

- GPT-2 small, decode input, LIST and SVO as in #136 (`certified_substitutes.py`, unchanged; the 12,000-context
  dump).
- **Settings:**
  - **LIST `d_F=32`** (compressed: 960 → 768);
  - **SVO `d_F=32`** (a control: 96 ≤ 768, so `W` is injective and the ridge decoder's `ε` is small but not zero).
- Objectives `mse`, `cert`, `t6` (#136/#139 code, unchanged); fit seeds **{0, 1, 2}**. That is 2 × 3 × 3 = **18 fits**.
- **Fresh seeds:** stimulus **111**, split **112**; split 60 / 10 / 30 by context, dropping test contexts with an unseen
  (filler, role) pair, as before.

## 2. Decoder and quantities

- **Decoder:** ridge `P = (WᵀW + λI)⁻¹Wᵀ`, with **`λ = 10⁻³·σ_max(W)²`** (fixed now).
- **Dual role readouts** `w_s` from `er⁻¹`, as in #139 (flagged no-dual-readout if `cond(er) > 10⁸`).
- **Compression error:** `ε(σ) = ‖(P·W − I)T(σ)‖/‖T(σ)‖` per context. Reported: `ε_train` = the max over train
  contexts, plus the median and maximum over test contexts.
- **Exact per-context clean-up slacks:** `slack(c, s, a) = ‖d_a‖²/2 − ⟨g_s − f_σ(s), d_a⟩`, with
  `g_s = unbind(P(u − b₀), w_s)` and `d_a = f_a − f_σ(s)`. `F_s` and the rivals are as in #141.
- **Host agreement:** `hmin(c) = min_{v≠t} ⟨u, U_t − U_v⟩`, with `t` the code point's decision.
- **Degraded radius per context**, at the context's own `ε(σ)`:
  `ρ_ε(σ) = min_{s,a} (‖d_a‖²/2 − ε(σ)‖T(σ)‖‖w_s‖‖d_a‖)/‖M_sᵀd_a‖`. Also the uncompressed directional radius
  `ρ_dir(σ)` for the same `P` (that is, `ε` set to 0).

## 3. Verdicts

- `clean(c)`: every slack > τ. `agree(c)`: `hmin(c) > τ`. `full = clean ∧ agree`.
- These are Soufflé queries over fixed-point facts (scale 2³⁰, τ = 10⁻⁶), using `halfspace_verdict.verdicts_souffle`
  unchanged, with a Python twin.

**Aborts (any failure stops the run):**
- **(i) the iff:** `clean` must equal actual clean-up exactness (run clean-up through `P`), and `agree` must equal
  host agreement, ties within 10⁻⁶ excepted (counted);
- **(ii) the offset bound:** `|⟨unbind(E(σ), w_s), d_a⟩| ≤ ε(σ)‖T(σ)‖‖w_s‖‖d_a‖·(1 + 10⁻⁹)` for every checked
  (context, role, rival) (`compression_offset_bound`);
- **(iii)** `ρ_ε(σ) ≤ ρ_dir(σ)` everywhere.

## 4. Metrics (seed mean and range per cell)

- **Primary:** per-context `F_cov` (the `full` certified share of test contexts).
- **Secondary:**
  - `E` (exact clean-up) and `A` (host agreement);
  - `ε_train`, and the median and max test `ε(σ)`;
  - the **radius shrink**: the median over test contexts of `ρ_ε/ρ_dir`;
  - the share of contexts with `ρ_ε ≤ 0` (compression alone exhausts the slack);
  - the code-centred ball coverage `‖n‖ < min(ρ_ε, β)` (expected to be tiny, as in #139/#140, but reported).

## 5. Decision rule (fixed now)

- **H1:** `F_cov ≥ 0.05` (seed mean) in at least one LIST `d_F=32` cell.
- **No expected outcome is stated.**

## 6. Not claimed

- The certificate is per context and needs the observed residual. Coverage is `empirical`.
- One decoder (ridge with a fixed `λ`). Other decoders could do better.
- GPT-2 small, two templated tasks, the TPR family only.

## 7. Protocol and artifacts

- Script `experiments/compressed_cleanup.py`, frozen before the real run. `--smoke`: stimulus 999, 600 contexts,
  30 steps; smoke numbers are not results.
- The run is wrapped in `systemd-inhibit --what=sleep:idle`.
- Any change after smoke goes in a dated addendum, before the real run.
- Outcome: `docs/notes/compressed_cleanup_outcome.md`, the summary JSON, and a table test. The outcome is edited only
  through a labelled post-review section.

---

## Addendum A (2026-10-05, after a smoke run, before any real run): disclosure only, no design change

- **What was run:** the `--smoke` pipeline (stimulus 999, 600 contexts, 30 steps) for all 18 fits, under
  `systemd-inhibit`. Seeds 111/112 are untouched.
- **Result:** it completed. None of the three aborts fired (iff, offset bound, `ρ_ε ≤ ρ_dir`), and there were 0 ties.
- **Disclosed smoke values** (undertrained fits, non-study data):
  - LIST `d_F=32`: `ε` ≈ 0.55–0.77, and `ρ_ε ≤ 0` for every context (the compression error alone exhausts the slack);
  - SVO `d_F=32`: `ε` ≈ 0.03–0.07 (from the ridge `λ`), and the median `ρ_ε/ρ_dir` ≈ 0.5–0.8;
  - `F_cov` = 0 everywhere.
- **The design is unchanged. No expected outcome is stated;** 30-step fits do not predict trained ones.
