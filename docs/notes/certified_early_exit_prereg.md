# Certified Early Exit — Pre-Registration

**Status: DRAFT — awaiting signature. Decision rules are fixed BEFORE numbers. No source dump for this probe has
been generated or read.** Successor to fieldrun `experiments/certified_prune_step0` (#107), whose verdict names
this as the surviving role of `PIC_Prune`: *"a certified early-exit / decode-attribution probe (the late-third that
is skippable)"*. Motivated by the external PIC review (pairwise-margin certificates with coverage, checking cost and
runtime savings measured separately).

## 1. The move
Step 0 certified dropping **arbitrary** block sets per token. That certificate is **direct-effect only**: dropping a
mid-stack block changes every later block's input, and Step 0's "0 flips" is checked on the DLA sums, not by
re-running the model. Its droppable blocks were also mostly early, so they are not skippable compute. It
checked only the top-K candidates.

**Early exit is the one coalition shape where all three problems disappear.** Stop after layer `k`: the prefix
residual `x_k` is *exactly* what the model computed (causally exact, no direct-effect caveat); the skipped suffix is
real skipped compute; and the check can run over the **full vocabulary** with one unembedding pass.

## 2. The certificate (Qwen2.5 = RMSNorm, no final bias, tied unembedding)
Final logits are `⟨θ_f ⊙ x/rms(x), U_v⟩`, and `rms(x) > 0` is common to all tokens, so the decision is
`argmax_v ⟨x, w_v⟩` with `w_v = θ_f ⊙ U_v`. At exit layer `k`, let `t_k = argmax_v ⟨x_k, w_v⟩` (full vocabulary) and
let `B ≥ ‖x_final − x_k‖` bound the norm of everything layers `> k` will write. By Cauchy–Schwarz,
`t_k` is the final decision if

> `∀ v ≠ t_k:  ⟨x_k, w_{t_k} − w_v⟩ > B · ‖w_{t_k} − w_v‖.`

This is the pairwise form of the margin certificate (`PIC_Prune` / `PIC_Quant`): it uses each rival's own distance
`‖w_t − w_v‖` instead of one uniform `2δ`. Soundness given `B` is classical (not yet kernel-checked). A small
i-orca theorem is a follow-up, **not** part of this gate.

## 3. PINS (fixed a-priori)
- **PIN A — model.** `fieldrun/bundles/Qwen2.5-0.5B-Instruct` (int8 weights with per-column scales, f16 norms and
  embedding) — the exact weights fieldrun runs; `n_ℓ = 24`, `d = 896`, `V = 151936`. The fieldrun commit that
  produces the dumps is recorded in the outcome.
- **PIN B — data and splits.** `fieldrun --source-dump --texts`. Units: prose = `data/wikitext2_train.txt`
  paragraphs with ≥ 400 characters; code = `data/code_train.txt` chunks of 40 consecutive lines with ≥ 400
  characters. Each unit is truncated to its first 600 characters. Eligible units are shuffled with seed 0.
  **CAL** = the first 20 prose + first 20 code units. **EVAL-PROSE** = the next 20 prose units. **EVAL-CODE** =
  the next 20 code units. At most 16 positions per unit (`--n 16`). Splits are disjoint by construction.
- **PIN C — coordinates.** Dumps store `d̃_j = s · θ_f ⊙ d_j`, with `s = 1/rms(x_final)` per position. Work in
  `y_j = d̃_j / θ_f = s · d_j` (the certificate is invariant to the common factor `s > 0`). Recover `s` from the
  embedding block, whose raw write is the embedding row: `s = ⟨d̃_embed, θ_f ⊙ E[cur]⟩ / ‖θ_f ⊙ E[cur]‖²`.
  **Self-test:** relative residual of that fit `< 1e-3` on every position, else abort (wrong bundle or fold).
- **PIN D — exit points.** `k ∈ {0,…,22}` = after layer `k`'s MLP (prefix = embed + blocks of layers `≤ k`); `k = 23`
  is the full model. `k_c(B)` = the earliest certified `k`; if none, the position runs the full model.
- **PIN E — the three bounds on `‖suffix_k‖`** (in `y` coordinates):
  1. **Oracle** `B_or(k) = ‖Σ_{j > k} y_j‖` — the actual suffix. Not available at run time; the coverage ceiling
     for *any* norm-only bound.
  2. **Weight-derived** `B_w(k) = s · Σ_{ℓ > k} (A_ℓ + M_ℓ)`, computed from the dequantized bundle before any dump is
     read. With `ĥ = max|θ_in|·√d` (RMSNorm output bound):
     - attention: `A_ℓ = ‖W_O‖₂ · sqrt(Σ_{q=1}^{14} (‖W_V^{g(q)}‖₂ ĥ + ‖b_V^{g(q)}‖)²)`. Each query head `q`'s
       output is a convex combination of the value vectors of its KV group `g(q)` (grouped-query attention, 2 KV
       heads for 14 query heads). RoPE does not touch values, and `o_proj` has no bias;
     - MLP (SwiGLU; `|silu(z)| ≤ |z|`): `M_ℓ = ‖W_down‖₂ · max_i‖W_gate[:,i]‖ · ĥ_post · ‖W_up‖₂ · ĥ_post`.

     Sound and input-independent; expected to be very loose.
  3. **Calibrated** `B_cal(k) = q̂_k · ‖y_k‖`, where `q̂_k` is the conformal `(1−α)` quantile over CAL positions of
     `ρ_k = ‖suffix_k‖ / ‖y_k‖` (scale-free, so available at run time), with `α = 0.05` and the finite-sample rank
     `⌈(n+1)(1−α)⌉`. A statistical guarantee under exchangeability, **not** a certificate — tagged `empirical`.
- **PIN F — cost model.** One full-vocabulary check ≈ `V·d` = 136.1M multiply-adds. One layer ≈
  `d(d + 2·128 + d) + 3·d·4864` ≈ 14.9M (attention projections + MLP; attention-score cost ignored, which
  favours the check). So **one check ≈ 9.1 layer-equivalents** (`c_chk`). Policies:
  - **P-every**: check after every layer until certified; cost `(#checks)·c_chk`.
  - **P-single(k₀)**: one check at a single layer `k₀`, chosen on **CAL** to maximize mean net saving.
  - Net saving per position (layer-equivalents) = `layers skipped − (#checks)·c_chk`, where
    `layers skipped = 23 − k_c` if certified at `k_c`, else 0.

## 4. Metrics (per EVAL split × bound)
Coverage (fraction certified at some `k ≤ 22`); distribution of `k_c`; mean layers skipped; net saving under
P-every and P-single; **violations** = certified positions whose `t_{k_c}` ≠ the model's decision. Uncertified
reference: `k★` = the earliest `k` after which the prefix argmax stays equal to the final decision; report how much
of the `k★` headroom each bound captures.

## 5. Pre-registered decision rules (FIXED BEFORE NUMBERS)
- **FIRES (certified)**: `B_w` certifies ≥ 10% of positions on either EVAL split with ≥ 2 layers skipped. → A sound
  early exit exists on this model: next steps are the i-orca theorem and a fieldrun engine path.
- **FIRES-EMPIRICAL**: `B_w` misses, but `B_cal` reaches coverage ≥ 25% on either EVAL split, with observed
  violation rate ≤ 2α there, **and** P-single net saving > 0. → A statistically guaranteed early exit; the
  certificate needs a tighter sound bound. Tag `empirical`.
- **IN-BETWEEN**: `B_or` coverage ≥ 25% on either split, but neither `B_w` nor `B_cal` clears its bar. → The
  norm-only pairwise test has headroom; the bound, not the certificate form, is the bottleneck.
- **HALTED**: `B_or` coverage < 25% on both splits. → Even perfect knowledge of the suffix *norm* rarely certifies;
  early exit needs directional information about the suffix (for example the averaged Jacobians from
  `--jlens-export`). Norm-bounded early exit is dead on this model.
- **SOUNDNESS GATE (any outcome)**: violations must be **0** for `B_or` and `B_w`. A nonzero count is a bug and
  voids the run.

## 6. Architect's predictions (recorded, not decision variables)
- `B_w` is vacuous (coverage < 1% at every `k`): products of operator norms over up to 23 layers.
- `B_or` certifies mostly in the last few layers. Code covers more than prose, as in Step 0, where code margins
  were about 4× larger.
- `B_cal` sits between the two.
- Net saving is negative under P-every (each check costs about 9 layers) and small at best under P-single.
- Most likely outcome: **IN-BETWEEN**. The main reading will be the gap between the oracle and the other two bounds.

## 7. Controls
- The `s`-recovery self-test (PIN C).
- Full-vocabulary reconstruction: `argmax_v ⟨Σ_j y_j, w_v⟩` = the model's decision on ≥ 99% of positions.
- CAL and EVAL disjoint.
- `q̂_k` and `k₀` are fit on CAL only, and EVAL is read once.
- `B_w` is computed and committed before any dump is read.
- pil `ruff` + `pytest` gate, including property tests that the certificate is sound on synthetic data with an
  exact suffix bound.

## 8. Risks / open
- **Check cost dominates at this scale.** The unembedding is large relative to a 0.5B layer (`c_chk ≈ 9.1`). At 7B
  the ratio falls to about 2.3, so a negative net saving here does not transfer upward. A cheap *sound* pre-screen
  that avoids the full-vocabulary pass is an open lever, not part of this gate.
- **Direction-agnostic.** A norm bound ignores which way the suffix points. HALTED would say directional
  information is required, not that early exit is impossible.
- **Multiple looks.** The conformal guarantee holds per exit layer `k`. P-every checks up to 23 layers per
  position, so the chance that *some* check is wrong can exceed `α`. The ≤ 2α violation bar is read against that,
  and P-single (one look) is the clean case.
- **Exchangeability.** `B_cal`'s guarantee assumes CAL and EVAL are exchangeable. Mixing prose and code in CAL,
  then evaluating each separately, tests how well it transfers.
- Single model, single seed, CPU.

## 9. Scope fences
Qwen2.5-0.5B-Instruct only, the bundle as served by fieldrun. RMSNorm path only (Pythia/LayerNorm needs the bias
term and is out of scope). Measurement only: no engine change, no retraining.
