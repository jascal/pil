# Restricted-Decision Early Exit at Scale, with a Directional Bound — Pre-Registration

**Status: SIGNED (approved as-is) — 2026-09-25. Decision rules fixed BEFORE numbers. No 3B/7B dump had been generated, and
no directional quantity has been computed on any model.** A follow-up to
[`restricted_early_exit_outcome.md`](./restricted_early_exit_outcome.md) (#129, HALTED on Qwen2.5-0.5B). That
outcome named two remaining routes, and this prereg tests both:
1. **A model that decides more confidently** (a scale ladder: 0.5B → 3B → 7B), in case larger models have larger
   restricted leads.
2. **A directional bound.** With `K = 6` options, the certificate needs only how far the remaining layers push
   *against each of the 5 rival directions*, not the full norm of what they write.

## 1. Two arms
Notation as #126/#129: `y_k` = the prefix residual after layer `k` (in `y = s·x` coordinates), `suffix_k =
y_final − y_k`, `w_v = θ_f ⊙ U_v` over the option tokens `O`, `t = t_k` = the restricted prefix argmax, and the
certified radius `R_k = min_{v∈O∖t} (⟨y_k, w_t − w_v⟩ / ‖w_t − w_v‖)`.
- **Arm A — norm bounds** (unchanged from #129): certify iff `B < R_k`, with `B ≥ ‖suffix_k‖` (oracle /
  weight-derived / calibrated).
- **Arm D — directional.** Let `u_v = (w_t − w_v)/‖w_t − w_v‖` and the **adverse push**
  `a_k = max_{v∈O∖t} ⟨−suffix_k, u_v⟩`, the most the suffix moves toward any rival along that rival's
  separating direction. Then `t_k` is final if `R_k > a_k` (sufficient, since
  `gap_v + ⟨suffix, w_t − w_v⟩ ≥ ‖w_t − w_v‖ (R_k − a_k)`).
  - **Oracle-D** uses the actual `a_k`. This is the ceiling for any directional bound.
  - **Calibrated-D** uses `B_D(k) = q̂^D_k · ‖y_k‖`, where `q̂^D_k` is the conformal `(1−α)` quantile over CAL of
    `a_k / ‖y_k‖` (`α = 0.05`, rank `⌈(n+1)(1−α)⌉`). The quantile may be negative, meaning the suffix usually helps
    `t`. It is not clipped, and the violation rate measures the consequence.

  Arm D can **never** be a certificate: there is no sound input-independent bound on a projection of the suffix.
  Its best outcome is a statistical guarantee, tagged `empirical`.

## 2. PINS
- **PIN A — models.** fieldrun bundles `Qwen2.5-0.5B-Instruct` (the #129 dumps, reused), `Qwen2.5-3B-Instruct`
  (36 layers, d = 2048, tied unembedding) and `Qwen2.5-7B-Instruct` (28 layers, d = 3584, **untied**: the read-out
  is `lm_head`). All int8, as fieldrun runs them.
- **PIN B — decisions.** The #129 decision prompts, unchanged: parallel-decisions @ `45820320`, field `location`,
  6 options, bAbI qa1–qa3, CAL 150 / EVAL 300, canonical (alphabetical) order. **Control:** the Qwen2.5-3B and
  -7B tokenizers must encode every prompt to the same ids as the 0.5B tokenizer, and the option first tokens must
  compile identically. If not, rebuild per model with that model's tokenizer and state it.
- **PIN C — capture.** fieldrun `--source-dump --tail 1` (master after #135) on each model. **No option-order
  reruns** at 3B/7B: the order-bias read was descriptive, and at 7B it would add about 6 CPU-hours.
- **PIN D — harness.** pil's bundle reader gains `rowi8` (per-row int8, scale per row). The read-out uses `lm_head`
  when present, otherwise `embed`. `s`-recovery uses the dequantized embedding row (fieldrun's `rows_f32` path).
  Everything else in `pil/early_exit.py` is per-bundle already.
- **PIN E — weight-derived bound.** Computed per model from its bundle with the #126 formula (Addendum A
  activation factors), and **committed before any 3B/7B dump is generated**.
- **PIN F — cost.** `K·d / layer` per restricted check, per model. P-every (check after every layer). Net saving =
  layers skipped − checks × cost.

## 3. Decision rules (FIXED BEFORE NUMBERS), applied **per model**
**Arm A** (the #129 rules):
- **FIRES:** weight-derived bound ≥ 10% EVAL coverage with ≥ 2 layers skipped.
- **FIRES-EMPIRICAL:** calibrated bound ≥ 25%, violation rate ≤ 2α of certified, and mean net saving ≥ 1 layer.
- **IN-BETWEEN:** oracle ≥ 25%.
- **HALTED:** oracle < 25%.

**Arm D:**
- **D-FIRES:** Calibrated-D coverage ≥ 25% of EVAL, violation rate ≤ 2α (≤ 10%) of certified, **and** mean net
  saving (P-every) ≥ 1 layer. → A statistically guaranteed restricted early exit, worth building.
- **D-IN-BETWEEN:** Oracle-D ≥ 25% but Calibrated-D misses a bar. → Direction is enough in principle; the
  calibration is the bottleneck.
- **D-HALTED:** Oracle-D < 25%.

**Soundness gate:** Arm A's oracle and weight bounds must have 0 violations, and Oracle-D must have 0 violations
(it is exact given `a_k`). Otherwise the model's run is void.

**Scale trend (descriptive):** per model, report
- median `‖suffix_k‖ / R_k` and median `a_k / R_k` at the last three exits;
- restricted accuracy vs gold;
- uncertified headroom `k★`.

A trend is only *stated* as shrinking or growing if it is monotone across all three models.

## 4. Architect's predictions (recorded, not decision variables)
- **Arm A stays HALTED** at 3B and 7B, but `‖suffix‖/R` at the last exit shrinks with scale (0.5B: about 56×). I
  can't say whether it gets below 1.
- **Arm D does much better.** Oracle-D ≥ 25% on every model, since the final decision agrees with `t_k` for 64% of
  0.5B decisions at `k = 22`, and the adverse push is a small slice of the suffix norm.
- **Calibrated-D:** D-FIRES is plausible at 3B/7B. At 0.5B, **D-IN-BETWEEN** (low accuracy means near-ties).
- Accuracy rises strongly with scale (qa1 > 80% at 7B).
- Headroom grows with scale in layers skippable, though Qwen's convergence depth did not fall with scale in
  `pic` §8.

## 5. Controls
- `s`-recovery residual and fieldrun reconstruction (per model).
- The tokenizer-identity control (PIN B).
- No fitted parameter besides `q̂` (CAL-only). EVAL read once.
- **Library agreement at 3B (reported, not a gate):** parallel-decisions' Torch engine on `Qwen/Qwen2.5-3B-Instruct`
  vs fieldrun int8's restricted decision on EVAL. At 7B this is **not run**: fp16 7B (~15 GB) does not fit this
  machine's 14 GB RAM.
- A unit test that Arm D certifies soundly given the exact `a_k` (synthetic). pil `ruff` + `pytest`.

## 6. Risks / open
- **Multiple looks** under P-every (per-`k` conformal), as in #126.
- **Arm D's guarantee is marginal:** exchangeability of CAL and EVAL across three bAbI tasks.
- **Qwen-only ladder** (Pythia's decreasing convergence depth is the other family in `pic` §8). One task.
- **CPU time:** about 2 h (3B) and 4.5 h (7B) of dumping.

## 7. Scope fences
Measurement only. No fieldrun change (the #135 capture suffices). No new decision prompts.
