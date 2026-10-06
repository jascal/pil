# Calibrated conditional certificates: a distribution-free bound on the false-certificate rate (Learn-then-Test) — pre-registration

**Status: written 2026-10-06, committed and pushed before any code for this study existed.** Follows #148.
- #148's word-blind gate issued the stable student's certificate at about 20% coverage with a **measured**
  false-certificate rate of 0.050.
- Its threshold τ was the lowest score reaching 0.95 empirical precision on 400 validation classes. That gives no
  guarantee, and coverage swung from 0.095 to 0.303 across seeds.
- This study replaces that rule with **Learn-then-Test** (LTT; Angelopoulos, Bates, Candès, Jordan, Lei, 2021) on a
  larger calibration split. The output is a statement of the form *"with probability ≥ 1 − δ over the calibration draw,
  the false-certificate rate among issued certificates is ≤ α."*

**Why LTT, not plain conformal risk control.** Conformal risk control bounds an expected loss, and the natural loss here
gives the **joint** rate P(issued ∧ wrong). The quantity that matters is the rate **among issued certificates**, a
precision, which is not of that form. LTT controls it with a per-threshold binomial test.

**No new probe.** The design draws on #148's results and #148's disclosed probe (gate precision about 0.93–0.96 at
coverage 0.25, about 0.89 at 0.50). **The thresholds below were set with both in view.**

## 1. Data

- **Same template, words, held-out set and student/gate recipes as #148.**
- **Fresh seeds:** stimulus **171**, split **172**. **6,000 σ** (up from 4,000), split in this order:
  - **train 2,400** (the same size as before, so the student and gate train exactly as in #148);
  - **calibration 1,800**;
  - **test 1,800**.
- **Fit and gate seeds {40, 41, 42, 43, 44}.**

## 2. The calibrated threshold (fixed now)

- **Risk:** for a threshold τ, `FC(τ)` is the population probability that GPT-2 is **not** constant over the 16 seen
  words, given gate ≥ τ and `cert16`.
- **Targets:** **α = 0.10** and **δ = 0.10**.
- **Grid, fixed independently of the calibration data:** τ_j is the train-split gate-score quantile at q_j = 0.95, 0.94,
  …, 0.01 (95 points, in descending τ). The grid uses only train classes, which never enter calibration or test.
- **Test per grid point (on calibration):**
  - let n_j be the calibration classes with gate ≥ τ_j and `cert16`, and w_j those among them that are not
    seen-constant;
  - the p-value is `p_j = P(Binomial(n_j, α) ≤ w_j)`, valid for H₀ʲ: FC(τ_j) ≥ α. Given n_j, the w_j are binomial
    because calibration classes are i.i.d. draws from the stimulus distribution;
  - if n_j = 0, p_j = 1.
- **Fixed-sequence testing:** go down the grid from the highest τ, rejecting while p_j ≤ δ, and stop at the first
  non-rejection.
  - The selected τ̂ is the **lowest rejected** τ_j. If the first point is not rejected, nothing is issued.
  - This controls the family-wise error at δ, so P(FC(τ̂) > α) ≤ δ (LTT, Theorem 1).
- **Issued certificate:** gate ≥ τ̂ **and** `cert16`, as in #148.
- **Reported, no hypothesis attached:** #148's empirical rule on the same calibration split (lowest τ with ≥ 0.95
  precision and ≥ 20 classes).

## 3. Metrics (test classes; as #148)

- coverage, `FCi`, faithfulness among issued, gate AUC;
- held-out-word constancy and agreement among issued (extrapolation, not certified);
- per seed: τ̂, the number of grid points rejected, and calibration n and w at τ̂.
- **Soundness checks (an abort):** as #148 (certified-class predictions, the exact k = 1 box, the gate's input
  identical across variants). In addition, every rejected grid point must have p_j ≤ δ.

## 4. Decision rules (set with #148's results in view)

- **H1 (the bound holds on test):** at most **1 of 5** seeds has test `FCi` > α = 0.10.
  - The guarantee allows each seed to fail with probability ≤ δ, and the test estimate adds its own noise.
  - A seed that issues nothing counts as **not** violating.
- **H2 (useful coverage under the guarantee):** seed-mean test coverage **≥ 0.20**.
- **H3 (stable coverage):** per-seed test coverage range (max − min) **≤ 0.10** (#148: 0.208).
- **Reported:** faithfulness among issued, held-out agreement among issued, and the empirical-rule comparison.

## 5. Not claimed; tags

- **The guarantee is the LTT theorem's.** It assumes calibration and test classes are exchangeable draws from the
  stimulus distribution (true by construction here: random σ, a random split).
  - It says nothing about other templates, other nuisance positions, or a shifted input distribution.
  - It is **not machine-checked**. A kernel-checked proof of the binomial fixed-sequence lemma in i-orca is `open`.
  - The per-seed outcomes are `empirical`.
- **The student's certificate stays `proved`** (a sound IBP, float32, sound up to rounding), and the gate stays
  invariant by construction.
- Held-out words are not certified.

## 6. Protocol and artifacts

- Script `experiments/ltt_certificate.py` and tests, frozen (commit and sha256) before the real run.
- **`--smoke`:** stimulus 999, 300 σ, 300 steps. Smoke numbers are not results.
- The run is wrapped in `systemd-inhibit --what=sleep:idle`, launched in the repo.
- Any change after smoke goes in a dated addendum, before the real run.
- **Outcome:** `docs/notes/ltt_certificate_outcome.md`, the summary JSON, and a table test. The outcome is edited only
  through a labelled post-review section.

---

## Addendum A (2026-10-06, after the smoke runs, before any real run): disclosure only, no design change

- **The smoke run** (stimulus 999, 300 σ, 300 steps) completed, and the soundness checks held.
  - **LTT issued nothing on any seed**, as expected: at 300 steps the student and gate are poor, and a 90-class
    calibration split cannot reject at the first grid point.
  - To exercise the issuing path end to end, the smoke was re-run with the test loosened **in-process only**
    (`ltt_tau` wrapped with α = 0.6, δ = 0.5; the script unchanged). The path ran on all 5 seeds.
- **Simulation check** (a unit test): scores ~ U(0, 1) with P(wrong | s) = 0.3(1 − s), 1,800 calibration classes, 400
  draws.
  - The selected τ's true false rate exceeded α in **8%** of draws (bound: δ = 10%).
  - The mean selected τ was 0.42, against the oracle 0.33. The rule is valid and not trivially conservative.
- **Script frozen at `da3976e`**, sha256 of `experiments/ltt_certificate.py` `0220780a827408c5…`. Check it with
  `git show da3976e:experiments/ltt_certificate.py | sha256sum`.
- Thresholds in §4 unchanged.
