# Learn-then-Test re-test: many independent calibration draws, scored on a large audit set — pre-registration

**Status: written 2026-10-06, committed and pushed before any code for this study existed.** A corrected re-test of
#149 (`ltt_certificate_*`). The pre-registered test there failed (4 of 5 seeds over α on test) for two design reasons.
- **One shared calibration/test split for all seeds.** The guarantee is over the calibration draw, so five seeds on
  one draw are one trial, not five.
  - #149's **post-hoc re-splits (exploratory, disclosed here as this study's probe)**: the calibration estimate was
    unbiased for test, with mean calibration 0.079–0.083 against mean test 0.079–0.084.
  - Swapping the two halves reversed the gap. Across 200–300 re-splits, the test *estimate* exceeded α in 12–17% of
    them, from a 1,800-class test split.
- **The fixed-sequence grid started at the extreme top** (the train-score quantile 0.95). One seed failed its first
  point and issued nothing.

**The question is unchanged:** does LTT deliver P(FC(τ̂) > α) ≤ δ for the gated student certificate? **The thresholds
below were set with #149's results and post-hoc analysis in view.**

## 1. Data

- **Same template, words, held-out set and student/gate recipes as #148/#149**, unchanged.
- **Fresh seeds:** stimulus **181**, split **182**. **14,400 σ**, split in this order:
  - **train 2,400**, as before;
  - a **calibration pool of 6,000**;
  - an **audit set of 6,000**.
- **Fit and gate seeds {50, 51, 52, 53, 54}.**

## 2. Procedure (fixed now)

- **Per seed:** train the student and gate (as #149), then make **R = 200 calibration draws**. Each draw is 1,800 classes
  sampled without replacement from the calibration pool, with the draw RNG seeded by (split seed, fit seed).
- **Per draw:**
  - run LTT exactly as #149 (α = 0.10, δ = 0.10, the binomial p-value, fixed-sequence testing), **except the grid
    starts at q = 0.80**. τ_j is the train-split gate-score quantile at q = 0.80, 0.79, …, 0.01 (80 points);
  - issue on the **audit set** with gate ≥ τ̂ and `cert16`;
  - record the audit coverage and the audit false-certificate rate `FCa`. With about 2,000 issued audit classes, its
    standard error is about 0.006.
- **Violation:** a draw that issues something with audit `FCa` > α. A draw that issues nothing is not a violation (it
  makes no claim).
- **Caveat:** the draws are subsamples of a finite pool, and the audit set is a separate finite sample. Both
  approximate draws from the stimulus distribution, and the pool's own sampling variation is shared within a seed.

## 3. Metrics

- Per seed and pooled:
  - the violation rate;
  - mean audit coverage (zero for non-issuing draws);
  - the non-issuance rate;
  - mean `FCa` over issuing draws;
  - mean faithfulness among issued (audit).
- Reported: gate AUC (audit); held-out-word agreement among issued (extrapolation, not certified).
- **Soundness checks (an abort):** as #149 (certified-class predictions, the exact k = 1 box, the gate's input
  identical across variants, p ≤ δ at every rejected point).

## 4. Decision rules (set with #149 in view)

- **H1 (validity):** the pooled violation rate over 5 × 200 draws is **≤ 0.15**. That is δ = 0.10 plus a margin for
  audit-estimate noise and the finite pool.
- **H2 (coverage):** pooled mean audit coverage **≥ 0.25**.
- **H3 (reliability):** the pooled non-issuance rate is **≤ 0.10**.
- **Reported, no hypothesis attached:**
  - the per-seed violation rates;
  - the first draw of each seed as a single-split replicate, comparable to #149.

## 5. Not claimed; tags

- The guarantee is the LTT theorem's, cited and **not machine-checked**. A kernel-checked binomial fixed-sequence lemma
  in i-orca is `open`.
- **H1 is an empirical audit of a stated guarantee** on this task, this template, and this nuisance position. It is
  not a proof of it, and nothing about distribution shift is claimed.
- The student's certificate is a sound IBP (float32, sound up to rounding); the gate is invariant by construction;
  held-out words are not certified.

## 6. Protocol and artifacts

- Script `experiments/ltt_retest.py` and tests, frozen (commit and sha256) before the real run.
- **`--smoke`:** stimulus 999, 1,440 σ (the same fractions), 300 steps, R = 20. Smoke numbers are not results.
- The run is wrapped in `systemd-inhibit --what=sleep:idle`, launched in the repo.
- Any change after smoke goes in a dated addendum, before the real run.
- **Outcome:** `docs/notes/ltt_retest_outcome.md`, the summary JSON, and a table test. The outcome is edited only
  through a labelled post-review section.

---

## Addendum A (2026-10-06, after the smoke run, before any real run): disclosure only, no design change

- **The smoke run** (stimulus 999, 1,440 σ, 300 steps, R = 20) completed, and the soundness checks held. At 300 steps
  every student is degenerate (faithfulness 0.03–0.05). LTT issued on 1 of 20 draws for seed 50 and on none for the
  others, so the issuing and audit path ran end to end. These are not results.
- **Script frozen at `d4aa121`**, sha256 of `experiments/ltt_retest.py` `c39f1672a8eefc65…`. Check it with
  `git show d4aa121:experiments/ltt_retest.py | sha256sum`.
- Thresholds in §4 unchanged.

## Addendum B (2026-10-06, after a failed launch, before any result): an implementation fix, no design change

- **The first dump hung**, at 100% CPU with the GPU idle and nothing written, for about 53 minutes. A timed copy of
  the same steps ran alongside it and every step finished in seconds. The hung process was killed and the identical
  command re-run, finishing in 3 min 48 s. The cause was not identified (probably a stuck CUDA synchronisation).
- **The first real run crashed with CUDA out-of-memory** in seed 50's evaluation, after training and before any metric
  was computed or printed.
  - Cause: #146's `evaluate()` and #148's `cert16_and_decision()` bound a whole split in one batch, and this study's
    splits are 6,000 classes (earlier ones were at most 1,800).
  - **Fix:** both are now called over chunks of 1,500 classes and the results combined. The bound is per row, so the
    results are unchanged. A new unit test checks that chunked and unchunked outputs are identical, and the smoke run
    reproduced its previous output exactly.
- **No number from the real data was seen.** The crash log contains only the data line (audit seen-constant 0.721).
- **Re-frozen at `f9b2690`**, sha256 of `experiments/ltt_retest.py` `8adac330d020be76…` (it supersedes addendum A's freeze).
  Check it with `git show f9b2690:experiments/ltt_retest.py | sha256sum`. Thresholds in §4 unchanged.
