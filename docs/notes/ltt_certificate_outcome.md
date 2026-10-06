# Calibrated conditional certificates (Learn-then-Test) — outcome

**Pre-registration:** [`ltt_certificate_prereg.md`](ltt_certificate_prereg.md), with addendum A.
- Provenance, in order:
  - pre-registration `31bac30`, pushed before any code; no new probe, and the thresholds were set with #148's results
    in view;
  - script and tests frozen at `da3976e` (`experiments/ltt_certificate.py`, sha256 `0220780a…`);
  - addendum A `b1a113c` (smoke and the simulation check). The run used the frozen script unchanged, launched in the
    repo.
- Seeds: stimulus 171, split 172, fit and gate {40–44}. 6,000 σ-classes; split 2,400 / 1,800 / 1,800; 107 outputs.
  GPT-2 is seen-constant on **70.4%** of calibration and **70.7%** of test classes.
- α = 0.10, δ = 0.10.
- The run was held under `systemd-inhibit` and was not interrupted (1,162 s).
- Files: summary [`ltt_certificate_summary.json`](ltt_certificate_summary.json); log
  [`ltt_certificate_run.txt`](ltt_certificate_run.txt).

## Verdict

- **H1 (at most 1 of 5 seeds with test `FCi` > 0.10): FAIL.** **4 of 5** seeds exceeded it (0.112–0.126). The fifth
  issued nothing.
- **H2 (seed-mean coverage ≥ 0.20): PASS** (0.320).
- **H3 (coverage range ≤ 0.10): FAIL** (0.428). Seed 40 issued nothing; the others gave 0.373–0.428.
- **Soundness:** no abort fired. Every rejected grid point had p ≤ δ, the k = 1 box was exact, and the gate's input
  was identical across variants.

## Per seed (test classes)

| seed | faithfulness (seen) | gate AUC | grid points rejected | calibration wrong / issued at τ̂ | coverage | `FCi` | faithful among issued | student = GPT-2 on held-out, among issued | #148 rule: coverage | #148 rule: `FCi` |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 40 | 0.834 | 0.760 | 0 | — | 0.000 | — | — | — | 0.076 | 0.051 |
| 41 | 0.842 | 0.779 | 43 | 59/714 | 0.404 | 0.114 | 0.879 | 0.919 | 0.226 | 0.061 |
| 42 | 0.852 | 0.760 | 45 | 61/737 | 0.428 | 0.126 | 0.869 | 0.921 | 0.191 | 0.073 |
| 43 | 0.840 | 0.744 | 40 | 59/699 | 0.373 | 0.126 | 0.863 | 0.912 | 0.084 | 0.079 |
| 44 | 0.851 | 0.750 | 42 | 54/680 | 0.396 | 0.112 | 0.882 | 0.927 | 0.223 | 0.082 |

## Post-hoc diagnostics (not pre-registered; exploratory)

The four violations share one pattern: calibration says 0.079–0.084, test says 0.112–0.126. To find out whether that
is systematic or a property of this split, two scripts were run on the same dump. They are committed as
`experiments/ltt_posthoc_logistic.py` and `experiments/ltt_posthoc_gates.py`, and take `cert16` as all-true (the
study's students had 0.998–1.000).

| gate | pre-registered split (calibration → test) | calibration and test swapped | random calibration/test re-splits: mean calibration / mean test | re-splits with test > 0.10 |
|---|---|---|---|---|
| logistic on one-hot S, V, O | 0.084 → 0.092 | 0.082 → 0.066 | 0.083 / 0.084 (300 of 300 issued) | 0.157 |
| study gate, seed 41 (retrained) | 0.083 → **0.114** | 0.079 → 0.062 | 0.081 / 0.081 (200 of 200) | 0.135 |
| study gate, seed 42 (retrained) | 0.083 → **0.126** | 0.078 → 0.060 | 0.079 / 0.081 (199 of 200) | 0.146 |
| study gate, seed 43 (retrained) | 0.083 → **0.122** | nothing issued | 0.072 / 0.085 (53 of 200) | 0.170 |
| study gate, seed 44 (retrained) | 0.079 → **0.112** | 0.073 → 0.050 | 0.079 / 0.079 (200 of 200) | 0.115 |

- **Retrained gates:** seeds 41, 42 and 44 reproduce the study's test numbers exactly. Seed 43 comes close (0.122 vs
  0.126; coverage 0.364 vs 0.373), from GPU nondeterminism and from taking `cert16` as all-true.
- **Re-splits:** for the three gates that issue on (almost) every re-split, the calibration estimate is **unbiased** for
  test.
- **Swapping the halves reverses the gap:** test is 0.050–0.066 when the two halves trade places.
- **Seed 43's gate** often fails at the first grid point (it issues on 53 of 200 re-splits). Among the re-splits where
  it does issue, test exceeds calibration (0.085 vs 0.072). That is the expected selection effect of issuing only when
  the top region looks clean on calibration.
- **What this means:**
  - The procedure is not broken.
  - The split is not easier in base rate (70.4% vs 70.7%).
  - On the gates' top-scoring region, the pre-registered calibration half happens to be easier than the test half,
    and all five seeds share that one split.
- **Test exceeded 0.10 in 12–17% of re-splits.** That rate is the test *estimate* exceeding α, so it includes
  test-sample noise on top of the δ = 0.10 bound for the true rate. It is consistent with a valid procedure.

## Reading (interpretation)

1. **H1 failed because of a design flaw, not a broken guarantee.**
   - LTT's guarantee is over the calibration draw. The pre-registration used **one** calibration/test split for all
     five seeds, so the five seeds are not five independent chances. Their gates are also similar functions trained on
     the same data, so one unlucky split fails them together.
   - "At most 1 of 5" silently assumed independence. The post-hoc re-splits show the procedure is unbiased, with a
     test-exceedance rate near δ plus noise.
   - **The pre-registered verdict stays FAIL.** A valid test of the guarantee needs independent calibration draws per
     seed, or many re-splits pre-registered as the unit.
2. **Fixed-sequence testing is fragile at the top of the grid.** Seed 40 failed to reject the first grid point and
   issued nothing, which alone breaks H3. A grid starting further from the extreme top, or a different family-wise
   procedure, would avoid this. That is a design choice for a future study, not a fix to this one.
3. **When it issues, LTT gives about twice #148's coverage** (0.37–0.43 against 0.08–0.23 for the 0.95-empirical rule
   on the same calibration split). The cost is a higher realised false rate (0.11–0.13 on this split) and lower
   faithfulness among issued (0.86–0.88). At α = 0.10 the guarantee is weaker than #148's 0.95 target, so this is
   expected.
4. **What the direction has, and has not, established.**
   - The gated certificate works empirically (#148).
   - A calibrated bound on its false rate is achievable in principle: the procedure is valid in re-splits.
   - It was **not** demonstrated by this pre-registered test, because of the shared split and the fragile grid start.

## Not claimed

- No distribution-free guarantee is claimed as demonstrated. The LTT theorem is cited, not machine-checked, and the
  pre-registered test of it failed for the reasons above.
- Everything in the post-hoc section is exploratory and is not used to change any verdict.
