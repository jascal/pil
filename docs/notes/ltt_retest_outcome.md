# LTT re-test: independent calibration draws scored on an audit set — outcome

**Pre-registration:** [`ltt_retest_prereg.md`](ltt_retest_prereg.md), with addenda A and B.
- Provenance, in order:
  - pre-registration `97bfa49`, pushed before any code; the thresholds were set with #149 and its post-hoc analysis in
    view;
  - script frozen at `d4aa121` (addendum A, smoke);
  - **re-frozen at `f9b2690`** (`experiments/ltt_retest.py`, sha256 `8adac330…`; addendum B).
- **What addendum B covers:**
  - the first dump hung, so it was killed and re-run identically;
  - the first real run hit CUDA out-of-memory before any metric, so the IBP evaluation now runs in chunks (tested
    result-identical).
  - No real-data number was seen before the final run.
- Seeds: stimulus 181, split 182, fit and gate {50–54}. 14,400 σ-classes: train 2,400, calibration pool 6,000 (200
  draws of 1,800 per seed), audit 6,000. GPT-2 is seen-constant on **71.4%** of the pool and **72.1%** of the audit set.
- α = 0.10, δ = 0.10, with the grid starting at the train quantile 0.80.
- The run was held under `systemd-inhibit` and was not interrupted (1,189 s).
- Files: summary [`ltt_retest_summary.json`](ltt_retest_summary.json); log [`ltt_retest_run.txt`](ltt_retest_run.txt).

## Verdict (pooled over 5 seeds × 200 draws)

- **H1 (violation rate ≤ 0.15): PASS** (0.068).
- **H2 (coverage ≥ 0.25): FAIL** (0.177).
- **H3 (non-issuance ≤ 0.10): FAIL** (0.330).
- **Soundness:** no abort fired. Every rejected grid point had p ≤ δ, the k = 1 box was exact, and the gate's input
  was identical across variants.
- **When a draw issued,** the mean audit false-certificate rate was **0.083**, faithfulness among issued 0.908, and
  held-out-word agreement among issued 0.937 (uncertified).

## Per seed (audit set)

| seed | faithfulness (seen) | gate AUC | violation rate | coverage | non-issuance | mean `FCa` when issuing | faithful among issued | student = GPT-2 on held-out, among issued | first draw: coverage / `FCa` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 50 | 0.842 | 0.760 | 0.025 | 0.234 | 0.175 | 0.086 | 0.900 | 0.929 | 0.161 / 0.070 |
| 51 | 0.848 | 0.761 | 0.030 | 0.217 | 0.235 | 0.081 | 0.911 | 0.940 | 0.325 / 0.084 |
| 52 | 0.851 | 0.751 | 0.135 | 0.124 | 0.535 | 0.095 | 0.896 | 0.924 | 0.000 / — |
| 53 | 0.846 | 0.758 | 0.000 | 0.145 | 0.390 | 0.069 | 0.924 | 0.950 | 0.270 / 0.075 |
| 54 | 0.852 | 0.748 | 0.150 | 0.163 | 0.315 | 0.085 | 0.911 | 0.940 | 0.217 / 0.082 |

## Reading (interpretation)

1. **The calibrated bound holds.**
   - Over 1,000 calibration draws, the selected threshold's audit false-certificate rate exceeded α = 0.10 on 6.8% of
     them, inside δ = 0.10, measured on an audit set of 6,000 classes the calibration never touched.
   - So #149's failure was the shared split, as its post-hoc analysis indicated, not the procedure.
   - The rate is not uniform across seeds: 0.000–0.150, with two seeds (0.135, 0.150) above δ.
     - Within a seed, the draws are subsamples of one 6,000-class pool, so that pool's sampling variation is shared
       across all 200 draws.
     - The audit estimate also carries noise (standard error about 0.006 to 0.01 at these coverages).
     - So per-seed rates do not each estimate the guarantee independently. The pooled rate is the pre-registered unit.
2. **The guarantee costs coverage.**
   - The calibrated certificate issues on 17.7% of sentences on average, and in a third of draws not at all. That
     misses both pre-registered bars.
   - When it issues, coverage is 0.12–0.23 per seed, with about 8% false certificates and 91% faithful.
   - The gates here were weaker than #148's (AUC 0.748–0.761, against 0.775–0.792). A gate whose precision near the
     top of the score range sits close to 1 − α leaves the binomial test little room to reject, so fixed-sequence
     testing stops early.
   - Coverage is bounded by gate quality, not by the calibration.
3. **Where the direction stands.**
   - A student whose invariance certificate is proved for itself.
   - Plus a word-blind gate.
   - Plus Learn-then-Test calibration, which gives certificates that are wrong about GPT-2 with a **controlled**
     probability (the bound held on 93.2% of draws at α = δ = 0.10).
   - The price is coverage of about a sixth of sentences on this task. Raising it is a gate-quality problem.

## Not claimed

- **The LTT guarantee is cited, not machine-checked.** H1 is an empirical audit of it on this task, template and
  nuisance position.
- The draws share a finite pool within each seed. Nothing about distribution shift is claimed.
- Held-out-word agreement is extrapolation, not certified.
