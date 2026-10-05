# The T6(b) clean-up certificate on GPT-2's decode input — outcome

**Pre-registration:** [`cleanup_certificate_prereg.md`](cleanup_certificate_prereg.md), with addendum A.
- Provenance, in order:
  - pre-registration `be1004f`, pushed before any code;
  - addendum A `2bc09d7` (smoke disclosure, no design change), before any real run;
  - script and tests frozen at `9faa885` (`experiments/cleanup_certificate.py`, sha256 `39469c10…`). The real run
    used that file unchanged.
- Seeds: stimulus 41, split 42, fit {0, 1, 2}. 12,000 contexts per task; 7,200 train and 3,600 test, with 0 test
  contexts dropped for an unseen pair.
- Summary: [`cleanup_certificate_summary.json`](cleanup_certificate_summary.json).
- Theorem: i-orca#30 `PIC_Cleanup.cleanup_certified` (kernel-checked). Every coverage count below is `empirical`.
- Wall time was 34,000 s. One fit (SVO `d_F=32` `t6`, seed 1) logged about 9 hours after the previous one; every
  other fit took about 2 minutes. The host most likely slept. Each fit is fully determined by its seed and the
  frozen script, so this changes no result.

## Verdict

**H1 (the clean-up certificate covers what the uniform one cannot): NO** — 0 of 3 settings.
**H2 (the `t6` objective matters): NO** — 0 of 3 settings.

- **T6 coverage is 0 in every one of the 27 fits** (9 cells × 3 seeds).
- **Both abort checks held.** Clean-up was run on every test context. With nothing certified, the "certified ⇒ exact
  and agrees" check held trivially on the real run; the unit tests exercise it on planted violations. Soufflé and
  the Python twin agreed on every context.
- **No flags triggered.** The smallest `σ_min/σ_max(W)` was 2.8 × 10⁻³ (threshold 10⁻⁸), the largest `‖PW − I‖_max`
  was 7.7 × 10⁻¹⁵ (10⁻⁸), and the largest `cond(er)` was 8.9 × 10⁴ (10⁸).
- Every objective ties at 0 coverage, so the H1 "best objective" is the first in order (`mse`); this does not
  affect the verdict.

## Per cell (seed mean [min, max] over 3 fit seeds; test split of 3,600)

| setting | objective | T6 cov | uniform cov | exact clean-up | cleaned agreement | median `‖n‖` | median `ρ` | median `β` | R / B | label |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| LIST `d_F=8` | mse | 0 | 0 | 0.000 | 0.035 [0.024, 0.058] | 12.76 | 0.0009 | 0.111 | 1.00 / 1.00 | both |
| LIST `d_F=8` | cert | 0 | 0 | 0.000 | 0.653 [0.644, 0.660] | 15.61 | 0.0115 | 0.120 | 1.00 / 1.00 | both |
| LIST `d_F=8` | t6 | 0 | 0 | 0.000 | 0.221 [0.164, 0.269] | 12.63 | 0.0005 | 0.079 | 1.00 / 1.00 | both |
| SVO `d_F=8` | mse | 0 | 0 | 0.021 [0.000, 0.042] | 0.056 [0.019, 0.080] | 9.00 | 0.0188 | 0.099 | 1.00 / 1.00 | both |
| SVO `d_F=8` | cert | 0 | 0 | 0.011 [0.000, 0.018] | 0.678 [0.661, 0.694] | 11.81 | 0.0437 | 0.137 | 1.00 / 1.00 | both |
| SVO `d_F=8` | t6 | 0 | 0 | 0.031 [0.000, 0.070] | 0.227 [0.062, 0.316] | 8.57 | 0.0249 | 0.117 | 1.00 / 1.00 | both |
| SVO `d_F=32` | mse | 0 | 0.0019 | 0.836 [0.801, 0.872] | 0.770 [0.739, 0.786] | 6.22 | 0.0722 | 0.257 | 1.00 / 1.00 | both |
| SVO `d_F=32` | cert | 0 | 0.0006 | 0.273 [0.136, 0.350] | 0.870 [0.869, 0.873] | 9.66 | 0.1390 | 0.427 | 1.00 / 1.00 | both |
| SVO `d_F=32` | t6 | 0 | 0.0002 | 0.808 [0.777, 0.826] | 0.838 [0.829, 0.848] | 6.28 | 0.0790 | 1.018 [0.985, 1.057] | 1.00 / 1.00 | both |

`R` and `B` are the shares of uncertified contexts with `‖n‖ ≥ ρ` and `‖n‖ ≥ β`. Medians are seed means of per-seed
medians.

## Reading (interpretation)

1. **Both radii bind in every cell, by a wide margin.** Every uncertified context — that is, every context — has
   `‖n‖` above both `ρ` and `β`.
   - The closest approach is SVO `d_F=32` with `t6`: `‖n‖ ≈ 6.3` against `β ≈ 1.0` (≈ 6×) and `ρ ≈ 0.08` (≈ 80×).
   - On LIST, `ρ` is 0.0005–0.0115 against `‖n‖ ≈ 13–16`, three to four orders of magnitude short.
2. **The `t6` objective does what it was built to do, but not enough.** On SVO `d_F=32` it raises the median `β`
   from 0.26 (`mse`) and 0.43 (`cert`) to 1.02, without raising `‖n‖` (6.28 vs `mse`'s 6.22). It does not move `ρ`,
   which the objective does not target. Neither change reaches the certificate.
3. **Clean-up works far outside its certified radius.** On SVO `d_F=32`, clean-up recovers the exact structure on
   81–84% of contexts (`mse`, `t6`) although `‖n‖` is about 80× `ρ`. The clean-up radius is a worst-case bound
   (`K = 1/σ_min(W)` and half the closest filler separation), and real noise is far from worst-case. This is the
   same gap between exact per-context checks and uniform bounds that #135 found for substitution.
4. **Uniform coverage stays ≤ 0.002 in every cell,** consistent with #135, #136 and #138.

## What this says about T6 (interpretation)

The clean-up certificate does not rescue threshold certification on GPT-2 small. Using the code point's margin
instead of GPT-2's raises the margin side (β up to ≈ 1.0 with `t6`), but residuals sit ≈ 6–15 away from their
nearest TPR code point. That is far outside both the agreement ball and, much more so, the worst-case clean-up
ball. Whether a tighter clean-up bound (per-direction rather than `1/σ_min`), or substitutes with much smaller
residual error, could close the gap stays `open`.

## Not claimed

- The certificate is per context and needs `u`. A zero result says nothing about whether GPT-2 computes a TPR.
- The clean-up radius `ρ` is the kernel-checked **sufficient** condition. Exact recovery beyond it (point 3) is
  measured, not certified.
- GPT-2 small, two templated tasks, the TPR family only. LIST at `d_F=32` is out of scope (no left inverse).
- Smoke-data figures in addendum A are not results.
