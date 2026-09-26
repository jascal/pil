# Directional early-exit calibration — fresh EVAL2 outcome

Pre-registration: [`directional_calibration_prereg.md`](./directional_calibration_prereg.md), signed 2026-09-26
14:29 UTC and committed before EVAL2 was built. Development selection: C0, recorded in
[`directional_calibration_dev_outcome.md`](./directional_calibration_dev_outcome.md) before EVAL2 capture. Full
confirmation numbers: `results/directional_calibration_confirm.json`.

## Prespecified verdicts `[empirical]`

| Model | EVAL2 verdict | C0 certified | Violations among certified | Mean net saving | Median certified exit | Oracle-D coverage |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-3B-Instruct (primary) | **PARTIAL** | 44/150 (29.3%) | 0/44 | **+0.368 layers** | 34/36 | 138/150 (92.0%) |
| Qwen2.5-0.5B-Instruct | **FAILED** | 8/150 (5.3%) | 0/8 | +0.052 layers | 22/24 | 83/150 (55.3%) |

The primary 3B result clears the coverage and observed-violation bars but misses the ≥1-layer net-saving bar.
The 0.5B result fails its <10% coverage rule. This does not reverse the #130 conclusion: C0 still certifies mostly
at the end, while Oracle-D shows much earlier directional headroom (median exit 11/36 on 3B and 14/24 on 0.5B).
The preregistered C1/C2 development candidates did not beat C0, so there is no new calibration improvement to
deploy. These are empirical verdicts, not formal simultaneous error guarantees.

Wilson 95% intervals are 22.6–37.1% for 3B coverage and 0–8.0% for its certified violation rate; the 0.5B
intervals are 2.7–10.2% for coverage and 0–32.4% for the certified violation rate. The intervals describe the
small fresh sample; they are not a substitute for the signed point-estimate decision rules.

## Controls and provenance

- EVAL2 consists of 50 previously unused contexts per bAbI qa1–qa3, seed-0 permutation indices 150–199, six
  canonical `location` options, using parallel-decisions `45820320`. The builder checked disjointness from the
  450 seen CAL/EVAL contexts and identical 0.5B/3B prompt token IDs. Both ID files have SHA-256
  `25a66e157b9ca230284d9336ba6607a0786dad1b14129fb90c8a4af2690c0226`.
- fieldrun `3e38dfe` (CPU int8, `--tail 1`) wrote 150 records per model with reconstruction argmax 1.00 on both.
  Oracle-D had zero violations on each EVAL2 dump, so the soundness gate passed. Maximum `s`-recovery residuals
  were `2.20e-5` (3B) and `6.35e-6` (0.5B), below the `1e-3` gate.
- The 3B library fp16 versus fieldrun int8 agreement control is reported separately in
  `results/directional_calibration_library_agreement.json` if its CPU run completes; it is not a verdict gate.
- The preregistered synthetic binning/quantile tests and the full pil suite passed (862 tests); ruff passed.

## Scope

This is one 150-decision fresh confirmation for each of two Qwen2.5 models on one six-option bAbI field. The
selected C0 rule was refitted on all 450 seen decisions before EVAL2. No further candidate was tried on EVAL2.
