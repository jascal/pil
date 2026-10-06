# Conditional certificates: a word-blind gate decides where to claim invariance — outcome

**Pre-registration:** [`conditional_certificate_prereg.md`](conditional_certificate_prereg.md), with addendum A.
- Provenance, in order:
  - pre-registration `31cfb27`, pushed before any code, disclosing an exploratory probe; **the thresholds were set
    after the probe**;
  - script and tests frozen at `2002d8b` (`experiments/conditional_certificate.py`, sha256 `52de3eee…`);
  - addendum A `da0bd38` (smoke; the gate check moved from outputs to inputs; a smoke-only minimum of 5). The run used
    the frozen script unchanged, launched in the repo.
- Seeds: stimulus 161, split 162, fit and gate {30–34}. 4,000 σ-classes × 23 words; class split 2,400 / 400 / 1,200;
  103 outputs. Of the test classes, GPT-2 is **seen-constant on 73.6%** and all-constant on 71.8%.
- The run was held under `systemd-inhibit` and was not interrupted (1,147 s).
- Files: summary [`conditional_certificate_summary.json`](conditional_certificate_summary.json); log
  [`conditional_certificate_run.txt`](conditional_certificate_run.txt).
- Tags:
  - each issued certificate is **proved** for that student: a sound interval bound over the 16 seen words, in float32,
    sound up to rounding;
  - the gate is invariant to the nuisance word **by construction**: its integer input excludes position 0, which is
    checked on every class;
  - whether a certificate is right about GPT-2 is `empirical`.

## Verdict

- **H1 (`FCi` ≤ 0.10): PASS** (0.050).
- **H2 (coverage ≥ 0.20): PASS, narrowly** (0.205; per-seed range 0.095–0.303).
- **H3 (faithfulness among issued ≥ 0.90): PASS** (0.943).
- **Soundness:** no abort fired. The k = 1 box was exact, and the gate's input was identical across variants on every
  class.

## Per seed (test classes)

| seed | faithfulness (seen) | `cert16` | `FC16` ungated | gate AUC | coverage | issued | `FCi` | faithful among issued | GPT-2 constant on held-out, among issued | student = GPT-2 on held-out, among issued |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 0.854 | 0.999 | 0.264 | 0.787 | 0.244 | 293 | 0.085 | 0.915 | 0.908 | 0.962 |
| 31 | 0.851 | 1.000 | 0.264 | 0.787 | 0.095 | 114 | 0.018 | 0.974 | 0.982 | 0.982 |
| 32 | 0.854 | 1.000 | 0.264 | 0.790 | 0.252 | 302 | 0.043 | 0.950 | 0.954 | 0.970 |
| 33 | 0.859 | 1.000 | 0.264 | 0.775 | 0.303 | 363 | 0.074 | 0.923 | 0.923 | 0.964 |
| 34 | 0.862 | 1.000 | 0.264 | 0.792 | 0.131 | 157 | 0.032 | 0.955 | 0.968 | 0.975 |

`FCi` is the share of issued classes where GPT-2 is not constant over the 16 seen words. The last two columns are
extrapolation to the 7 held-out words, which are **not certified**.

## Reading (interpretation)

1. **Gating turns #147's 26% false certificates into about 5%, at about 20% coverage.**
   - The stable student is certified over the seen words on essentially every class (`cert16` 0.9998), so issuing is
     decided by the gate.
   - On the classes issued, the claim "the student's decision does not depend on the nuisance word, and neither does
     GPT-2's" was true on 95% of them (range 91.5–98.2%). The certificate itself is proved for the student.
2. **The issued decisions are also faithful** (0.943 on the seen words). On the held-out words they agree with GPT-2
   **0.971** of the time, though uncertified. The gate was trained only on seen-word constancy, yet its picks are
   constant over the held-out words too (0.947). So the "nuisance-insensitive sentence" signal it learned is not
   specific to the trained words (`empirical`).
3. **Coverage is the weak axis, and it is noisy.**
   - Seed-mean coverage (0.205) clears its bar by 0.005, and per seed it ranges 0.095–0.303.
   - The cause is τ: it is set as the lowest score with ≥ 0.95 precision on 400 validation classes, which is a small,
     discrete sample.
   - The gate's ranking is stable (AUC 0.775–0.792), so the threshold choice, not the gate, drives the spread.
   - Read H2's pass as marginal.
4. **What the result is, and what it isn't.**
   - The certificate is a proof about the student, plus an invariance-by-construction argument for the gate. The
     claim that the certificate is right about GPT-2 is a **measured rate**, about 5% wrong at this τ. It is not a
     proof, and not yet a statistical guarantee.
   - τ was tuned to a precision target on held-out validation data. Turning the measured rate into a distribution-free
     bound (for example, conformal risk control on the validation split) is the natural next step, and is `open`.
   - The base rate here (73.6% seen-constant) was higher than in the probe (70.8%), which makes precision targets
     easier to hit.

## Not claimed

- Held-out-word invariance is not certified; the 0.97 agreement is extrapolation.
- No guarantee about GPT-2 is claimed beyond the measured `FCi` on this test split.
- One template, one nuisance position, GPT-2 small, one gate design, and one precision target (0.95).
