# How much could a model-side bound certify? The nuisance-class hull ceiling — outcome

**Pre-registration:** [`nuisance_hull_prereg.md`](nuisance_hull_prereg.md), with addendum A.
- Provenance, in order:
  - pre-registration `6ae0e66`, pushed before any code; it discloses that exploratory measurements on separate
    seeds were made first, and that **the thresholds were set after seeing them**;
  - addendum A (smoke disclosure, no design change);
  - the prerequisite kernel lemma, i-orca#35 `hull_certified_conditions`, merged (`5c67951`) before the run;
  - script and tests frozen at `7e7d741` (`experiments/nuisance_hull.py`, sha256 `6a0e5d21…`). It reuses
    `domain_exhaustion.py` (#142), `halfspace_verdict.py` (#141) and their dependencies unchanged; the run used all of
    them unchanged.
- Seeds: stimulus 91, split 92, fit {0, 1, 2}. 1,200 σ-classes × 8 time phrases = 9,600 contexts; class split 720 /
  120 / 360. 0 out of vocabulary.
- The dump and the run were held under one `systemd-inhibit --what=sleep:idle` lock, uninterrupted (797 s).
- Summary: [`nuisance_hull_summary.json`](nuisance_hull_summary.json).
- Tags: the hull certificate of a class is **proved** for that fit (per-context certificates plus
  `hull_certified_conditions`). Every count below is `empirical`.

## What "hull-certified" means

A σ-class is the 8 sentences that share S, V and O and differ only in the opening time phrase. The substitute sees
only σ. A class is hull-certified when all 8 variants pass #141's per-context certificate. Both conditions are
strict linear inequalities in the residual, so they then hold on the whole convex hull of the 8 residuals.

Any sound bound on GPT-2's output over the 8 phrases must contain those residuals. So the hull-certified share is
the **ceiling** for any model-side (set-level) certificate over the class. **No bound is computed here.**

## Verdict

**H1 (non-trivial ceiling, `H_all ≥ 0.10`): PASS** — `t6` gives `H_all` = **0.151** [0.131, 0.172].

**H2 (`H_const ≥ 0.20` on decision-constant classes): PASS** — `t6` gives `H_const` = **0.232** [0.201, 0.265].

- Both verdicts are carried by `t6`. `mse` (0.071 / 0.110) and `cert` (0.014 / 0.021) are below both thresholds.
- The thresholds were set after the disclosed exploration, which had given about 0.19 / 0.28 for `t6` on 2 seeds.
  The confirmatory values are lower: 0.151 / 0.232.
- **Soundness:**
  - all **4,080** random convex combinations of hull-certified classes kept exact clean-up and the code decision
    (the hull abort);
  - all **73,400** neighbourhood perturbations held;
  - 0 ties; Soufflé and Python agreed on every context; no fit was flagged.

## Per cell (test classes, seed mean [min, max])

| objective | `H_all` | `H_const` | hull-certified classes per seed | per-context `F_cov` | median `ρ_loc` |
|---|---:|---:|---|---:|---:|
| mse | 0.071 [0.000, 0.136] | 0.110 [0.000, 0.209] | 0, 28, 49 | 0.262 [0.014, 0.421] | 0.096 |
| cert | 0.014 [0.003, 0.025] | 0.021 [0.004, 0.038] | 1, 9, 5 | 0.127 [0.066, 0.162] | 0.082 |
| t6 | **0.151** [0.131, 0.172] | **0.232** [0.201, 0.265] | 54, 47, 62 | 0.461 [0.425, 0.529] | 0.139 |

- The host decision is constant across all 8 phrases in **65.0%** of test classes (234 of 360). Only these can be
  hull-certified.
- The median within-class residual radius is **9.68**, against median certified neighbourhoods of 0.08–0.14.

## Reading (interpretation)

1. **The model-side route has a non-zero ceiling.** For about 15% of all held-out σ-classes, 23% of the
   decision-stable ones, with `t6`, a perfectly tight sound bound over the 8 time phrases would certify every variant
   without running GPT-2 on each. That holds even though the variants' residuals spread over a radius about 70× the
   per-context neighbourhoods: the certificate region is a long polytope, and the phrase variation runs along benign
   directions.
2. **The ceiling is fit-sensitive.** `mse` seed 0 certifies no class. `cert` is low throughout. `t6`, the objective
   that pushes code points into the host's decision cell, is consistently best.
3. **What this does not show** is whether any computable bound gets near the hull. That is study (2), and its
   feasibility is open. Interval and linear-relaxation bounds through 12 layers of GPT-2 are expected to be loose. A
   bound only needs to control the few linear functionals that define the certificate, which helps.
4. **A design consequence for study (2).** With 8 phrases, enumerating all variants costs 8 forward passes, which is
   probably no more than computing a bound. A model-side bound pays off only for nuisance sets much larger than its own
   cost, such as products of several independent nuisance slots.

## Not claimed

- No bound is computed; the ceiling is what a perfectly tight sound bound would achieve.
- One nuisance set (8 time phrases). Adjectives failed in exploration because GPT-2 copies them into its prediction.
- GPT-2 small, one template, TPR `d_F=32` only.
