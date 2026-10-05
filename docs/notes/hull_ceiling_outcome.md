# Why the uniform substitution certificate fails: hull ceiling vs alignment — outcome

**Pre-registration:** [`hull_ceiling_prereg.md`](hull_ceiling_prereg.md), with addenda A and B.
- Provenance, in order:
  - pre-registration `60a2808`, pushed before any code;
  - addendum A `9f49527` and addendum B `bc3d76a`, both after smoke runs on non-study data (seed 999), before any
    real run;
  - script and tests frozen at `be00b28` (`experiments/hull_ceiling.py`, sha256 `df9f0826…`). This run used that
    commit unchanged.
- Seeds: stimulus 31, split 32, fit 0. The real run took 30 min (GPU training, float64 analysis).
- Summary: [`hull_ceiling_summary.json`](hull_ceiling_summary.json).
- Theorems: i-orca#29 (merged), `certificate_hull_ceiling` and `certificate_hull_ceiling_biased`, kernel-checked.
- Tags: the ceiling is `proved`; every count below is `empirical`.

## Verdict

**Headline: NOT-EXCLUDED-BOUND, in 16/16 substitute cells** (addendum A rule: ≥ 80% of a cell's decided refusals,
majority over 16 cells).

- **Control passed** on both tasks: PCA-240 uniform coverage is 0.936 on LIST (#134 had 0.942) and 0.616 on SVO,
  against a threshold of 0.50.
- **Soundness:** both aborts held on every context (the bias-free bound `m ≤ ‖u‖·h_hi`, and the lifted bound for
  every `s`).
- **Solver:** every hull bracket converged on both tasks, for every lift scale (LIST 100/100 tokens, SVO 87/87;
  at most 105 Frank–Wolfe iterations against a cap of 20,000). At most 1 context per cell is UNDECIDED.
- **Original, vacuous classification** (bias-free, §3; kept as secondary): ALIGNMENT-BOUND in 16/16 cells, as the
  smoke runs predicted.

## Per cell (fit seed 0; refused = uniform-refused test contexts, of 3,600)

| task | cell | agree | uniform | refused | CEILING | NOT-EXCLUDED | UNDECIDED |
|---|---|---:|---:|---:|---:|---:|---:|
| LIST | B8 tpr mse | 0.026 | 0.000 | 3600 | 335 | 3265 | 0 |
| LIST | B8 tpr cert | 0.643 | 0.000 | 3600 | 205 | 3394 | 1 |
| LIST | B8 paircode mse | 0.003 | 0.000 | 3600 | 421 | 3178 | 1 |
| LIST | B8 paircode cert | 0.496 | 0.000 | 3600 | 285 | 3314 | 1 |
| LIST | B32 tpr mse | 0.742 | 0.000 | 3600 | 131 | 3468 | 1 |
| LIST | B32 tpr cert | 0.968 | 0.000 | 3600 | 109 | 3490 | 1 |
| LIST | B32 paircode mse | 0.091 | 0.000 | 3600 | 314 | 3285 | 1 |
| LIST | B32 paircode cert | 0.830 | 0.000 | 3600 | 388 | 3212 | 0 |
| LIST | control PCA-240 | 0.998 | 0.936 | 232 | 0 | 232 | 0 |
| SVO | B8 tpr mse | 0.086 | 0.000 | 3600 | 82 | 3518 | 0 |
| SVO | B8 tpr cert | 0.650 | 0.000 | 3600 | 31 | 3569 | 0 |
| SVO | B8 paircode mse | 0.030 | 0.000 | 3600 | 93 | 3507 | 0 |
| SVO | B8 paircode cert | 0.801 | 0.000 | 3600 | 27 | 3572 | 1 |
| SVO | B32 tpr mse | 0.741 | 0.001 | 3598 | 12 | 3586 | 0 |
| SVO | B32 tpr cert | 0.879 | 0.001 | 3598 | 10 | 3588 | 0 |
| SVO | B32 paircode mse | 0.855 | 0.010 | 3565 | 6 | 3559 | 0 |
| SVO | B32 paircode cert | 0.881 | 0.004 | 3584 | 9 | 3575 | 0 |
| SVO | control PCA-240 | 0.981 | 0.616 | 1381 | 0 | 1381 | 0 |

`agree` is decision agreement, which equals pairwise certified coverage (#135). The CEILING share of refusals is
0.2–3% on SVO and 3–12% on LIST. Where CEILING occurs, the median shrink factor `2δ / ceiling` is 1.05–1.21, so
those contexts sit just past the ceiling.

## The host quantities (independent of any substitute)

| task | `‖u‖` median | `‖mean u‖` | `‖u − mean u‖` median | host margin `m` median | lifted ceiling median (best `s`) | ceiling fraction `m / ceiling`, quartiles |
|---|---:|---:|---:|---:|---:|---|
| LIST | 146.9 | 144.0 | 23.8 | 3.87 | ≈ 47.7 (`s` = 5) | 0.048 / **0.077** / 0.120 |
| SVO | 226.6 | 225.7 | 14.0 | 1.45 | ≈ 26.6 (`s` = 3) | 0.026 / **0.051** / 0.084 |

## Reading (interpretation)

1. **The hull ceiling does not explain the uniform certificate's failure.** For almost every refused context, the
   substitute's error 2δ is below what GPT-2's unembedding geometry would allow at that residual norm. Yet coverage
   is ≤ 0.010 in every trained cell. The binding constraint is the host's own margin, not fit error measured
   against the geometry.
2. **GPT-2's margins are a small fraction of the ceiling: median 5% on SVO and 8% on LIST.** The lifted ceiling is an
   upper bound and is not attained in general. So this is a **lower bound** on the share of the achievable margin
   GPT-2 uses; the true share could be higher, by an unknown amount.
3. **NOT-EXCLUDED is not "an aligned residual exists".** Addendum A says so, and it holds here too. The ceiling does
   not rule certification out on the fixed `s` grid. Whether any residual GPT-2 could produce reaches a certifiable
   margin is not established, so whether margin widening (PROPOSAL T6) can help stays `open`.
4. **Better fits move refusals away from the ceiling,** as expected. On SVO, CEILING falls from 27–93 refusals at
   B8 to 6–12 at B32. The `cert` objective lowers it in 6 of 8 (task × family × budget) pairs. The exceptions are B32 paircode on both tasks (LIST 314 → 388, SVO 6 → 9).
5. **This confirms the smoke disclosure in addendum A only partly.** That addendum's predicted outcome (NOT-EXCLUDED)
   was withdrawn in addendum B as resting on untrained substitutes. The trained fits landed on the NOT-EXCLUDED side
   anyway.

## Not claimed

- One fit seed (0) per cell. These are the pre-registered counts, not seed means.
- `h(t)` and the ceiling are properties of GPT-2 small's tied unembedding, on these two templated tasks.
- The classification concerns **threshold** certificates. The exact pairwise certificate has no ceiling, and its
  coverage equals `agree` above.
- Nothing extends to unseen contexts. The T5(b) domain certificate needs an error bound on the whole domain, and
  that bound stays `open`.
