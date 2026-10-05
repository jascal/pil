# Per-context half-space verdicts and certified neighbourhoods — outcome

**Pre-registration:** [`halfspace_verdict_prereg.md`](halfspace_verdict_prereg.md), with addendum A.
- Provenance, in order:
  - pre-registration `d2396f3`, pushed before any code;
  - addendum A `bc8e7ab` (smoke disclosure, no design change);
  - the prerequisite kernel theorem, i-orca#34 `cleanup_local_certified`, merged (`217e195`) before the real run;
  - script and tests frozen at `c6272a6` (`experiments/halfspace_verdict.py`, sha256 `73226e35…`). It reuses
    `cleanup_certificate.py` (#139) and `cleanup_radius.py` (#140) unchanged. The run used all three files unchanged.
- Seeds: stimulus 61, split 62, fit {0, 1, 2}. 7,200 train and 3,600 test per task; 0 test contexts dropped. The run
  was held under `systemd-inhibit --what=sleep:idle` throughout and was not interrupted.
- Summary: [`halfspace_verdict_summary.json`](halfspace_verdict_summary.json).
- Theorems: `nearest_iff_halfspace`, the pairwise iff, `cleanup_local_certified` (i-orca, kernel-checked). Every count
  below is `empirical`; each certified context's certificate is `proved` for that context.

## Verdict

**H1 (verification): PASS.** On all 27 fits, the Soufflé `clean` verdict equalled measured clean-up exactness, and
`agree` equalled measured host agreement. There were 0 mismatches and 0 ties; Soufflé and Python agreed on every
context.

**H2 (per-context certificates are substantial): PASS.** Full per-context coverage reaches **0.666** (SVO `d_F=32`,
`mse`) and **0.693** (SVO `d_F=32`, `t6`).

**H3 (the neighbourhoods are not negligible): PASS** in all 3 eligible cells (`F_cov ≥ 0.05`): SVO `d_F=32` with
`mse` (0.0271), `t6` (0.0260) and `cert` (0.0104). The `cert` cell passes narrowly, against the 0.01 threshold.

**Neighbourhood soundness:** all **173,460** sampled perturbations held: 8 random plus the 2 worst-case directions
at `0.999·ρ_loc`, around every certified context. Each kept exact clean-up and the code point's decision. No fit was
flagged.

## Per cell (seed mean [min, max]; 3,600 test contexts)

| setting | objective | E | A | **F_cov** | median `ρ_loc/‖n‖` | median `ρ_loc` | median `δ_c` | median `δ_h` | median `‖n‖` | `ρ_loc/ρ_dir` | host-bound share | label |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LIST `d_F=8` | mse | 0.000 | 0.034 | 0.000 | — | — | — | — | 12.77 | — | — | — |
| LIST `d_F=8` | cert | 0.000 | 0.620 | 0.000 | — | — | — | — | 15.84 | — | — | — |
| LIST `d_F=8` | t6 | 0.000 | 0.217 | 0.000 | — | — | — | — | 12.62 | — | — | — |
| SVO `d_F=8` | mse | 0.024 | 0.058 | 0.003 | 0.0082 | 0.0786 | 0.0810 | 0.7357 | 9.05 | 0.31 | 0.059 | clean-up-bound |
| SVO `d_F=8` | cert | 0.013 | 0.693 | 0.010 | 0.0053 | 0.0589 | 0.0679 | 0.5129 | 11.93 | 0.13 | 0.093 | clean-up-bound |
| SVO `d_F=8` | t6 | 0.031 | 0.307 | 0.015 | 0.0102 | 0.0765 | 0.0787 | 0.5343 | 8.65 | 0.24 | 0.041 | clean-up-bound |
| SVO `d_F=32` | mse | 0.839 | 0.775 | **0.666** [0.640, 0.701] | 0.0271 | 0.1759 | 0.2014 | 0.4785 | 6.24 | 0.43 | 0.187 | clean-up-bound |
| SVO `d_F=32` | cert | 0.247 | 0.881 | **0.220** [0.119, 0.294] | 0.0104 | 0.0923 | 0.0989 | 0.4551 | 9.68 | 0.18 | 0.092 | clean-up-bound |
| SVO `d_F=32` | t6 | 0.805 | 0.856 | **0.693** [0.657, 0.714] | 0.0260 | 0.1718 | 0.2033 | 0.4632 | 6.36 | 0.31 | 0.211 | mixed |

- `E` = exact clean-up, `A` = host agrees with the code point, `F_cov` = both, certified per context.
- Neighbourhood medians are over certified contexts. "Host-bound share" is the share of certified contexts with
  `δ_h < δ_c`.

## Reading (interpretation)

1. **Per-context certification works where code-centred balls failed.** In #139 and #140, every ball centred on the
   code point certified nothing. Here, about two-thirds of SVO `d_F=32` contexts carry a kernel-backed, per-context
   proof that clean-up is exact and the host decides the code point's decision. This repeats the
   exact-versus-uniform pattern of #135, now for clean-up.
2. **The certificates extend to neighbourhoods.** Around each certified observed residual, every residual within
   `ρ_loc` is certified (`cleanup_local_certified`). `ρ_loc` is a median 0.17–0.18 on the best cells, about 2.6–2.7%
   of the residual's distance from its code point. This is robustness to perturbation of observed residuals, and the
   first coverage in this line beyond the evaluated points. It is not coverage of unseen contexts.
3. **Clean-up, not the host, is the binding slack.** `δ_c` (≈ 0.20) is well below `δ_h` (≈ 0.46–0.48) on the best
   cells. 79–96% of certified contexts are clean-up-bound; `t6` is "mixed" at 21% host-bound.
4. **Local and code-centred radii measure different things.** `ρ_loc` is 0.13–0.43× the code-centred `ρ_dir`, but it
   bounds a perturbation of the observed residual. `ρ_dir` bounds the residual's whole offset from the code point,
   which here is about 6.2, far beyond any `ρ_dir`.
5. **LIST still has no exact clean-up** (E = 0), so it has no certified contexts, as in #139/#140.

## Not claimed

- Per-context certificates need the observed residual. They do not certify unseen contexts, and they do not say
  where an unseen context's residual will land.
- `F_cov ≤ min(E, A)` by construction. The certificate adds proof, not coverage beyond what clean-up and agreement
  already achieve.
- GPT-2 small, two templated tasks, the TPR family only; LIST `d_F=32` is out of scope (no left inverse).
