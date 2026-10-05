# How much exact clean-up does the directional radius certify? — outcome

**Pre-registration:** [`cleanup_radius_prereg.md`](cleanup_radius_prereg.md), with addendum A.
- Provenance, in order:
  - pre-registration `3c3bbb7`, pushed before any code;
  - addendum A `402f8f1` (smoke disclosure, no design change), before any real run;
  - script and tests frozen at `b577937` (`experiments/cleanup_radius.py`, sha256 `9cf68715…`). It reuses
    `experiments/cleanup_certificate.py` unchanged from #139 (sha256 `39469c10…`). The reported run used both files
    unchanged.
- Seeds: stimulus 51, split 52, fit {0, 1, 2}. 7,200 train and 3,600 test per task; 0 test contexts dropped.
- Summary: [`cleanup_radius_summary.json`](cleanup_radius_summary.json).
- Theorems: i-orca#32, `PIC_Cleanup.thy` (kernel-checked). Every count below is `empirical`.

## Disclosure: a stalled first run

- The first real run started at 09:24 and logged 15 of 27 fits. The machine then entered a system suspend at
  09:43:50 (journal: `PM: suspend entry (s2idle)`), 53 s after the last log line, and suspended again three more
  times. The process never resumed useful work: after resume it busy-looped in native code with the GPU idle.
  - Its log is kept verbatim: [`cleanup_radius_stalled_run.txt`](cleanup_radius_stalled_run.txt).
- It was killed at 12:16 and **restarted with the same frozen script, the same seeds and no code change**, under a
  `systemd-inhibit --what=sleep:idle` lock. **The restart is the reported run.**
- **The 15 fits both runs logged:**
  - the 6 SVO `d_F=8` fits are identical to every printed digit;
  - the 9 LIST fits differ, from GPU nondeterminism in training. E and C_dir are 0 in both runs, but the per-fit
    gain moves (by up to 63.5 → 45.9 in one fit).
- **No verdict depends on which run is used.** With the stalled run's LIST values, LIST's H2 median gain would be
  ≈ 46.6 instead of 44.0, still ≥ 10. H1 involves only SVO `d_F=32` cells, which the stalled run never reached.
- **The same cause explains #139's 9-hour gap.** The journal shows a 22:40 → 07:24 suspend, so #139's note that
  "the host most likely slept" was correct.

## Verdict

**H1 (the directional radius explains exact clean-up): FAIL** — 0 of 3 eligible cells.
**H2 (the tightening is substantial): FAIL** — 1 of 3 settings.

- The eligible cells (E ≥ 0.2) are SVO `d_F=32` with `mse` (E = 0.82), `cert` (0.26) and `t6` (0.80). In all three,
  `ρ_dir` certifies **none** of the exact recoveries: S = 0.
- The median gain `ρ_dir/ρ` is 44.0 on LIST `d_F=8` (passes), 7.8 on SVO `d_F=8`, and 5.8 on SVO `d_F=32`.
- `dir`, `wc` and `full` coverage are 0 in all 27 fits.
- All three abort checks held: certified ⇒ exact; `full` ⇒ host agrees; `ρ_dir ≥ ρ`. With nothing certified, the
  first two held trivially; `ρ_dir ≥ ρ` held on every context. Soufflé and Python agreed on every context. No fit
  was flagged.

## Per cell (seed mean [min, max]; 3,600 test contexts)

| setting | objective | E (exact) | S | gain `ρ_dir/ρ` | median `‖n‖` | median `ρ` | median `ρ_dir` | median `β` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LIST `d_F=8` | mse | 0.000 | — | 43.96 [36.43, 49.55] | 12.86 | 0.0005 | 0.0203 | 0.126 |
| LIST `d_F=8` | cert | 0.000 | — | 3.51 [2.86, 4.34] | 15.87 | 0.0058 | 0.0215 | 0.112 |
| LIST `d_F=8` | t6 | 0.000 | — | 45.23 [35.27, 53.29] | 12.65 | 0.0007 | 0.0284 | 0.123 |
| SVO `d_F=8` | mse | 0.023 [0.001, 0.055] | 0.000 | 8.80 [6.08, 11.18] | 8.97 | 0.0220 | 0.1484 | 0.086 |
| SVO `d_F=8` | cert | 0.014 [0.001, 0.023] | 0.000 | 5.32 [4.55, 6.24] | 11.67 | 0.0541 | 0.3013 | 0.142 |
| SVO `d_F=8` | t6 | 0.033 [0.000, 0.086] | 0.000 | 7.83 [5.56, 9.04] | 8.57 | 0.0267 | 0.1665 | 0.089 |
| SVO `d_F=32` | mse | 0.822 [0.770, 0.859] | 0.000 | 5.79 [5.15, 6.56] | 6.17 | 0.0726 | 0.4123 | 0.260 |
| SVO `d_F=32` | cert | 0.260 [0.164, 0.324] | 0.000 | 3.86 [3.62, 4.29] | 9.63 | 0.1368 | 0.5174 | 0.423 |
| SVO `d_F=32` | t6 | 0.800 [0.757, 0.857] | 0.000 | 7.09 [6.33, 7.98] | 6.20 | 0.0804 | 0.5576 | 1.034 |

`C_dir`, `C_wc` and `full` are 0 in every cell (not shown). Among `full`-uncertified contexts, the share with
`‖n‖ ≥ ρ_dir` and the share with `‖n‖ ≥ β` are both 1.00 in every cell. Medians are seed means of per-seed medians.

## Reading (interpretation)

1. **The directional radius is 3.5–45× the worst-case one, and still certifies nothing.** On SVO `d_F=32`, clean-up is
   exact for 80–82% of contexts (`mse`, `t6`) at `‖n‖ ≈ 6.2`. `ρ_dir` is only 0.41–0.56 there, so the observed
   noise is 11–15× the largest certified ball. On LIST the gap is about 450–740×.
2. **This is not looseness in the bound.** `ρ_dir` is exact per half-space (`directional_radius_tight`): along the
   worst rival's direction, noise of size just over `ρ_dir` does break clean-up. The clean-up region is a long
   polytope, and GPT-2's noise mostly points away from its nearest faces.
3. **So no ball certificate can explain the observed recovery,** whichever radius is used. What would is a
   per-context check of the half-space conditions, `⟨M_s n, d_a⟩ < ‖d_a‖²/2` for every role and rival: exact by
   `nearest_iff_halfspace`, and equal to E by construction. This is the same pattern as #135, where the exact
   pairwise substitution check certified what the uniform bound could not.
4. **`β` would bind next.** Even a `ρ_dir` large enough would leave `‖n‖ ≥ β` everywhere (median `β` ≤ 1.04).

## Not claimed

- A ball certificate's failure does not mean clean-up is unreliable: E is measured directly, and is high on SVO
  `d_F=32`.
- The per-context half-space check in point 3 is kernel-checked (`nearest_iff_halfspace`) but was not run as a
  pre-registered verdict here.
- GPT-2 small, two templated tasks, the TPR family only; LIST `d_F=32` is out of scope (no left inverse).
