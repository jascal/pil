# Restricted-Decision Early Exit at Scale, with a Directional Bound — Outcome

Pre-reg: [`restricted_early_exit_scale_prereg.md`](./restricted_early_exit_scale_prereg.md) (SIGNED 2026-09-25;
Addendum A 2026-09-26, before any 3B number: 7B deferred with the user's approval, and the Gram spectral norm).
- **Data:** the #129 decision prompts (parallel-decisions @ `45820320`, 6-option `location` field, bAbI qa1–qa3,
  CAL 150 / EVAL 300).
- **Capture:** fieldrun `--tail 1` at `3e38dfe` on CPU int8.
- **Numbers:** `results/restricted_scale_{0.5b,3b,trend}.{txt,json}`.

## Verdicts `[empirical]`
| model | Arm A (norm bounds) | Arm D (directional) |
|---|---|---|
| Qwen2.5-0.5B-Instruct | **HALTED** | **D-IN-BETWEEN** |
| Qwen2.5-3B-Instruct | **HALTED** | **D-IN-BETWEEN** |

**Soundness gate PASS on both** (0 violations for oracle, weight and Oracle-D). Controls: `s`-fit residual
≤ `2.2e-5`; fieldrun reconstruction 1.00; the tokenizer-identity control passed (3B/7B encode all 450 prompts
identically); **library fp16 vs fieldrun int8 agreement 92.7% (0.5B), 93.7% (3B).**

## The headline: direction is what the certificate was missing
| | 0.5B | 3B |
|---|---|---|
| Arm A oracle coverage (exact suffix **norm**) | 0% | 0% |
| **Oracle-D coverage** (exact **adverse push**) | **58%** | **89%** |
| Oracle-D median exit layer / layers | 14 / 24 | **9 / 36** |
| Oracle-D mean layers skipped | 7.1 | **19.8** |
| Calibrated-D coverage (violations) | 3.3% (0) | **32% (0)** |
| Calibrated-D median exit / net saving | 22 / +0.03 | 34 / +0.39 |
| median suffix-norm / radius, last exit | 56 | 53 |
| **median adverse push / radius, last exit** | 1.8 | **0.31** |

- **The size of the remaining write is irrelevant; its direction decides.** The suffix norm stays 50+× the radius
  at both scales. But its push *against the rivals* is only 1.8× the radius at 0.5B, and **0.31×** at 3B: already
  below the certifiable threshold of 1 at the median.
- **Why D-IN-BETWEEN and not D-FIRES at 3B.** Calibrated-D clears the coverage bar (32% ≥ 25%) and the
  violation bar (0 ≤ 10%), and fails only the **net-saving** bar (+0.39 < 1 layer). The 95% conformal quantile of
  `push/‖y_k‖` is loose at early layers, so it certifies only near the end (median exit 34 of 36). The oracle shows
  early exit at layer 9 is available. **The calibration, not the certificate form, is now the bottleneck.**

## Descriptive and exploratory reads (not decision variables)
- **Accuracy did not rise with scale:** mean 0.377 at both (qa1 0.45 → 0.42). This contradicts my prediction (qa1
  > 80% at 7B, and implicitly higher at 3B). In this prompt format both models are weak bAbI readers.
- **0.5B partly collapses; 3B does not.** 0.5B chooses `office` 46% of the time, and its Oracle-D coverage is
  concentrated in some answers (bathroom 7%, kitchen 0%). 3B's choices spread across all six options (≤ 22% each),
  with Oracle-D 65–100% within every answer. So **3B's early certification is not a constant-answer artifact.**
  Certified and uncertified decisions have similar accuracy (0.38 vs 0.31 at 3B).
- **3B leans on recency more:** on qa1 it picks the last-mentioned location 47% of the time (0.5B: 21%).
- **Late layers wander, then restore.** At 3B the prefix answer equals the final answer for 89% of decisions by
  about layer 9, yet only 34% at the third-to-last exit and 87% at the last. The late layers temporarily rotate the
  restricted read-out and the last write restores it. So the uncertified `k★` ("stays final from here on":
  2.5–3.0 layers) **understates** how early a correct and certifiable exit exists (Oracle-D: 7–20 layers).
- The two-point comparison (0.5B vs 3B) is directionally clear on Oracle-D, Calibrated-D and push/radius, but with
  two models it cannot fire the pre-registered monotone-trend rule (Addendum A).

## Predictions vs outcome
| prediction | outcome |
|---|---|
| Arm A stays HALTED; suffix/R shrinks with scale | ✓ HALTED; suffix/R ~flat (56 → 53) |
| Oracle-D ≥ 25% on every model | ✓ (58%, 89%) |
| D-FIRES plausible at 3B; D-IN-BETWEEN at 0.5B | ½ — 3B is **D-IN-BETWEEN** (fails only net saving); 0.5B as predicted |
| accuracy rises strongly with scale | ✗ flat |
| headroom grows with scale | ✗ uncertified `k★` 3.0 → 2.5, but certifiable depth grows (Oracle-D 7 → 20 layers) |

## Process notes
- The prereg's CPU timing estimate was about 7× too optimistic (3B: 106 s per decision under memory contention).
  7B was deferred (Addendum A) and never dumped.
- `runs/skip7b_watch.log` has a misleading first line, "03:31 killed 7B fieldrun pid 191765". That was one of the
  architect's own shells, killed by an unanchored first version of the watcher. No 7B process existed then. The
  anchored watcher then killed the two real 7B starts at 08:11.

## Consequences
1. **Arm A (norm-bounded early exit) is closed** at 0.5B and 3B.
2. **The live lead is a better-calibrated directional bound.** The per-`k` marginal quantile of `push/‖y_k‖` wastes
   the early layers where the oracle already certifies. Natural next tests, each needing its own prereg:
   - calibrate per exit layer against the radius-relevant scale, rather than `‖y_k‖`;
   - **Mondrian / conditional conformal** by margin or by task;
   - a **one-look P-single** policy chosen on CAL.
3. **7B** remains worth doing on a larger machine, under this prereg. The 0.5B → 3B change in Oracle-D (58 → 89%)
   and push/radius (1.8 → 0.31) makes it the most informative next rung.

## Scope
Qwen2.5-0.5B/3B-Instruct (fieldrun int8, CPU), one 6-option field, bAbI qa1–qa3, 300 EVAL decisions per model.
Arm D is empirical by construction: no sound input-independent bound on a projection exists.
