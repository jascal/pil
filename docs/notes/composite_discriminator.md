# Composite discriminator (slice #88) — one shot, plateau; the coupling thread closes

**Status:** measured (2026-07-12), registered rules applied verbatim (workspace prereg
log). The single authorized attempt at the multivariate discriminator #87 priced; steering
fixed the bar and ended the thread either way. `experiments/campaign_composite_discriminator.py`
(+10 tests). Measurement only; frozen B0′; features FROZEN to the four #87 axes; no
iteration. sklearn absent → torch L2-logistic (Adam lr 0.1, 500 steps, C=1), disclosed.

## Registered outcome: PLATEAU — both clauses fail

| clause | bar | measured | pass |
|---|---|---|---|
| 1 · discriminability | CV median AUC ≥ 0.70 **and** beats permutation null (p<0.05) | AUC **0.618** (min 0.602 / max 0.652 over 20 repeats); null median 0.491; **p = 0.037** | **NO** (AUC fails; null-beat passes) |
| 2 · rescue | a zero-regression-admissible operating point exists | val sweep 100 points → **0 admissible**; #86 grid replay 0/50 (val, test untouched) | **NO** |

The composite clears the *null* (weak signal is real, consistent with #87's four ~0.60
axes) but not the *bar*: four weakly-separable axes compose to AUC 0.618 — barely above the
best single axis, nowhere near 0.70. With no discriminator that separates gain from
regression rows well enough, no filtered operating point rescues an admissible point at
either #82's frontier or #86's grid. Half-B eval was correctly skipped (nothing selected);
#86's test_te stayed untouched (the rescue claim there is val-level, flagged).

## What this closes, and the contract answer it forces

The routing/harvesting thread (#81 → #88) is **closed**: the wikitext routing gain
(+0.0088, p≈5e-08, real) is **not safely extractable** at any measured granularity —
claim-time gating (#82), admit-time key-retreat (#83), mine-time thresholds (#86), and now
a multivariate discriminator over the coupling's own measured axes (#88) all fail the
zero-regression bar. The coupling is **stage-invariant and not separable at these four
axes**; #87 located it (boundary-concentrated) but #88 shows the location's axes don't
carry enough signal to sort paying rows from breaking rows.

**Contract corollary (registered, now triggered):** zero-regression trusted-tier growth is
not achievable at this granularity by any measured mechanism; the package contract should
**price an explicit regression budget** next to the certificate — the field specced in
#87's registration (`{regression_budget: {rate, eval_unit, measured_on, sweep@sha,
domain}}`), recommended for the next package-emit slice. Achievability stays **open**: a
richer feature family than the four axes is untested (but not to be chased per steering —
"weak axes composite to weak discriminators," now measured, not asserted).

## Tags

| Claim | Tag |
|---|---|
| composite CV AUC 0.618 < 0.70 (beats null p=0.037) | **empirical** (20-repeat CV + 1000-shuffle null) |
| 0 zero-regression-admissible operating points (val sweep + #86 replay) | **empirical** |
| routing gain not safely extractable at any of four measured granularities (#82/#83/#86/#88) | **empirical** (four registered negatives) |
| contract defaults to an explicit regression budget | **empirical** (registered corollary) |
| a richer feature family clears the bar | **open** — not to be chased (steering); recipe plateau, achievability open |

## Honesty notes

One shot, no iteration (steering); features frozen; all seeds registered (CV 0..19, null
42, split 0). Clause-1 CV models and clause-2's half-A model are distinct trainings backing
separate claims (discriminability exists / it rescues a point) — stated. Torch-logistic
substituted for absent sklearn (fixed recipe, registered). Frozen B0′; WYLY_SEED=0; v5
untouched (repro invariant not triggered). Fourth consecutive registered non-definitive
result in this arc — the discipline recording thin signal at true size, by design.
