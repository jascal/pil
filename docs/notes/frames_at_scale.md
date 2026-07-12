# Mined frames at scale (slice #86) — the signal exists at 2.8b; the coupling recurs at the mining layer

**Status:** measured (2026-07-11), registered rules applied verbatim (workspace prereg
log). Steering item 1, candidate (a): was the judge's 2.8b decline of mined frames a
SCALE fact or a MINER fact? `experiments/campaign_frames_at_scale.py` (+8 tests; the
parameterized miner is line-traced against `MinedGates.mine()` and reproduces it exactly
at unmodified thresholds — unit-tested). Substrate reconstruction faithful: both scales'
`_cov_ol_sw` states re-scored **bit-identically** to their stored payloads (410m 0.2883,
2.8b 0.2701), and the historical outcomes reproduce from the real sleep-loop EMIT_INFO
(410m: 4 frames admitted; 2.8b: 3 mined, declined).

## Registered outcome: NO ZERO-REGRESSION SELECTION (test never touched)

Fixed 3×5 grid over the miner's (support × interaction) gates at 2.8b, judge bar
untouched (5e-4). **5 of 15 grid points clear the judge's admission bar** — including one
at the *current, unchanged* support threshold (15) with only the interaction gate lowered
0.25 → 0.109, val marginal +0.000736. But **every judge-clearing point carries 2–7 val
regressions**, so zero points pass the registered no-regression-excess selection filter;
per the registration, no test evaluation occurred (`test: null`, spy-asserted).

Verdict-branch bookkeeping, honestly: the registered SCALE-FACT branch ("no grid point
yields an admission") is **factually unmet** — admissions exist; the MINER-FACT branch
requires a test eval that never legally happened. The registration had an unnamed branch
(admissions exist but all regression-positive); the lane created the conservative named
outcome above rather than forcing either registered verdict. No flip; the gap is
disclosed.

## What the anatomy shows (Part A, 410m reference vs 2.8b)

- Error pools are nearly identical in size (~66% vs ~67% of sampled rows) and the miner's
  candidate volumes match across scales (≈1,700–1,800 singleton candidates per offset).
- **The interaction-score gate is the binding gate at BOTH scales** — support-surviving
  candidates number ~25–31 per offset, and the interaction gate then kills essentially
  all of them (top-decile interaction scores barely reach the 0.25 default anywhere).
- So the scale story is not "2.8b's residual lacks frame candidates"; it is "the
  interaction threshold sits above where 2.8b's (and marginally 410m's) candidates live."

## The finding that matters most: the coupling is stage-invariant

**Frame-shaped signal exists at 2.8b** (existence-wise, the decline is a miner fact), but
revealing it imports regressions — the same gain/regression coupling now measured at
**three different pipeline stages**: claim time (#82: the two-sided gate frontier — safety
keeps ⅓ of the gain), admit time (#83: key-retreat defeated by gain/leak entanglement at
the same keys), and now **mine time** (#86: every threshold relaxation that clears the
judge also carries regressions). Three registered safety criteria, three stages, one
recurring shape. This is no longer a per-slice nuance; it is a measured property of how
marginal signal presents in this cover's residual: **new signal at the margin arrives
entangled with regressions on previously-correct rows, wherever in the pipeline you try
to harvest it.** (Candidate explanation, untested: near the cover's frontier, the rows a
new rule can claim are exactly the rows where the incumbent's confidence is least
informative — both directions at once. Open.)

## Caveat (scope limitation, not a defect)

The historical process mines across ~8 sleep episodes with a persistent store; the sweep
is necessarily **single-shot** against the frozen final model. Single-shot at default
thresholds on 410m finds 2 frames and does not clear the bar, though the multi-episode
history admitted 4 — so all sweep numbers are **lower bounds**; the 15-point comparison
is internally valid (shared methodology). Priced follow-up: a multi-episode sweep;
trigger: this family being needed at scale for the ladder story.

## Tags

| Claim | Tag |
|---|---|
| substrate reconstruction faithful (bit-identical re-scores; history reproduced) | **empirical** |
| frame-shaped, judge-clearing signal exists at 2.8b; the interaction gate (not support) hides it | **empirical** (5/15 grid points; single-shot lower bound) |
| registered outcome NO ZERO-REGRESSION SELECTION; test untouched | **empirical** (verbatim; prereg gap disclosed) |
| the gain/regression coupling recurs at mine time (third stage: #82, #83, #86) | **empirical** (pattern across three registered designs) |
| why the coupling is stage-invariant | **open** (frontier-rows hypothesis, untested) |
| multi-episode sweep clears the zero-regression bar | **open** (priced; lower-bound argument) |

## Honesty notes

Miner thresholds swept; the judge's bar untouched throughout. Single test-eval discipline
preserved by never reaching test. Frames are `template_fixed`-class; `frac_induced`
unaffected. WYLY_SEED=0 pinned (the #85 knob); v5 untouched (repro invariant not
triggered).
