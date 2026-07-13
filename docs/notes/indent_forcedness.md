# Indent-forcedness probe — DEAD; computed-route-on-code PLATEAUED at register n=1 (slice #99)

**Status:** measured + registered, **cross-vendor confirmed** (2026-07-12). Lead-approved
(steering 19:36): trigger probe first, build deferred. Pre-registered in
`PIL_CODE_REGISTER_PREREG.md` ("Indent-forcedness probe") BEFORE numbers — rule, thresholds
(FIRES ≥ 0.5 of class / DEAD < 0.2), ceiling, and framing fixed there.
`experiments/campaign_indent_forcedness.py` (+ tests; + `experiments/verify_indent_forcedness.py`,
codex's independent cross-check). **This slice CLOSES the code-register exploration (#97–#99).**

## Result: DEAD, cross-vendor
The #98 anatomy left `whitespace_indent` (0.104 of residual) as the ONE live register-#2
candidate. This probe measured how much of that 531-row class a simple indent-continuation rule
(same-depth / +1 after opener / −1 before closer / file-local unit, via certified `depth_feature`)
recovers exactly.

| implementation | structural gate_class | naive baseline | verdict |
|---|---|---|---|
| grok (probe) | 0.102 (54/531) | 0.198 (105/531) | DEAD |
| codex (independent, stricter causal) | **0.075 (40/531)** | 0.115 (61/531) | DEAD |

- **DEAD, robust:** both structural gates well below the 0.2 line (0.102, 0.075); codex's stricter
  left-context-only implementation finds LESS, not more — so the verdict is not
  operationalization-sensitive at the boundary. gate_abs ≈ 0.008–0.011 of total residual (far below
  the ~0.104 ceiling).
- **The structural machinery is net-NEGATIVE:** the naive "copy the previous line's indent" baseline
  (0.198 / 0.115) BEATS the depth/opener/closer rule — indent isn't structurally forced; it's
  copying. Even the generous naive rule can't clear 0.2.
- Control not statistically separable (moot at DEAD). `n_class = 531` reproduced exactly by both.

## The honest framing (registered)
**C/C++ indent is convention-consistency, NOT grammar legality** — a file-local STYLE deriver, a
different kind of object from the sudoku legality register (grammar-forced) or bracket-mate. And it
doesn't force: even trivial copying, the ceiling of simple indent prediction, is < 0.2.

## Conclusion: the arc closes as a PLATEAU
The computed-route-on-code is **PLATEAUED at register n=1** (recipe plateau; achievability open —
NOT "impossible"). Arc:
- **#97:** bracket legality (the cheapest, already-certified register) does not recover code
  residual (DEAD, cross-vendor).
- **#98:** residual anatomy — code residual is 50% identifier prediction (semantic,
  register-immune); `whitespace_indent` (0.104) the one live candidate (ESCALATE, thin).
- **#99:** indent forcedness — the candidate is DEAD (cross-vendor); indent is convention, not
  legality.
→ The sudoku legality register (#92) remains the SOLE certified compute-register. Code does not
yield a second one at the token level. Move to items 2–3 (SequenceEnergyScore arc; uniform-reveal
sudoku).

## Tags
| Claim | Tag |
|---|---|
| indent-continuation rule recovers < 0.2 of the whitespace class | **empirical**, cross-vendor (0.102 / 0.075) |
| indent is convention-consistency, a file-local style deriver, NOT grammar legality | **framing**, committed |
| computed-route-on-code has no register beyond n=1 | **recipe plateau; achievability open** — NOT "impossible" (a richer/different register form is untested) |
| the structural machinery beats a naive copy baseline | **REFUTED** (naive beats structural, both lanes) |

## Process (lane balancing)
grok built the probe → codex independently confirmed (stricter causal impl, lower gate) —
[[lane-balancing-rule]]. Full arc: grok probe→codex verify (#97), codex anatomy→grok verify (#98),
grok probe→codex verify (#99). Both lanes on every load-bearing slice.
