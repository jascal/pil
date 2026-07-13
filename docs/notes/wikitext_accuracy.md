# Wikitext accuracy — REGRESSION: soft-margin lookahead hurts real text (cross-vendor, bit-identical)

**Status:** measured + registered, **cross-vendor bit-identical** (2026-07-13). The lead's directed
last-registered slice of the SequenceEnergyScore arc — **condition 3's soft-margin question in serving
form**. Pre-registered in `PIL_WIKITEXT_ACCURACY_PREREG.md` (SIGNED before numbers).
`experiments/campaign_wikitext_accuracy.py` (grok) + `experiments/wikitext_accuracy_codex.py` (codex),
independent lanes (raced). **Verdict: REGRESSION** — the M-step soft-margin beam HURTS real-text agreement.
A null was the pre-registered expectation; the measured result is *stronger than null* (a sign, not just an
absence).

## What was tested
Does energy-beam `DECIDE(cover="energy-beam", M>1)` change served token AGREEMENT on real wikitext vs the
**M=1 corner** (= bit-exact classic `serve_sw`), and at what LATENCY? On an **energy-mode** wikitext package
(emitted via `wyly_lm_v5.py WYLY_ENERGY_MODE=1` → the count-bearing kinds carry raw `(cnt,tot)` so the
beam's `det_rank` is a real soft-margin ranker, NOT a hash ranker — the load-bearing non-degeneracy gate).

## Result — REGRESSION, cross-vendor bit-identical
n = 11,477 held-out windows; M=1 corner agreement 0.3286, cover 0.9914.

| M | agree | Δagree | prune_bite | latency ×corner |
|---|---|---|---|---|
| 2 | 0.3136 | **−0.0150** | 0.0047 | ~2.1 |
| 3 | 0.3159 | −0.0126 | 0.0033 | ~3.7 |
| 4 | 0.3197 | −0.0089 | 0.0027 | ~6.4 |
| 5 | 0.3219 | −0.0067 | 0.0024 | ~10 |

**Both lanes returned these numbers BIT-IDENTICALLY** (agree/Δagree/prune to full float precision; only
wall-clock latency differs) — two independent implementations, each emitting its own deterministic
energy-mode package (`..._energy_pilot_grok` / `..._energy_codex`), each **non-degeneracy-verified**
(grok `frac_with_counts=1.0`, codex `0.99995`; `det_rank_differentiates=true` both, with concrete
differentiating examples). REGRESSION per the prereg (M=2,3 breach −0.01). `cover` is bit-identical across
all arms (no silent abstain-shift — the reading's parity precondition holds).

The beam sacrifices next-token accuracy for trajectory confidence: on soft tokens the next token IS the
target, so weakest-link-`det_rank` lookahead adds bias, worst at M=2 and recovering toward (never beating)
baseline as M grows — at 2–10× latency. `prune_bite ≈ 0.003` is NOT legality pruning (text has no hard
term; `Step.margin` is pinned 0) — it is the no-firing-rule DEAD-END case (empty `expand` counted as a
prune); a minor correction to the prereg's "expect exactly 0".

## What it means — the split-by-hard-term-availability decomposition, with a sign
| endpoint | verdict |
|---|---|
| ALL-HARD (sudoku legality, #107) | **FIRES** — lookahead + hard pruning commit gold per-token can't |
| ALL-SOFT (wikitext, this) | **REGRESSION** — soft-margin lookahead is net-negative |

So **the beam's value is the certified HARD term, not soft-margin lookahead.** Blanket M>1 on
soft-dominated real text is net-negative. This does NOT kill the beam — it defines HOW to deploy it: a
**hard-term-GATED beam** (fire M>1 only where the hard margin discriminates within M; else commit the M=1
corner) is ≥ greedy everywhere and > greedy on hard tokens — a free option whose gain scales with a
domain's hard-token density, i.e. exactly the bounded experts (family 4) the stack targets. Full design +
next experiment (a gated beam on a MIXED domain — code: hard brackets/indent, soft identifiers) is in
`WYLY_TRUNK_REASSESSMENT.md`'s what's-next escalation (user-flagged).

## Honest scope + tags
| Claim | Tag |
|---|---|
| M-step soft-margin lookahead HURTS real-text agreement (Δ −0.015 @M=2 → −0.007 @M=5, never positive), cross-vendor bit-identical | **empirical** — REGRESSION |
| the beam is a genuine soft-margin ranker (not a hash artifact) | **verified** — non-degeneracy PASS both lanes (frac_with_counts ≈1.0, det_rank differentiates) |
| the beam's value is the HARD term, not soft-margin lookahead | **supported** — the all-hard/all-soft endpoints (#107 FIRES / this REGRESSION) |
| a hard-term-gated beam would clear greedy on a mixture | **NOT shown** — motivated by the two endpoints, untested; the registered next experiment |

## Process
Raced grok + codex from scratch (independent generators + emit switch + agreement metric); bit-identical
results = the strongest cross-vendor confirmation. grok hit + handled the known grok-CLI edit-cancel issue
([[grok-cli-permission-recipe]]); both non-degeneracy-verified before the verdict counted (the soft-margin
analog of #107's `prune_bite>0` gate). See [[sequence-energy-direction]], [[beam-arc-followups]],
[[lane-balancing-rule]].
