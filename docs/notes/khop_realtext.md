# khop-in-real-text — DEAD: natural text holds no recoverable 2-hop chains (cross-vendor)

**Status:** measured + registered, **cross-vendor confirmed** (2026-07-13). B1 of the lead's post-arc menu
(the session owner's standing "generalize khop to natural text" steer). Pre-registered in
`PIL_KHOP_REALTEXT_PREREG.md` (SIGNED before numbers). `experiments/campaign_khop_realtext.py` (grok) +
`experiments/khop_realtext_codex.py` (codex), independent lanes (raced). **Verdict: DEAD — decisive.**

## What was tested
Does natural wikitext hold recoverable **2-hop chains** (predict C from context A via a bridge B: A→B→C,
bridge-site exclusion i≠p1+1) that a **1-hop / memorizer baseline MISSES**? Reused the certified #95 khop
chain unchanged; swapped ONLY the query gate — synthetic `[lo,hi]` → real-text **bridge-set membership**
(mined from the FIT split). Baseline = per-token `max(1-hop bigram, memorizer best-per-key)`.

## Result — DECISIVE DEAD, cross-vendor
| lane | recovery | agree_2hop | agree_baseline | n_fired | bridge-set | ablated recovery |
|---|---|---|---|---|---|---|
| grok  | **−0.1107** | 0.031 | 0.142 | 3369 | 15,591 | −0.1419 |
| codex | **−0.1092** | 0.031 | 0.141 | 3408 | 20,240 | −0.1209 |

`recovery` (bar ≥ +0.005) is ~20× BELOW the bar and NEGATIVE in both lanes: the 2-hop rule (~0.031) is far
WORSE than direct lookup (~0.14). It is right-where-baseline-wrong only ~0.9% of the time, wrong-where-
baseline-right ~12%. On natural text, chaining A→B→C **compounds error** and loses badly to a memorizer.
The population is healthy (n≈3.4k, ~29% of held-out, well above the STOP floor). Both lanes independent
(different bridge sets — 15,591 vs 20,240 — settling any mining-sensitivity concern: two mining choices,
same decisive DEAD).

## Confound guard — bridge-ablation
The recovery is already NEGATIVE, and under a wrong bridge `agree_2hop` collapses to ~0.0003 (grok) —
`recovery_ablated` goes MORE negative. There is no survive-vs-vanish ambiguity: the guard "holds"
trivially because there is no positive recovery to preserve. The 2-hop rule is simply a worse predictor
than direct lookup, and a worse-still one with a wrong bridge.

## What it means
Natural-text khop does NOT generalize — the certified 2-hop chain was a **synthetic-vocab convenience**
(the `[lo,hi]` gate). On real text the chaining is a net-negative predictor vs direct lookup + memorization.
**No natural-text khop build** (per the prereg's DEAD branch). A clean registered negative — the synthetic
khop win (#95) does NOT transfer, exactly the "synthetic wins are not real-text evidence" discipline.

## Honest scope + tags
| Claim | Tag |
|---|---|
| natural wikitext holds no 2-hop chains a 1-hop/memorizer baseline misses (recovery −0.11, cross-vendor) | **empirical** — DEAD |
| the DEAD is robust to the bridge-mining threshold | **supported** — two independent minings (15,591 / 20,240), same result |
| khop's chaining is a WORSE predictor than direct lookup on real text (error compounds) | **empirical** — agree_2hop 0.031 vs baseline 0.14 |
| khop is useless in general | **NO** — the CERTIFIED synthetic khop (#95) still holds over ITS stated domain; this only says the chaining doesn't earn its keep on natural text under this protocol |

## Process
Raced grok + codex from scratch (independent generators + bridge-mining + baselines). grok caught + fixed
its own vocab-dimension bug (a corrupted memorizer baseline) mid-run and reproduced bit-for-bit; codex
independently confirmed the same DEAD with a different bridge set. Both flagged the shared-working-tree
noise from the concurrent gated-beam lane (verified unrelated — each deliverable clean in isolation). See
[[lane-balancing-rule]], [[khop-package-emission-outcome]].
