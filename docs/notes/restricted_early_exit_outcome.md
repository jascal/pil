# Certified Early Exit on Restricted Decisions — Outcome: HALTED

Pre-reg: [`restricted_early_exit_prereg.md`](./restricted_early_exit_prereg.md) (SIGNED 2026-09-25).
- **Decisions:** built with `rorshopping/parallel-decisions` @ `45820320` (its own `build_prompt` + field
  compilation). One enum field `location`, 6 options, bAbI qa1–qa3 contexts. CAL 150, EVAL 300, plus EVAL
  re-run with the gold option first and last.
- **Capture:** fieldrun `--source-dump --tail 1` at `cb37079` (fieldrun #135).
- **Numbers:** `results/restricted_early_exit.{txt,json}`.

## Verdict: HALTED `[empirical]`
Restricting the certificate to the field's 6 options certifies **0/300** EVAL decisions at every exit layer, under
all three bounds, **including the exact suffix norm** (oracle). The check cost that capped #126 is gone (6 options
cost 0.00036 layer-equivalents). But the radius the certificate needs did not grow enough:

| exit `k` | median restricted radius | median suffix norm | median suffix / radius (`t_k` = final) |
|---|---|---|---|
| 12 | 0.10 | 29.9 | ~290 |
| 20 | 0.14 | 29.9 | ~160 |
| 21 | 0.34 | 29.8 | ~60 |
| 22 (one layer left) | 0.35 | 26.3 | **~56** |

For comparison, #126's full-vocabulary radius at `k = 22` was 0.21. Restricting to 6 options raised it only about
1.7×, because on this task the options are **close rivals**. The model is often unsure (see accuracy below), so the
winning option's lead over its nearest alternative is small. The late write is unchanged: about as large as the
whole residual.

## Controls: all pass
- `s`-recovery residual max `6.3e-6`; fieldrun reconstruction 1.00 on all four dumps.
- fieldrun #135's `--tail` reproduces earlier dumps byte-for-byte without the flag, and matches full dumps at the tail.
- No option collisions (first tokens `b`, `bed`, `g`, `hall`, `k`, `office`). Note that "bathroom" begins with the
  generic token `b`.
- **Engine agreement:** the library's own Torch fp16 decision matches fieldrun int8's restricted decision on
  **92.7%** of EVAL. The certificate question is therefore about essentially the same decisions the library makes.
- Soundness gate: 0 violations (trivially).

## Headroom and accuracy (descriptive)
- **Uncertified headroom** is larger than #126, as predicted: mean **3.0 layers** skippable (median `k★` = 22), up
  from about 1.1–1.3. A 6-way choice settles earlier than a 152k-way one, but only after mid-depth: the prefix
  choice equals the final choice for just 21% of decisions at `k = 12` and 64% at `k = 22`.
- **Accuracy vs gold is low:** qa1 **0.45**, qa2 0.34, qa3 0.34 (chance ≈ 0.17). The 0.5B model is a weak
  decision-maker on this task in this prompt format, which is also why the options sit close together.

## Option-order bias (descriptive)
- **No bias on this task:** accuracy is 0.373 with the gold option listed first and 0.377 with it listed last. The
  README's 96.7% vs 24.5% gap (7B, harder Choice questions, a different evaluation) does **not** reproduce here.
- The per-block order effect is still concentrated **late and in attention**. The top-5 blocks carry 69% of the
  mean effect (L22.attn +0.12, L20.attn −0.09, L21.attn +0.04, L23.mlp −0.03, L16.attn +0.02). They partly cancel,
  which fits the small net accuracy difference.

## Predictions vs outcome
| prediction | outcome |
|---|---|
| weight-derived bound vacuous | ✓ |
| oracle HALTED or IN-BETWEEN (least confident) | ✓ **HALTED** |
| larger uncertified headroom (3–6 layers) | ✓ 3.0 (bottom of the range) |
| qa1 accuracy > 80%, qa2/qa3 lower | ✗ — qa1 0.45 |
| order bias visible but smaller than the README's | ✗ — no bias (0.373 vs 0.377); effect is late-attention, as predicted |

## What this adds to #126 / #127
Restriction removed one of #126's two obstacles (the check cost) and not the other (the radius). On
Qwen2.5-0.5B, certified early exit now fails in three settings: full-vocabulary norm bounds, J-lens bounds, and
restricted-option norm bounds. The common cause is the same each time: the last layers write a vector about as large
as the whole residual, while the decision's lead is roughly 1–2% of that.

**Where a positive result could still come from:**
1. **A model that decides confidently.** A larger model, or a task this model gets right, would have larger
   restricted leads. The radius, not the cost, is the bottleneck.
2. **Directional suffix bounds on the option subspace.** With only 6 option directions, the certificate needs the
   suffix's projection onto 5 difference vectors `w_t − w_v`, not its full norm. A per-option-pair calibrated
   bound is the natural next test, and it is cheap on these same dumps.

## Scope
Qwen2.5-0.5B-Instruct (fieldrun int8), one 6-option field, bAbI qa1–qa3, 300 EVAL decisions. Single-position
decodes only (collision-free).
