# Grounded Relation-Labeler — Outcome (feature-ladder rung 2, slice 2b)

**Verdict: IN-BETWEEN (cross-vendor, verified). Tag: `empirical`.** Grounded-φ beats surface (#119) and
lifts labeling + case, but misses the pre-registered +0.03 strict-LAS FIRES bar — and the miss is
**head-gated**, not a representation failure. Governed by [`grounded_labeler_prereg.md`](./grounded_labeler_prereg.md)
(§5 = law). Predecessors: [`biaffine_labeler_outcome.md`](./biaffine_labeler_outcome.md) (#119, surface-φ HALTED)
and [`windowed_labeler_outcome.md`](./windowed_labeler_outcome.md) (#120, cheap windows HALTED). **This is the
first rung on the post-#119 ladder to move the needle.**

## What ran
The #119 biaffine relation-labeler with `φ` = per-word **grounded residual** from Qwen2.5-3B-Instruct (dump 2a,
`data/grounded/qwen3b_gsd_{dev,test,train}.npz`, 4 layer-checkpoints [8,17,26,35]), measured serve-honest on the
**doubly-covered arc set** (both endpoints grounded) vs the UNARY count-table AND surface-φ #119, same L0 predicted
heads, DEV-only layer sweep, TEST read once. Cross-vendor race: grok (`experiments/campaign_grounded_labeler.py`)
+ codex (`experiments/grounded_labeler_codex.py`). Both lanes verified by the architect: ruff clean, 25 tests pass,
both scoreboards reproduce byte-identical.

## The numbers (both lanes, doubly-covered arcs; UAS identical across arms — asserted)
| metric | grok unary → grounded | codex unary → grounded |
|---|---|---|
| strict-LAS gain **vs unary** | +0.0195 | +0.0175 |
| gain **vs surface-#119** | **+0.0539** | **+0.0560** |
| deprel-only (label) gain vs unary | **+0.0613** | **+0.0761** |
| pairwise-sensitive gain | **+0.171** | **+0.196** |
| local gain | +0.012 | +0.025 |
| serve-honest CASE (unary → grounded) | 0.488 → 0.624 | 0.519 → 0.596 |
| DEV-selected layer | 17 (mid) | 26 (mid) |
| doubly-covered fraction (dev / test) | 0.718 / 0.632 | 0.645 / 0.567 |

Per §5 (mechanical: gain > 0 but < +0.03 → IN-BETWEEN): strict-LAS +0.0195 / +0.0175 both under +0.03 → **IN-BETWEEN**.
No discretion; both lanes agree. Not diffuse (pairwise ≈ 8–10× local), case DID improve, beats #119 — the sole miss
is the +0.03 strict-LAS threshold.

## Why the strict-LAS miss is head-gated (from shipped numbers, no new read)
strict-LAS = head ∧ label. Both arms share the same weak L0 heads (UAS ~0.49). Grounding strongly improves the
**label**: deprel-only gain **+0.061 / +0.076** (more than double the FIRES magnitude). But LAS only credits a right
label where the head is also right, so the label win half-converts.
- **On the correct-head subset** (LAS/UAS): grounded **0.905 / 0.884** vs unary **0.865 / 0.848** → **+0.040 / +0.036** —
  above bar-magnitude even where heads are right.
- **Honesty caveat** (Fable, keeps the projection sober): the deprel-only gain (+0.061/+0.076 overall) is *larger* than
  the correct-head-conditional gain (+0.040/+0.036) — i.e. grounding helps **most on the arcs whose L0 head is wrong**
  (the pairwise-sensitive core args). So a naive "fix heads and the +0.06 converts to LAS" **overstates it**; the
  realistic conversion is ≈ the +0.04 conditional figure.
- **Lane divergence (recorded, not papered over):** the lanes DEV-selected different mid-layers (17 vs 26) and measured
  slightly different doubly-covered sets (codex's stricter → smaller). Qualitatively identical; the exact best layer is
  not pinned. A layer/coverage-definition sensitivity to carry into the next slice.

## Disposition (per §5 + Fable-checked)
IN-BETWEEN → **not** the HALTED disposition, so **not rung 3 (R4 LLM-hybrid)**. Grounding clearly helps (> unary,
beats #119); R4 would be premature. Also **not 2c harden** — §5 says "2c iff 2b FIRES"; hardening a non-fired scorer is
exactly the drift the prereg exists to block. And **not a re-registration on a head-controlled metric** — the
deprel-only / correct-head-conditional numbers above are already in the shipped JSONs and are reported here as
diagnostics; moving the primary metric post-hoc is goalpost-shifting.

**Next lever (recommended, pending sign-off): an attach primitive over grounded φ** — test whether grounding also helps
**HEADS** (the actual bottleneck). If UAS rises, the label win converts to LAS and the through-line lifts. This is the
lead-adopted axis-B bilinear attach direction ([[german-r3-outcome]]), stays cheap-by-design
([[cheap-by-design-no-reflexive-nn-fallback]]), and attacks the diagnosed bottleneck rather than the symptom.
- **The risk that decides it — coverage.** Test doubly-covered fraction is only 0.57–0.63; heads must be predicted for
  *all* arcs, so uncovered arcs fall back to L0 and dilute any serve-honest UAS gain by ~40%. Pre-register the attach
  gate on the covered set with the diluted full-set number reported alongside.
- **Consider first:** MWT-alignment recovery ([[mwt-aware-alignment-todo]]) directly raises this slice's coverage
  ceiling (~28% dev loss is MWT-driven) — may be worth folding in before the attach gate rather than after.

## Tag discipline
`empirical`: grounding improves German relation-labeling (deprel-only +0.06–0.08, pairwise +0.17–0.20, case +0.08–0.14,
beats surface-#119 +0.055), cross-vendor, verified. `open`: whether that labeling win converts to a serve-honest
strict-LAS / case gate above bar — head-gated, answered only by the attach lever. No `proved` claim: this is a soft
grounded scorer, not yet a certified lookup (2c is not triggered).
