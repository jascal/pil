# Biaffine Relation-Labeler — Outcome (HALTED, cross-vendor)

**Tag: `empirical`.** A pre-registered, cross-vendor-confirmed **negative**. Pre-registration:
[`biaffine_labeler_prereg.md`](./biaffine_labeler_prereg.md) (signed, committed pre-build; decision rules §5 fixed
before numbers).

## Setup
Axis-B test: does a gradient-fit low-rank **bilinear** deprel labeler beat the shipped **unary count-table**
labeler (`predict_deprels`), serve-honest, on GSD? Both arms label the **same** serve-honest L0-predicted heads
(UAS 0.484 — isolates labeling); identical `φ` (R1-predicted POS + hashed surface + shape + position) and case
cascade; biaffine fit by cross-entropy on gold arcs; ≥3 fixed seeds; TEST READ ONCE. Raced grok + codex on the
identical spec (independent parameterizations); architect verified both diffs + re-ran.

## Result — HALTED, both lanes, both parameterizations
| arm | heads | deprel-only | serve-honest case |
|---|---|---|---|
| unary | gold | **0.790** | 0.791 |
| biaffine | gold | 0.749 (codex) / 0.753 (grok) | 0.776 |
| unary | predicted | 0.696 | 0.741 |
| biaffine | predicted | 0.658 | ~0.47–0.72 |

Pre-registered §5 read = **HALTED**: mean biaffine−unary LAS gain ≤ 0 (codex −0.033, grok −0.031), **negative at
every seed**, and the pairwise-sensitive partition degraded **more** than local (opposite of the FIRES structural
condition). Numbers byte-identical-reproducible; the two independent implementations converge.

## Diagnostic — the wall is FORM/ESTIMATOR, not heads
- **Not the heads.** On *gold* heads the biaffine still loses to unary (0.749–0.753 vs 0.790). Fixing attachment
  does not rescue it — an early head-confound hypothesis was tested and refuted.
- **Not form-underpowering.** grok's *fuller* Dozat-Manning per-relation core **overfits more** (train−test gap
  **+0.151** vs codex's diagonal-core **+0.059**). Adding bilinear capacity memorizes more and generalizes worse.
- **Noise-matched training** (governor = predicted head) is indistinguishable — the train-gold/serve-predicted
  mismatch is not the confound.

→ A gradient-fit bilinear over surface features is simply a **worse estimator** than the smoothed count-table at
GSD's data size.

## Read
The count-estimated/gradient-fit bilinear labeler recipe **HALTS** (robust, cross-vendor). The form-level
(bilinear expressivity) claim stays formally **open** — the "unary" baseline is not truly unary (head-offset is a
pairwise, noise-robust feature), and `φ` is too weak for a bilinear form to beat a count-table estimator; biaffine's
literature wins ride on *contextual* encoders. So "biaffine loses here" ≠ "the pairwise rule type fails."

**Practically the wall is REPRESENTATION, not form or heads — a cheap-features problem.** Redirect (per
cheap-by-design discipline, no reflexive NN): the next move is *cheap richer/learned features* — a **windowed
count-table** (M-context, fully in the served tier), then **learnable abstractions hardened to a discrete lookup**
(`find→harden→certify` applied to the feature basis), measured against an NN only as a **yardstick we do not ship**.
Not a fuller bilinear (tested — overfits), not an NN forward pass.

## Artifacts
Cross-vendor pair (both retained as the robustness record): `experiments/biaffine_labeler_codex.py` (+diagnostic),
`experiments/campaign_biaffine_labeler.py`; tests `tests/test_biaffine_labeler_codex.py`,
`tests/test_campaign_biaffine_labeler.py`. Generated scoreboards under `data/` (gitignored).
