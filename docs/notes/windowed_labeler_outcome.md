# Windowed Count-Table Labeler — Outcome (HALTED, cross-vendor; rung 1)

**Tag: `empirical`.** A pre-registered, cross-vendor-confirmed **negative** — feature-ladder rung 1.
Pre-registration: [`windowed_labeler_prereg.md`](./windowed_labeler_prereg.md). Predecessor:
[`biaffine_labeler_outcome.md`](./biaffine_labeler_outcome.md) (#119 — wall is representation; redirect to richer
features; rung 1 = the cheapest, most-certifiable rung).

## Setup
Keep the count-table (the estimator that won #119); give it **richer keys** — a window of POS + coarse surface
shape of the ±M neighbors of the dependent and ±1 of the governor, with the shipped unary key levels appended
**inline** and a **min-support floor** (T ∈ {5,10,20,50}) so rare windowed keys don't over-trust. Matched vs the
unary labeler on the **same** serve-honest L0 predicted heads (isolates labeling); same case cascade. Deterministic.
(M, T) tuned on DEV; TEST read once. Raced grok + codex.

## Corrective note (first pass discarded)
The first pass HALTED but was **contaminated by two defects** grok's own verification caught: (1) an OOV fallback
that passed a SLICED `head_offsets[index:index+1]` to `predict_deprels`, which re-enumerates from 0 → ~42% of
tokens keyed to sentence-position-0; (2) a bare count≥T threshold swept down to T=1, over-trusting single-sample
windowed keys. The corrected spec fixed both (inline unary levels; min-support floor T≥5), and added three
diagnostics + a **populated-table** regression test + a **forced-fallback integrity** assertion. The empty-table
test stubs of the first pass were exactly what hid the key bug.

## Result — HALTED, both lanes, cross-vendor
| arm | deprel-only | LAS-strict | serve-honest case | pairwise | local |
|---|---|---|---|---|---|
| unary | 0.696 | 0.427 | 0.741 | 0.418 | 0.920 |
| windowed | 0.691 | 0.421 | ~0.742 | 0.422 | ~0.91 |

**§5 read = HALTED** in both lanes: windowed−unary serve-honest LAS gain **−0.0052** (grok and codex identical),
case essentially flat. UAS identical across arms (matched heads).

## Diagnostic — it's the KEYS where they fire, not the backoff
- Windowed keys fire on only **~36%** of TEST tokens; the other ~64% correctly back off to unary. Backoff proven
  sound: codex's forced-fallback assertion (T=856 > max support → exact unary match, 0 mismatches); grok's DEV-vs-T
  curve flat (not flagged suspect).
- **Where the windowed keys fire — with genuine support — they are WORSE than the unary key on those same tokens:
  grok 0.853 vs 0.868, codex 0.848 vs 0.864.** The richer window is an *actively worse predictor* than the smoothed
  unary key precisely where it has enough support to be trusted. A genuine feature-quality finding, not a coverage
  or threshold artifact.

## Read
Cheap POS+shape abstract context does **not** help German case labeling → **HALTED → escalate to rung 2 (grounded
features)**, per the ladder and the prereg's HALTED branch. This *earns* the rung-2 investment rather than assuming
it (cheap-by-design: the cheapest rung was tried first and named the wall). Consistent with the domain-specific /
grounded-representation direction: abstract POS categories are too coarse; the constrained-meaning, grounded/learned
features are the next lever. Through-line: serve-honest case ~0.74 ≤ ~0.76 baseline (cascade unchanged, as expected
when labeling doesn't move).

## Open observations (not correctness concerns)
- grok: even at T=50 the DEV curve doesn't fully reconverge to the unary baseline (~0.687–0.689 vs 0.6916); a wider
  T grid ({100, 500}) would confirm asymptotic convergence. Not a defect (the forced-fallback assertion rules out a
  backoff bug); a natural follow-up only if this rung were revisited.
- The prereg's file pointer for `predict_deprels` was stale (it lives in `attach_levers_codex.py`, not
  `german_r3_codex.py`); corrected in the prereg. Both lanes found the right location.

## Artifacts
Cross-vendor pair (both retained as the robustness record): `experiments/windowed_labeler_codex.py`,
`experiments/campaign_windowed_labeler.py`; tests `tests/test_windowed_labeler_codex.py`,
`tests/test_campaign_windowed_labeler.py`. Generated scoreboards under `data/` (gitignored).
