# Anisotropy-Removal (Whitening) Gate — Outcome (feature-ladder, TERMINAL)

**Verdict: HALTED (cross-vendor, verified) → R4.** The re-sweep confirms the plateau is robust to hyperparameters
(both lanes R4_CLEAN), so this is not a fixed-budget artifact — the cheap grounded-bilinear recipe **plateaus**, and
the whole grounded/axis-B feature ladder reaches its terminal negative. Tag: `empirical`. Governed by
[`whitening_gate_prereg.md`](./whitening_gate_prereg.md) (§5 + the two PINS = law). Predecessors:
[`grounded_labeler_outcome.md`](./grounded_labeler_outcome.md) (#121, IN-BETWEEN, head-gated) and
[`grounded_attach_outcome.md`](./grounded_attach_outcome.md) (#122, IN-BETWEEN, straddle).

## What ran
1. **The gate**: strip the ~73%-variance rogue dim from the grounded φ (V1 per-dim standardize; V2 top-k PC removal),
   DEV-swept, re-run BOTH the #121 labeler and #122 attach on the whitened MWT φ (richer, ~93–94% coverage), pinned
   effrank (PIN A) + pinned log-prob decode (PIN B). Cross-vendor race grok + codex.
2. **The re-sweep** (DEV-only mechanism diagnostic, user-approved before the R4 pivot): give raw φ AND whitened φ the
   SAME lr×epochs×rank grid, fit on TRAIN, evaluate on DEV — does best-tuned-whitened beat best-tuned-raw? Cross-vendor.

All architect-verified: ruff clean, tests pass, codex gate re-run byte-identical HALTED, both re-sweep runs agree.

## The gate result (both lanes: HALTED)
Whitening **worked geometrically** (PIN A: centered participation-ratio effrank raw ~1.1 → whitened **127** grok /
**70–155** codex — anisotropy removed, near-orthogonality achieved). But accuracy **did not follow**:
- LABELER LAS-strict: whitened below unary AND raw-#121 (Δraw ≈ −0.011/−0.015 both lanes).
- ATTACH UAS (PIN-B decode): whitened well below L0 AND raw-#122 (grok Δraw −0.085; codex Δraw −0.071).
Per §5 (whitening ≤ raw on both primitives) → HALTED, cross-vendor.

## The mechanism — CORRECTED (Fable), do NOT over-read
The naïve read ("anisotropy wasn't the blocker → grounded φ lacks the signal") is **refuted by our own partition**:
- **The pairwise-sensitive arc-signal is INTACT under whitening** — labeler pairwise-sensitive accuracy 0.5748 whitened
  vs 0.5758 raw = **−0.001**; the entire accuracy loss is in the **local** bucket (det/case/punct/aux: 0.870 vs 0.916 =
  −0.046). Whitening did not damage arc-discrimination; it damaged the easy local relations.
- **The fit is geometry-sensitive**: V1 is an INVERTIBLE map (same information as raw) yet it HURT — definitional proof
  that the fixed recipe (lr 0.04, 5 epochs, rank 16, tuned on raw φ) is sensitive to geometry. So the gate's HALTED,
  though a valid §5 verdict, had a fit-dynamics **confound** — hence the re-sweep.

## The re-sweep — resolves the confound (both lanes: R4_CLEAN)
After a fair lr×epochs×rank search on DEV, whitened φ does NOT beat raw φ on either primitive:
- **labeler**: best_raw 0.762 vs best_whitened 0.758 (grok margin −0.0073; codex −0.0039) — whitening nearly matches
  but never beats raw.
- **attach**: best_raw 0.615 vs best_whitened 0.539 (grok margin −0.0877; codex −0.0766) — whitening clearly worse.
→ The plateau is **robust to hyperparameters**; the HALTED is NOT a fixed-budget artifact. The confound is ruled out.
Whitening (anisotropy removal) is a dead lever for the grounded bilinear, at any budget.

## Tag discipline (the honest framing)
`empirical`: removing the rogue-dim anisotropy (effrank ~1 → ~70–155) does NOT improve the grounded bilinear on German
labeling/attachment — cross-vendor, and robust to a fair hyperparameter search. **The cheap grounded-φ bilinear recipe
(raw or whitened, any budget) PLATEAUS.** `open`: achievability of a grounded-bilinear win via a DIFFERENT lever (a
richer/other-model φ, other layers, a non-bilinear relational form) — the pairwise arc-signal is present and intact, so
this is NOT "grounding lacks signal" and grounded-φ revisits are not foreclosed. No `proved` claim (no certified
artifact). NOT "irreducible" — the recipe plateaus; achievability open.

## Disposition → R4
Per §5 HALTED (confirmed robust by the re-sweep): the German expert ships at **rung 3 (R4 LLM-hybrid)** —
registers + tags + morph + local structure + the certified **grounded LABELS** (#121's pairwise gain is real and
survives) + a one-LLM-call hybrid for **attachment** (the head-quality bottleneck neither surface, grounded, nor
whitened features closed). The prereg priced this as still a large cost win over a full LLM.

**The feature ladder, terminal:** #119 surface bilinear HALTED (wall=representation) → #120 windows HALTED (feature
quality) → #121 grounded-labeler IN-BETWEEN (first to move; head-gated) → #122 grounded-attach IN-BETWEEN (straddle) →
whitening HALTED (anisotropy was not the blocker; recipe plateaus). The cheap axis-B/grounded route is exhausted;
attachment is the R4 lever. See [[biaffine-labeler-build]].
