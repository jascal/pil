# Grounded Attachment-Primitive — Outcome (feature-ladder attach gate)

**Verdict: IN-BETWEEN (cross-vendor; aggregation rule resolved conservatively). Tag: `empirical`.** Grounded-φ helps
attachment a little (beats surface-#119 both lanes; beats L0 both lanes) but does NOT robustly clear the
pre-registered +0.03 bar — the two lanes straddle it. Governed by [`grounded_attach_prereg.md`](./grounded_attach_prereg.md)
(§5 = law). Predecessor: [`grounded_labeler_outcome.md`](./grounded_labeler_outcome.md) (#121 — grounded LABELER
IN-BETWEEN, head-gated). Same shape now confirmed for heads: **grounding moves attachment too, but not over the bar.**

## What ran
The #119 biaffine, retried on the grounded substrate for HEADS: a bilinear attachment scorer over grounded φ
(Qwen2.5-3B), CLE-decoded (uncovered governors → L0 fallback), matched vs L0 predicted heads and the surface-φ attach
scorer (#118/#119), on the covered-dependent arc set, DEV attach-layer sweep, plus the conversion (grounded heads +
grounded labels → LAS + case). Cross-vendor race: grok (`experiments/campaign_grounded_attach.py`) + codex
(`experiments/grounded_attach_codex.py`). Both architect-verified: ruff clean, 781 tests pass, both scoreboards
re-run byte-identical; grok's decode confirmed serve-honest (no gold-head leak).

## The numbers (both lanes; covered-dependent arcs, matched, n=10374; DEV layer 17)
| metric | grok | codex |
|---|---|---|
| L0 baseline UAS | 0.4769 | 0.4769 |
| surface-#119 UAS | 0.4679 | 0.4679 |
| grounded UAS | **0.5074** | **0.4929** |
| grounded UAS gain vs L0 | **+0.0306** | **+0.0160** |
| gain vs surface-#119 | +0.0395 | +0.0250 |
| conversion LAS-strict gain (full − base) | **+0.0333** | **+0.0255** |
| case_full | 0.603 | 0.575 |
| §5 verdict | FIRES | IN-BETWEEN |

## Why the honest verdict is IN-BETWEEN, not FIRES (aggregation resolved conservatively)
§5 never pre-registered a **lane-aggregation** rule (it didn't need one — #121's lanes agreed). This is not overriding
§5; it is resolving an underspecified rule, and the conservative resolution is correct for two non-taste reasons:
1. **grok's +0.0306 is statistically indistinguishable from the +0.03 bar.** Binomial SE on UAS at n=10374 is ~0.005;
   grok clears by 0.0006 = ~⅛ of one SE. A "FIRE" 0.0006 over the bar is noise.
2. **The estimand itself wasn't pinned.** The 0.0145 cross-lane UAS gap is entirely a decode detail the spec left
   underspecified (both overwrite covered-dependent cells with bilinear softmax-probs and leave L0 raw scores
   elsewhere, then CLE-decode; the softmax-candidate / probability-vs-score scale reconciliation differs). You cannot
   credit a FIRE on a quantity the prereg underdetermines.

Both lanes beat BOTH baselines (L0 and surface-#119) → the result is **"promising"** (not a HALT). Grounding helps
heads a little — the same signature as #121 for labels — just not robustly over the bar.

## The C7 premise finding (reviewer-pinned check — it earned its keep)
The dumped φ carries a **massive-activation rogue dimension**: a single dimension holds **~73% of the total variance**
(top-dim variance share 0.732 at layer 17, stable). Mean pairwise cosine ~0.29–0.36. So the **near-orthogonality
precondition the induction bilinear needs is VIOLATED** — but by a rogue dim, not intrinsic low rank.
- **A measurement caveat, recorded (corrects a first read):** the participation-ratio effrank is *sample-unstable*
  around the rogue dim — n=300 (grok's C7) under-samples the extreme rows → effrank ~45; n≥500 (codex, + my direct npz
  computation) catches them, the top singular value jumps to ~2000–6700, and effrank collapses to ~1. Same formula,
  same data — **grok's C7 is not bugged; the metric is sample-sensitive.** The robust characterization is the
  73%-variance rogue dim, not the effrank number. (Next gate will PIN the effrank definition + sample.)
- **Mechanism is a HYPOTHESIS, not established** (Fable): a gradient-fit bilinear W can in principle null a rogue dim
  (expressivity is invariant to any invertible linear map), so anisotropy can only bite via **optimization
  conditioning** — and codex's DEV curve shows layer-35 losses exploding to ~66, evidence conditioning does bite
  somewhere. "Grounding only half-works BECAUSE of the rogue dim" is what the next experiment tests, not a proven claim.

## Disposition (§5 + Fable-checked)
IN-BETWEEN → **not 2c** (§5: harden iff FIRES), **not R4** (that's the HALTED disposition; grounding clearly helps).
Two forward moves, both cheap-by-design:
1. **NEXT ATTACH LEVER (recommended): anisotropy removal on the grounded φ**, then re-test attach (and re-test the
   #121 labeler on whitened φ). Cheapest-first = per-dim standardize (train-fit statistics only — targets the rogue
   dim directly); optionally strip the top-PC. This doubles as the mechanism discriminator (does killing the rogue dim
   unlock the bilinear?). **MUST be RE-REGISTERED as a fresh gate that PINS (a) the effrank definition + sample and
   (b) the softmax/scale-reconciliation decode rule** — otherwise it re-reads TEST only to reproduce the same
   cross-lane straddle. The deciding risk is **test reuse + the unpinned decode**.
2. **MWT-recovery is now UNLOCKED** (per §3: "run if this gate fires OR is promising"; both-beat-baselines =
   promising). It needs the ~6–20h re-dump ([[mwt-aware-alignment-todo]]); the [[fieldrun-gpu-forward-pass-todo]]
   wiring would make that re-dump cheap on GPU. Sequencing (whitening gate vs MWT re-dump) is a user/lead call.

## Tag discipline
`empirical`: grounded-φ attachment beats surface-#119 (+0.025–0.040) and L0 (+0.016–0.031), cross-vendor, but does
not robustly clear the +0.03 serve-honest bar (straddle). `empirical`: the grounded φ carries a ~73%-variance rogue
dimension (near-orthogonality precondition violated). `open`: whether anisotropy removal converts the marginal signal
into a clean FIRE — the whitening gate answers it. No `proved` claim (soft scorer, not a certified lookup; 2c not
triggered).
