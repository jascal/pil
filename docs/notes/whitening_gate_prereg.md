# Anisotropy-Removal (Whitening) Gate — Pre-Registration (feature-ladder, re-registered)

**Status: SIGNED (approved as-is) — 2026-07-16. Decision rules fixed BEFORE numbers; signed while the MWT CPU re-dump
runs, before any whitening number.** The disposition slice after #122
(grounded ATTACH IN-BETWEEN; the reviewer-pinned C7 check found the blocker: the grounded φ carries a
**massive-activation rogue dimension — one dim ≈ 73% of variance** — violating the near-orthogonality precondition the
bilinear needs). Predecessors on RAW φ: [`grounded_labeler_outcome.md`](./grounded_labeler_outcome.md) (#121, labeler
IN-BETWEEN, head-gated) and [`grounded_attach_outcome.md`](./grounded_attach_outcome.md) (#122, attach IN-BETWEEN,
cross-vendor straddle). Re-registered (NOT a re-read of an old gate) because it PINS the two things #122 left loose
(Fable's deciding risk = test reuse + unpinned decode).

## 1. The move
Strip the anisotropy from the grounded φ BEFORE the bilinear, then re-test **BOTH** the #121 labeler AND the #122
attach on the whitened φ. Reviewer's framing: *"if whitened φ converts the straddle, both #121's head-gate and #122's
near-miss resolve from one fix."* This is also the **mechanism discriminator**: does removing the rogue dim unlock the
bilinear (→ anisotropy WAS the blocker), or not (→ it's optimization conditioning / deeper, per Fable's caveat, and
grounding-as-substrate plateaus). Cheap-by-design intact: a whitening map is a fixed linear transform (fit offline on
TRAIN, applied at serve), hardens fine ([[cheap-by-design-no-reflexive-nn-fallback]]).

## 2. The transform (cheapest-first, DEV-swept)
Fit on TRAIN grounded φ only (no test/dev leakage). Variants, DEV-swept:
- **V1 — per-dim standardize** (z-score): `φ' = (φ − μ)/σ`, μ,σ from TRAIN. Cheapest; demotes the rogue dim's raw
  magnitude directly. The primary variant.
- **V2 — top-k PC removal**: project out the top-1 (and DEV-swept top-k∈{1,2}) TRAIN principal directions, then
  standardize the remainder. Stronger; removes the dominant direction outright.
Applied to φ for BOTH endpoints before the bilinear. The whitening params (μ,σ,PCs) are TRAIN-fit constants — part of
the served artifact (a fixed linear map). DEV-select the variant by the same criterion each primitive already uses.

## 3. Substrate & serve-honest
The **richer MWT φ** (`qwen3b_mwt_gsd_{dev,test,train}`, CPU int8 — SAME quant scheme as #121/#122, MWT coverage
restored per #123/#124). NOT the GPU φ (different scheme — never mixed; see #133). Same serve-honest discipline:
grounding from TEXT only, L0 predicted heads, doubly-covered arc set, whitening fit on TRAIN only, DEV-only variant
sweep, TEST read once (`CampaignTestReadGuard`). Matched arms (assert UAS/arc-set identity).

## 4. PINS (fixed a-priori — these are what makes this a fresh gate, not a re-read)
- **PIN A — the effrank / C7 definition + sample.** Near-orthogonality is measured as the **centered participation-ratio
  effective rank** `(Σσ²)²/Σσ⁴` on the centered TRAIN φ, on a FIXED sample of **n = 800** rows (seed 0), reported at
  each layer-checkpoint, for BOTH raw and whitened φ. (The #122 effrank instability was n-sensitivity around the rogue
  dim: n=300 read ~45, n≥500 read ~1. n=800 + fixed seed removes the ambiguity.) The C7 statement is descriptive
  (premise the result is read against), NOT a gate variable.
- **PIN B — the attach decode rule (the #122 straddle source).** All arc scores are **log-probabilities**, consistently
  normalized. For a covered dependent: the bilinear emits a **log-softmax over the covered-governor candidate set**
  (excluding self). Uncovered-governor cells and the ROOT cell take the **L0 arc log-prob**, and each row is
  renormalized to a proper log-distribution before decode. `chu_liu_edmonds` runs on the log-prob matrix. No mixing of
  raw scores with softmax probabilities (the underspecified reconciliation that split grok/codex on #122). Both lanes
  implement THIS rule; assert identical arc-set + UAS-matched arms.
- Fixed-a-priori labeler/attach hyperparameters (reuse #121/#122's). Deterministic given seed. Single seed (stated).

## 5. Pre-registered decision rules (FIXED BEFORE NUMBERS)
Run BOTH primitives on the whitened φ (DEV-selected variant), each against its OWN pre-registered bars AND its raw-φ
predecessor. Metrics as in #121/#122 (deprel-only, LAS-strict, case cascade, pairwise-sensitive vs local; UAS +
conversion for attach). Anchors: #121 raw-φ labeler (deprel-only +0.06–0.08, LAS gain +0.018 head-gated); #122 raw-φ
attach (UAS gain +0.016–0.031 straddle, conversion LAS +0.026–0.033).

- **FIRES**: whitening converts EITHER primitive cleanly over its #121/#122 bar — i.e. the **labeler** clears
  serve-honest deprel-LAS ≥ +0.03 over unary (the #121 FIRES bar), OR the **attach** clears UAS ≥ +0.03 over L0 AND
  the combined conversion LAS ≥ +0.03 over baseline (the #122 FIRES bar) — **AND cross-vendor robust** (both lanes on
  the pinned decode agree the bar is cleared, no straddle). Report which primitive fired and the whitened-vs-raw delta.
- **IN-BETWEEN**: whitening improves a primitive over its raw-φ predecessor but does not clear the bar, OR clears on
  one lane but straddles (not robust). The rogue dim mattered but isn't the whole story.
- **HALTED**: whitening ≤ raw-φ on both primitives → the rogue-dim anisotropy was NOT the blocker (mechanism =
  optimization conditioning / deeper; grounding-as-cheap-substrate for the bilinear plateaus) → **rung 3 (R4
  LLM-hybrid)**; the German expert ships as registers + tags + morph + local + grounded-labels + hybrid-attach.
- **THROUGH-LINE (case, cascade)**: ≥ 0.90 met; < 0.90 but > ~0.76 plateau language; ≤ baseline diagnose.
- **MECHANISM READ (recorded either way)**: report whitened effrank (PIN A) — does whitening actually raise the
  effective rank, and does any accuracy gain track that? (Separates "anisotropy removed AND helped" from "removed but
  didn't help" = the conditioning hypothesis.)

## 6. Controls
Serve-honest (whitening fit on TRAIN only; test-read-once). DEV-only variant sweep. Matched arms (assert identity).
Fixed-a-priori hyperparameters. Cross-vendor race grok + codex on the PINNED decode + effrank definition; architect
verifies both. pil ruff + pytest pre-merge gate.

## 7. Risks / open (stated)
- **Whitening is a fixed linear map → expressivity-invariant for the bilinear** (Fable): a bilinear W can in principle
  already absorb any invertible linear transform, so a pure-standardize (V1) gain, if any, is an OPTIMIZATION-conditioning
  effect (better-scaled inputs → the fixed-epoch gradient fit converges better), not new expressivity. V2 (PC removal)
  is non-invertible (drops directions) → can genuinely change what the bilinear sees. Report V1 vs V2 separately; read
  a V1-only gain as conditioning, a V2 gain as information-removal-helps.
- **The rogue dim may be f32-exact and task-relevant**: removing it (V2) could drop signal, not just noise. The DEV
  sweep + the both-primitives read guard against over-removing.
- **Harden (iff FIRES)** — the whitening map + the bilinear must quantize cleanly to the served lookup; a soft scorer
  that resists quantization stalls at `empirical` (the recurring lesson).

## 8. Scope fences
The richer MWT CPU φ (qwen3b_mwt), GSD gold, doubly-covered arcs. Whitening = a TRAIN-fit fixed linear map (NOT a
served NN; transformer-free after harden). Both primitives re-tested; the SAME whitened φ feeds both. R4 only if this
halts on both.
