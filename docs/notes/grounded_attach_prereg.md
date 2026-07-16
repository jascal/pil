# Grounded Attachment-Primitive — Pre-Registration (feature-ladder rung 2, attach gate)

**Status: SIGNED (approved as-is) — 2026-07-16. Decision rules fixed BEFORE numbers; signed before any attach number.
Build sequence (user-set 2026-07-16, measure-first): this attach gate runs FIRST on CURRENT coverage (0.57–0.63
doubly-covered). MWT-recovery is DEFERRED — the recon found it needs a ~6–20h re-dump (contraction residuals were
never persisted; MWT sentences are dropped whole before the npz is written), so it runs ONLY if this gate fires or is
promising. The gate is cheap and decisive today: does grounding help HEADS at all on the covered set?** The
disposition slice accepted by
the reviewer after [`grounded_labeler_outcome.md`](./grounded_labeler_outcome.md) (#121 rung-2 labeler IN-BETWEEN,
head-gated). Predecessors: #121 (grounded-φ **labeler** moves the needle — beats surface, +0.06–0.08 deprel-only, but
strict-LAS head-gated) and the #118/#119 diagnostics (a surface-φ **head-scorer** capped ~0.46–0.56 UAS, at/below the
L0 baseline — the reviewer's explicit baseline for this gate).

## 1. The move
#121 proved grounding fixes **labels** but the serve-honest strict-LAS is **head-gated** (both arms share the weak L0
heads, UAS ~0.49; the label win only half-converts). The lead's oracle-attachment rescore independently found
**attachment is the German expert's critical path** ("oracle attachment closes case → the expert reduces to ONE
learnable object = clause/government attachment"). So this slice retries #119's biaffine on the **grounded** substrate
for **HEADS, not labels**: a bilinear attachment scorer over grounded φ. The test: does grounding help ATTACHMENT
where surface (#118/#119) didn't — exactly as it helped labeling (#121)? If UAS rises, #121's label win converts to
LAS and serve-honest case lifts toward the 0.90 bar. Cheap-by-design intact: grounding teacher offline (TRAINING path
only); the served artifact is a hardened discrete lookup, not an NN ([[cheap-by-design-no-reflexive-nn-fallback]]).

## 2. The primitive
Reuse the shipped **`BilinearAttachmentScorer`** form template (`experiments/attach_levers_codex.py`, low-rank
U@Vᵀ, gradient-fit) — swap surface φ for the per-word **grounded residual** (Qwen2.5-3B dump, layer-swept over the 4
checkpoints [8,17,26,35]). Arc score `s(i→j) = φ(i)ᵀ W φ(j) + linear + bias`; decode a valid tree with the shipped
**`chu_liu_edmonds`**. **Fit on GOLD arcs** (dependent i, its gold head j) where both covered; gradient fit; ROOT
handled deterministically. **Serve**: score every candidate arc where both endpoints covered; **uncovered candidate
governor → fall back to the L0 arc score** (or a distance prior) so the CLE decode is well-defined; then CLE over the
mixed scores. Fixed-a-priori rank/epochs/lr/seed (reuse #119's).

## 3. Phases
- **This gate (runs FIRST, measure-first):** does grounding help attachment? On the CURRENT covered set (0.57–0.63
  doubly-covered). The decision in §5. Cheap and decisive today.
- **DEFERRED — MWT-recovery (run only if this gate fires/is promising):** recover the ~28% coverage lost to German
  MWT contractions (im/am/zum/zur/beim/vom). The recon found this needs a ~6–20h re-dump (raw residuals never
  persisted; MWT sentences dropped whole at `compute_word_spans` before the npz write) plus a bounded code fix (3
  alignment funcs + a contraction map, since GSD discards MWT spans at build). Its own scoped slice + verification
  (coverage rises, sanity holds, alignment-correctness test); NOT a decision gate. ([[mwt-aware-alignment-todo]].)
  Purpose if run: lift the covered fraction (~0.60 → ~0.75) so the gate's serve-honest number is less dilution-capped.
- **Harden (iff this FIRES):** grounded attach scorer → quantize φ → discrete lookup, certify hard==soft on
  GSD-test, serve transformer-free (the domain-specific codebook is the harden-time lever, per #121's 2c note).

## 4. Substrate & serve-honest
GSD gold. Grounding from sentence TEXT only — no gold heads/deprels touch the model or the features. Attachment
needs φ for the dependent AND each candidate governor → **coverage bites harder than for the labeler** (every word is
a candidate governor). Measure on the **covered-dependent arc set** (dependent covered; uncovered governors fall back
in the decode), report the diluted full-set UAS alongside. Layer swept on DEV. TEST read once
(`CampaignTestReadGuard`). The downstream conversion measured serve-honest: grounded-attach heads → the #121 grounded
labeler → case cascade.

## 5. Pre-registered decision rules (FIXED BEFORE NUMBERS)
Metrics: serve-honest **UAS** (primary attach), and the **conversion** — LAS-strict + serve-honest CASE (cascade)
under the grounded-attach heads combined with the #121 grounded labels. Baselines (reviewer pin — #119's halt numbers
explicit): (a) **L0 predicted heads** UAS ~0.49 (the shipped attach baseline); (b) **surface-φ bilinear attach
(#118/#119)** UAS ~0.46–0.56 (grounding-vs-surface control). Anchors: unary/L0 serve-honest case (#121) 0.49–0.52;
bar 0.90 case; #121 grounded-labels-on-L0-heads strict-LAS gain +0.018 (head-gated).

- **FIRES**: grounded-attach serve-honest **UAS beats L0 by ≥ +0.03** AND **beats surface-φ attach (#119)** — grounding
  helps heads where surface didn't — **AND the combined (grounded heads + grounded labels) serve-honest strict-LAS
  gain ≥ +0.03** over the shipped (L0 heads + unary labels) baseline (i.e. #121's label win now converts). Report the
  case-cascade lift alongside.
- **IN-BETWEEN**: UAS lifts but < +0.03, OR beats L0 but not surface-#119, OR UAS lifts but doesn't convert to a
  ≥ +0.03 combined LAS gain / no case improvement.
- **HALTED**: grounded-attach UAS ≤ L0 (and ≤ surface-#119) → grounding doesn't rescue attachment → **the R4
  LLM-hybrid for attachment stands as designed** (still a large cost win); the German expert ships as registers +
  tags + morph + local + grounded-labels + hybrid-attach.
- **THROUGH-LINE (case, cascade)**: ≥ 0.90 → met; < 0.90 but > ~0.76 → plateau language; ≤ baseline → diagnose.

## 6. Controls
Serve-honest (grounding from text; no gold in features; test-read-once). **C7 near-orthogonality premise check
(reviewer pin):** before the gate, measure and STATE the geometry of the dumped φ at the swept layers (pairwise
cosine / effective rank of the per-word grounded features) — the induction bilinear worked because incidence lived in
near-orthogonal grounded reps; record whether this substrate satisfies that precondition, as a stated premise the
result is read against (not a gate variable). **DEV-only layer sweep.** Matched arc set across arms. Fixed-a-priori
hyperparameters. Cross-vendor race grok + codex; architect verifies both; pil ruff + pytest pre-merge gate.

## 7. Risks / open (stated)
- **Coverage / decode corruption** (Fable's deciding risk): attach needs φ for all candidate governors; uncovered
  governors fall back to L0 → dilutes the UAS gain by ~40% at the pre-MWT fraction. Mitigated by running MWT-recovery
  first (slice N); the fallback keeps the decode well-defined; measured on the covered-dependent set with the diluted
  full-set reported.
- **The surface head-scorer already failed (#118/#119)** — a HALT here means grounding rescued labeling but not
  attachment (a real, publishable asymmetry), not a broken harness.
- **Harden (iff FIRES)** is where certifiability is won or lost — a soft grounded attach scorer that resists clean
  quantization stalls at `empirical` (the recurring lesson).

## 8. Scope fences
Qwen2.5-3B grounded φ (the #121 dump), GSD gold, POST-MWT coverage. NOT a served NN (transformer-free after harden).
Domain-specific codebook is a harden-time lever, not a gate variable. R4 hybrid only if this halts.
