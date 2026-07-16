# Grounded Relation-Labeler — Pre-Registration (feature-ladder rung 2)

**Status: SIGNED (approved as-is) — 2026-07-16.** Decision rules fixed BEFORE numbers; signed + committed after the
grounded dump but BEFORE any 2b labeler number. Rung 2 of the post-#119 feature
ladder. Predecessors: [`biaffine_labeler_outcome.md`](./biaffine_labeler_outcome.md) (#119 — biaffine over SURFACE
features HALTED; wall is representation) and [`windowed_labeler_outcome.md`](./windowed_labeler_outcome.md) (#120 —
cheap POS-window context HALTED; feature quality, not plumbing). Both earned this rung.

## 1. The move
Replace the labeler's surface `φ` with **grounded per-word residuals** from a real German-capable model
(**Qwen2.5-3B-Instruct**, chosen — gemma ruled out (arch has no fieldrun logits hook, can't source-dump); 7B ruled
out (~32s/sent CPU → ~30h; also >8GB VRAM); 3B converted from HF cache, int8, dim 2048, 36 layers), computed
**offline (grounding teacher in the TRAINING path only)**, then **harden to a
discrete lookup so the serve path stays transformer-free**. Tests whether grounded, contextual features (the program
says concepts are ~0.94 linearly present in residuals) give the labeler the signal that surface (#119) and cheap
abstract windows (#120) lacked. Cheap-by-design intact: the model runs once offline; the served artifact is a
hardened lookup, not an NN ([[cheap-by-design-no-reflexive-nn-fallback]]).

## 2. The primitive
Reuse the biaffine labeler's **φ SEAM** (`BilinearRelationLabeler._feature_builder.phi` in
`experiments/biaffine_labeler_codex.py`) — swap surface `φ` for the per-word grounded residual `r_x` from
`pil.fieldrun_io.load_source_dump` (`D`/`r`, **layer-swept** by slicing `D`'s blocks). Matched vs the shipped UNARY
count-table AND the surface-φ biaffine (#119), on the **same** serve-honest L0 predicted heads.

## 3. Phases (this prereg governs 2b)
- **2a — the German residual dump + subword→word alignment** (prerequisite; the correctness risk lives here — must
  be verified: a populated-alignment test + a coverage report). gemma-2-2b via `fieldrun --source-dump`; subset-first
  (dev+test + a train subset).
- **2b — the grounded-φ labeler GATE** (the decision below): does grounding beat unary?
- **2c — harden to a discrete lookup + certify** (iff 2b FIRES): quantize `φ`→codes→count-table per the
  GeoConcepts / `wyly_rel_harden` anneal→straight-through→extract recipe; certify hard==soft on GSD-test; serve
  transformer-free. USER refinement: a **domain-specific codebook** (constrained meanings → small enumerable →
  cheap+certifiable) is the harden-time lever.

## 4. Substrate & serve-honest
GSD gold (13,813 / 799 / 977). Grounding from the sentence TEXT only — **NO gold heads/deprels** touch the model or
the features (the model reads text; that is the training-path teacher). Layer(s) tuned on DEV. TEST read once
(reuse `CampaignTestReadGuard`). Same L0 predicted heads as #119/#120 (isolates labeling). R1-predicted POS may
accompany `φ`; gold-POS is an upper-bound diagnostic only.

**Coverage / where the gate is measured (registered, user-approved 2026-07-16):** the grounded dump (Qwen2.5-3B, 4
layer-checkpoints = layers [8,17,26,35]) covers ~63–77% of GSD words (fieldrun omits sentence-first/last decode
positions + German MWT-contraction drops; sanity ~99.5–100% on covered). The biaffine labeler needs `φ` for BOTH
the dependent AND its governor → an arc is **grounded only when BOTH endpoints are covered ≈ 0.7×0.7 ≈ ~50% of
arcs**. So **the gate is measured on the DOUBLY-COVERED arc set** (both-endpoints-grounded); uncovered arcs fall
back to the unary label. This asks "does grounding help WHERE it has `φ` for the pair." The matched UNARY + the
surface-φ #119 baselines are scored on the **SAME doubly-covered arc set** (apples-to-apples). Report the
doubly-covered fraction. (MWT-aware coverage recovery is a separate later follow-up, not a gate variable.)

## 5. Pre-registered decision rules (2b — FIXED BEFORE NUMBERS)
Metrics: serve-honest deprel-LAS (strict+coarse), deprel-only, case-bearing-deprel, serve-honest CASE (cascade),
and the pairwise-sensitive {nsubj,obj,iobj,obl,nmod,conj} vs local {det,amod,case,punct,aux,cop} partition. Anchors:
unary deprel-only 0.696 (pred heads); surface-φ biaffine #119 lost (−0.03); bar 0.90 case.

- **FIRES**: grounded-φ labeler beats the UNARY baseline on serve-honest deprel-LAS by **≥ +0.03 absolute**, AND
  the gain is **larger on the pairwise-sensitive partition than local**, AND it beats the surface-φ biaffine (#119)
  — grounding helps where surface didn't.
- **IN-BETWEEN**: gain > 0 but < +0.03, OR diffuse, OR no case-cascade improvement.
- **HALTED**: grounded-φ ≤ unary → grounding doesn't rescue the labeler either → **rung 3 (R4 LLM-hybrid)**; the
  German expert ships as registers + tags + morph + local + hybrid.
- **THROUGH-LINE (case, cascade)**: ≥ 0.90 → met; < 0.90 but > ~0.76 → plateau language; ≤ baseline → diagnose.

## 6. Controls
Serve-honest (grounding from text; no gold labels in features; test-read-once). **DEV-only layer sweep** (which
gemma block/layer subset; optionally J-corrected via `load_jlens`/`jcorrect_sources` vs the `lam=0` logit-lens
baseline). Matched heads (assert UAS identical). Fixed-a-priori labeler hyperparameters. Cross-vendor race grok +
codex; architect verifies both. pil ruff + pytest pre-merge gate.

## 7. Risks / open (stated)
- **Subword→word alignment correctness** (the 2a plumbing) — the class of bug cross-vendor caught in #120; verified
  with a populated-alignment test + coverage report.
- Whether gemma-2-2b's German residuals carry the syntactic signal — the gate answers it; a weak result may be the
  model, not the approach (a stronger/other-layer arm is the follow-up, not a re-registration).
- **The harden step (2c)** is where certifiability is won or lost — a soft grounded scorer that resists clean
  quantization stalls at `empirical` (the recurring lesson).

## 8. Scope fences
gemma-2-2b, subset-first, GSD gold. NOT a served NN (transformer-free after harden). Domain-specific codebook is a
2c (harden-time) lever, not a 2b variable. Rung 3 (R4 hybrid) only if this halts.
