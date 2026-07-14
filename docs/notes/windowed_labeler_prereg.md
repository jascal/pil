# Windowed Count-Table Labeler — Pre-Registration (feature-ladder rung 1)

**Status: SIGNED (approved as-is) — 2026-07-14.** No numbers yet; decision rules fixed BEFORE measurement;
committed pre-build to timestamp the pre-registration.
Feature-ladder rung 1, per WYLY_STEERING post-#119. Predecessor: [`biaffine_labeler_outcome.md`](./biaffine_labeler_outcome.md)
(#119 HALTED — the wall is REPRESENTATION not form; redirect = richer FEATURES, not fuller forms).

## 1. The move
#119 proved a bilinear over *static per-token* `φ` is a worse estimator than the unary count-table. Rung 1 keeps
the **count-table** (the winning estimator, and the served tier) but gives it **richer keys**: a **window** of
context features (M-context = left context + lookahead) around the token and its governor, with **backoff** for
sparsity. This is FEATURES, not FORM, and — being a count-table — it needs no harden step; it serves directly in
the certified tier. Cheapest rung; the NN is a yardstick only, never shipped ([[cheap-by-design-no-reflexive-nn-fallback]]).

## 2. The primitive
Extend the shipped unary labeler (`predict_deprels` / `_relation_key_levels`, keyed on token + POS + head-offset)
to a **windowed count-table**: key additionally on the **POS (and coarse surface shape) of the ±M neighbors** of the
dependent AND of the governor. **Backoff**: when the full-window key count < threshold T, back off through
progressively smaller windows down to the unary key (n-gram-style; per the #84/#85 smoothing/data-measure
contract). Deterministic (no seeds). M and T **tuned on DEV**, fixed for the test-once read.

## 3. Substrate & serve-honest
GSD gold (13,813 / 799 / 977). Fit tables on TRAIN; tune (M, T) on DEV; TEST READ ONCE. Same serve-honest L0
predicted heads as #119 (isolates labeling). Features from the SENTENCE only — surface, R1-predicted POS, position,
window; NO gold heads/deprels. Reuse: `predict_deprels` baseline, the `φ`/POS student, `chu_liu_edmonds` (not
needed — same heads), the case cascade, `score_dependencies`, `CASE_BEARING_DEPRELS`, `CampaignTestReadGuard`.

## 4. Controlled comparison
UNARY baseline (`predict_deprels`) vs WINDOWED count-table, **same** predicted heads, same `φ`/POS, same cascade —
differing ONLY in the key. Isolates the feature-window contribution. UAS identical across arms (assert).

## 5. Pre-registered decision rules (FIXED BEFORE NUMBERS)
Metrics: serve-honest UAS, LAS (strict + coarse), deprel-only accuracy, case-bearing-deprel accuracy, serve-honest
CASE (cascade), and the pairwise-sensitive {nsubj,obj,iobj,obl,nmod,conj} vs local {det,amod,case,punct,aux,cop}
partition. Anchors: unary deprel-only ~0.696 (pred heads) / 0.790 (gold heads); serve-honest case ~0.74–0.76 (R3);
bar 0.90.

**PRIMARY (does cheap context help?):**
- **FIRES**: windowed beats unary on serve-honest deprel-LAS by **≥ +0.03 absolute**, AND the gain is **larger on
  the pairwise-sensitive partition than the local partition** (context lands where it should).
- **IN-BETWEEN**: gain > 0 but < +0.03, OR diffuse, OR does not reach the case cascade.
- **HALTED**: windowed ≤ unary → raw-feature windows don't capture the needed context → escalate to **rung 2
  (grounded features)**, per the ladder.

**THROUGH-LINE (case, serve-honest, via cascade):** ≥ 0.90 → deliverable met; < 0.90 but > ~0.76 → "closes X of the
gap; achievability open" (plateau language); ≤ baseline → cascade-corruption / diagnose.

## 6. Certify
A windowed count-table with backoff **is** a count-table — it lives in the certified tier with the existing
primitives (margin over keys). No separate harden step; if it FIRES it ships a certified primitive directly. Tag
stays `empirical` until the certificate over the stated domain (GSD-test) exists; then `proved` over that domain.

## 7. Controls
Serve-honest (no gold heads/deprels in keys; test-read-once). Matched (unary vs windowed differ only in the key).
(M, T) tuned on DEV only, never test. R1-predicted POS is the deciding arm (gold-POS an upper-bound diagnostic).
Cross-vendor race grok + codex (both now available); architect verifies both. pil ruff + pytest pre-merge gate.

## 8. Scope fences
GSD gold only (not app silver). Rung 1 only — rung 2 (grounded features) is a separate slice iff this halts.
NOT a bilinear form (that was #119). NOT a served NN.
