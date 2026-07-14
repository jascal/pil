# Biaffine Relation-Labeler — Pre-Registration

**Status: SIGNED (approved as-is) — 2026-07-14.** No numbers yet; all claims `empirical`-pending. This document
fixes the decision rules BEFORE any measurement, per program discipline — committed pre-build to timestamp the
pre-registration.

## 1. The wall this answers (established)

The German R1–R3 arc located the wall: serve-honest German **case** plateaus at **~0.74–0.76** (R3-proper), below
the **0.90** germanapp bar. The lever-probe half-oracle factorization attributes **~79%** of the remaining gap to
deprel-**LABELING** (vs ~21% heads). The oracle-deprel case ceiling is **~0.875 (codex) / 0.92 (grok)**.

Labeling an arc is intrinsically a function of a **pair** (governor *i*, dependent *j*). A unary/additive reader
scores `f(i) + g(j)` and **provably cannot** represent the interaction "*i* governs *j* iff jointly compatible."
So the wall is a **rule-TYPE limitation**, not a tuning gap — the same separation as "additive readers can't do
induction, attention can."

## 2. The primitive (new rule type: pairwise / relational)

A **biaffine relation-labeler**: for an ordered pair (i,j), a bilinear ATTACH score `φ(i)ᵀ W_a φ(j)` ("j depends on
i") and a bilinear LABEL score `φ(i)ᵀ W_ℓ φ(j)` (the deprel). SOFT=0 / **no-SGD** acquisition: `W` is a low-rank /
feature-factored **pairwise COUNT TABLE** estimated from GSD-gold arcs (data-measure init, the [[pic-rule-learner]]
pattern) — **not** gradient-learned. Decode: Chu-Liu-Edmonds over attach scores → tree; label head → deprel.

## 3. Substrate & serve-honest conditions

- **Data:** GSD gold — 13,813 train / 799 dev / 977 test. **Estimate counts on TRAIN, tune on DEV, TEST READ ONCE.**
- **Serve-honest:** features from the SENTENCE only — surface form, POS, position, i–j distance + direction. **NO
  gold heads or gold deprels of any token** as features. Test-read-once guard applies.

## 4. The controlled comparison (the axis-B test)

Build BOTH arms in the same slice, **identical features + identical decode**, differing ONLY in the scoring form:
- **UNARY baseline** — additive readers (Wyly's existing rule type): score from single-position features.
- **BIAFFINE** — the pairwise bilinear form.

The **biaffine − unary** gap isolates the rule-TYPE contribution. This is the experiment that tests the axis-B
claim directly, not just "biaffine is good."

## 5. Pre-registered decision rules (FIXED BEFORE NUMBERS)

Metrics: serve-honest **UAS**, **LAS** (strict + coarse), and the through-line **CASE** number via the case cascade.
Relation partition (for the structural prediction below): **pairwise-sensitive** = core/long-distance args
{nsubj, obj, iobj, obl, nmod, conj}; **local** = adjacent/short {det, amod, case, punct, aux, cop}.

**PRIMARY — the axis-B capability claim:**
- **FIRES** (rule type validated): biaffine beats the matched unary baseline on serve-honest deprel-LAS by
  **≥ +0.03 absolute AND beyond seed/run variance**, AND the gain is **larger on the pairwise-sensitive partition
  than the local partition** (the gain lands where pairwise structure should matter, not diffuse).
- **IN-BETWEEN**: biaffine > unary but < +0.03 (or within noise), OR the LAS gain does **not** translate to a
  case-cascade improvement, OR the gain is diffuse (not concentrated in the pairwise-sensitive partition).
- **HALTED**: biaffine ≤ unary → count-estimation fails to realize the bilinear advantage; the SOFT=0 count recipe
  doesn't capture the interaction. (→ rethink the ESTIMATOR/factorization, not the rule type.)

**THROUGH-LINE — the germanapp deliverable (case, serve-honest, via cascade):**
- **≥ 0.90** → deliverable met (`empirical`; certify next).
- **< 0.90 but > ~0.76 baseline** → register as "the count-estimated recipe closes X of the ~0.75→0.90 gap;
  achievability of 0.90 **open**" (plateau language — never "irreducible/required").
- **≤ baseline** → the labeler gain didn't survive the cascade; diagnose (head errors dominating, cascade
  corruption per the R3 pattern).

## 6. Certify path (after empirical FIRES)

PIC-T3 / Soufflé certificate: over a **stated domain** (GSD-test), every emitted (arc, label) is the
**margin-separated argmax** of the pairwise count table — the certified-induction-head recipe, generalized to
pairs. Tag stays `empirical` until the certificate exists; then `proved` **over that domain only**.

## 7. Controls (anti-fooling)

- **Serve-honest guard:** no gold heads/deprels in features; test-read-once.
- **Matched-features control:** unary vs biaffine differ ONLY in scoring form (same features, same decode).
- **POS input:** two arms — (a) **predicted-POS** (fully serve-honest; decision rules apply to THIS arm) and
  (b) **gold-POS** (upper bound; reported, not the deciding number).
- **Cross-vendor:** race grok + codex on the identical prereg'd spec; architect verifies BOTH diffs before picking.
- **Pre-merge gate:** pil ruff + pytest green before merge (standing invariant, never bypass).

## 8. Risks / open (stated, not hidden)

- Count-estimation (no SGD) may not reach the bilinear ceiling a trained biaffine would → **achievability open**.
- The feature/factorization scheme + rank are build-spec details; a poor scheme could under-realize the form.
- Data noise: GSD-train gold is clean; the app-domain **silver** (the Slice-2 pipeline output) is noisier — a
  SEPARATE, later domain-adaptation concern, explicitly out of scope here.
- Register arbitration: how the new pairwise register composes with the unary readers (integration detail).

## 9. Scope fences (what we are NOT doing)

- NOT training by SGD/backprop (SOFT=0 preserved — the whole point).
- NOT using production/app silver yet — **GSD gold only** for the primitive's proof-of-capability.
- NOT deciding the UD-vs-component architecture fork (parked).
