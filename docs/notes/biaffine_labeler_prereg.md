# Biaffine Relation-Labeler — Pre-Registration

**Status: SIGNED (approved as-is) — 2026-07-14; mechanism corrected same day against WYLY_STEERING (see §2 note).**
No numbers yet; all claims `empirical`-pending. This document fixes the decision rules BEFORE any measurement, per
program discipline — committed pre-build to timestamp the pre-registration.

## 1. The wall this answers (established)

The German R1–R3 arc located the wall: serve-honest German **case** plateaus at **~0.74–0.76** (R3-proper), below
the **0.90** germanapp bar; **the SOFT=0 greedy family can't close case** (bad deprels corrupt the cascade). The
#118 lever-probe half-oracle attributes **~79%** of the gap to deprel-**LABELING** (~21% heads), and the cheap
bilinear *head*-scorer even DEGRADED case-bearing deprels — so the redirect (lead-accepted): the axis-B primitive
is a **RELATION-LABELER**, deprel-label priority, not a head-attach scorer. Oracle-deprel case ceiling ~0.875
(codex) / 0.92 (grok).

Labeling an arc is intrinsically a function of a **pair** (governor *i*, dependent *j*). A unary/additive labeler
scores `f(i) + g(j)` and **provably cannot** represent the interaction "*i* governs *j* with relation r iff jointly
compatible." So the wall is a **rule-TYPE limitation**, not a tuning gap — the same separation as "additive readers
can't do induction, attention can."

## 2. The primitive (new rule type: pairwise / relational)

A **biaffine relation-labeler**: for the arc (dependent *i* → its head *j*), a bilinear score
`φ(i)ᵀ W_r φ(j)` per relation *r* → a deprel distribution over the arc's two endpoints' features.

**Mechanism (induction recipe — CORRECTED):** fit the bilinear form **by gradient** ("a bilinear form gradient
can find"), reusing the shipped `BilinearAttachmentScorer` pattern in `attach_levers_codex.py` (dense `φ`, low-rank
`U @ Vᵀ`) but with a LABEL head instead of a head-attach head. Then **HARDEN** (freeze/discretize into the served
form) and **CERTIFY**. This primitive lives in the **CERTIFIED tier** — it grows *beyond* the SOFT=0 greedy serving
family (which serves with no wake at inference and provably can't close case); it is NOT itself no-SGD.

> **Correction note (2026-07-14):** an earlier draft framed this as "no-SGD / count-estimated" — wrong. That
> conflated Wyly's *serving* constraint (SOFT=0) with the primitive's *fitting* method. WYLY_STEERING (lines
> 272–274, 287–289) and the shipped SGD-trained `BilinearAttachmentScorer` both specify gradient-fit → harden →
> certify. Decision rules (§5) unchanged by this correction.

## 3. Substrate, reuse & serve-honest conditions

- **Data:** GSD gold — 13,813 train / 799 dev / 977 test. **Fit on TRAIN, tune on DEV, TEST READ ONCE.**
- **Reuse (do NOT reinvent):** `attach_levers_codex.py` / `german_r3_codex.py` already provide `chu_liu_edmonds`,
  the serve-honest `φ` (R1-predicted POS + hashed surface + shape + position), the case cascade
  (`run_predicted_case_sentence` / `full_oracle_case_sentence`), `score_dependencies`, `CASE_BEARING_DEPRELS`, the
  R1 POS student, the R3 unary labeler (`predict_deprels`, the baseline), and the `CampaignTestReadGuard`.
- **Serve-honest:** features from the SENTENCE only — surface, POS (R1-predicted, the deciding arm), position,
  i–j distance/direction. **NO gold heads or gold deprels of any token** as features. Test-read-once guard applies.

## 4. The controlled comparison (the axis-B test)

Same predicted heads, identical `φ`, identical case cascade for both arms — differing ONLY in the labeler:
- **UNARY baseline** — the shipped count-table deprel labeler (`predict_deprels` / R3 relation table, keyed on
  single-position features + offset). Wyly's existing rule type.
- **BIAFFINE** — the new gradient-fit low-rank bilinear label scorer over the arc's two endpoints.

The **biaffine − unary** gap on the SAME heads isolates the labeler rule-TYPE contribution (tests "additive can't,
bilinear can" for LABELING specifically — the #118-identified bottleneck).

## 5. Pre-registered decision rules (FIXED BEFORE NUMBERS)

Metrics: serve-honest **UAS**, **LAS** (strict + coarse), deprel-only accuracy, case-bearing-deprel accuracy, and
the through-line **CASE** number via the case cascade. Relation partition for the structural prediction:
**pairwise-sensitive** = core/long-distance args {nsubj, obj, iobj, obl, nmod, conj}; **local** = adjacent/short
{det, amod, case, punct, aux, cop}.

**PRIMARY — the axis-B capability claim:**
- **FIRES** (rule type validated): biaffine beats the matched unary baseline on serve-honest deprel-LAS by
  **≥ +0.03 absolute AND beyond seed/run variance**, AND the gain is **larger on the pairwise-sensitive partition
  than the local partition** (the gain lands where pairwise structure should matter, not diffuse).
- **IN-BETWEEN**: biaffine > unary but < +0.03 (or within noise), OR the LAS gain does **not** translate to a
  case-cascade improvement, OR the gain is diffuse.
- **HALTED**: biaffine ≤ unary → the bilinear labeler doesn't realize a labeling advantage over the count-table.
  (→ rethink features/rank/decode, or concede the labeling lever to golden-set curation / the R4 fallback.)

**THROUGH-LINE — the germanapp deliverable (case, serve-honest, via cascade):**
- **≥ 0.90** → deliverable met (`empirical`; certify next).
- **< 0.90 but > ~0.76 baseline** → "the recipe closes X of the ~0.75→0.90 gap; achievability of 0.90 **open**"
  (plateau language — never "irreducible/required").
- **≤ baseline** → the labeler gain didn't survive the cascade; diagnose (R3 cascade-corruption pattern).

## 6. Certify path (after empirical FIRES)

Per the induction recipe: **harden** the gradient-fit bilinear labeler (freeze weights / discretize into the served
form), then a PIC-T3 / Soufflé certificate that over a **stated domain** (GSD-test) every emitted label is the
**margin-separated argmax** of the hardened bilinear scores. Tag stays `empirical` until the certificate exists;
then `proved` **over that domain only**.

## 7. Controls (anti-fooling)

- **Serve-honest guard:** no gold heads/deprels in features; test-read-once (reuse `CampaignTestReadGuard`).
- **Matched control:** unary vs biaffine differ ONLY in the labeler — same heads, same `φ`, same cascade.
- **POS input:** **R1-predicted POS is the deciding arm**; gold-POS reported as an upper bound only.
- **Fixed-a-priori hyperparameters:** rank/epochs/lr/seed fixed before the test read (as the attach probe did);
  no test tuning.
- **Cross-vendor:** race grok + codex on the identical spec; architect verifies BOTH diffs before picking.
- **Pre-merge gate:** pil ruff + pytest green before merge (standing invariant, never bypass).

## 8. Risks / open (stated, not hidden)

- Gradient-fit bilinear may still not beat the count-table labeler on serve-honest labeling → **achievability open**.
- Feature/factorization/rank are build-spec details; a poor scheme could under-realize the form.
- The HARDEN step (gradient-fit → checkable served form) is where certifiability is won or lost — a soft gradient
  scorer that resists a clean margin-separated hardening stalls at `empirical`.
- Data noise: GSD-train gold is clean; the app-domain **silver** (Slice-2 pipeline output) is a SEPARATE, later
  domain-adaptation concern, explicitly out of scope here.

## 9. Scope fences (what we are NOT doing)

- **GSD gold only** for the primitive's proof-of-capability — NOT the production/app silver yet.
- NOT a head-attach scorer (that was #118; redirected to LABELING).
- NOT deciding the UD-vs-component architecture fork (parked).
