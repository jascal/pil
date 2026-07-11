# qa1 conditional-headroom probe (slice #79) — the residual was already gone

**Status:** measured (2026-07-11). First slice of the compositionality-on-text thrust.
Pre-registered as H3 → H1 → H2 (workspace prereg log, recorded before any code existed);
what the probe *found* is different from what it went looking for, and more useful: **the
qa1 benchmark can no longer measure generalization residual at all.**
`experiments/campaign_qa1_cond_headroom.py` + `tests/test_qa1_cond_headroom.py`
(11 tests; story-split disjointness asserted; moveloc family unit-tested incl.
stale-pointer and no-movement cases).

## Registered outcomes, verbatim

| Hypothesis | Registered rule | Outcome |
|---|---|---|
| H3 judge-distribution artifact | CONFIRMED iff query-batch marginal clears the admit threshold while window marginal does not | **NOT CONFIRMED** — window +0.0347 > 5e-4 |
| H1 family headroom | PASS iff moveloc recovers ≥ 0.5 of R at precision ≥ 0.8 | **SKIPPED** — R is empty (below) |
| H2 conditional-carry arbitration | gated ≥ flat + 0.02 ∧ regressions ≤ ∧ p < 0.05 | **SKIPPED** — gated on H1 |

**H3's failure is a pre-registration defect, named as such:** the registered binary did
not pin the *baseline ensemble*. The implementation's window-val baseline is a pure
suffix-k cover, against which moveloc picks up a small positive window marginal from rare
`A:`-ending mid-window positions; the historical ~0 window marginal was measured against
the full multi-tier cover. The robust, **unregistered** observation: the query-shaped
marginal is **+0.842 vs +0.035 window (≈24×)** — the deployment-vs-window judge gap is
real and large; it is reported as *empirical (unregistered)* and not used for any verdict.
**Lesson (binding on future preregs): pin the baseline ensemble.**

## The finding: the bench is saturated by construction

B0 (the on-disk v5 qa1 package, 38,147 rules, support-weighted cover) scores
**1.000 on all 1000 bench rows** (0 abstain, 0 wrong, every turn index). The doc-recorded
residual (0.527 served vs 0.782 teacher, [wyly_babi.md](wyly_babi.md)) is **stale** — it
measured an earlier, weaker cover.

Why this is a property of the *bench*, not an achievement of the cover (all measured):

| measurement | value |
|---|---:|
| bench rows whose multi-sentence story-prefix appears **verbatim** in the training corpus | 65 / 110 unique |
| bench-internal duplicate story-prefixes | 801 / 1000 |
| **B0 on the deduplicated, non-verbatim subset** | **1.000 (41/41, 0 abstain)** |

qa1's configuration space (entities × locations × short histories) is tiny; even "novel"
story prefixes decompose into locally-seen states that count keys cover. The bench's
unseen-configuration mass is ≈ 0, so **residual is 0 by construction for any sufficiently
large memorizing cover** — it cannot distinguish binding from memorization. The 27-point
"generalization gap" narrative of the qa1 note no longer has an instrument.

## Tags

| Claim | Tag |
|---|---|
| B0 = 1.000 on current bench and on the deduped non-verbatim subset (41/41) | **empirical** |
| bench prefix-overlap 65/110, internal duplication 801/1000 | **empirical** |
| wyly_babi.md's 0.527/0.782 residual is stale for the current artifact | **empirical** (erratum added there) |
| H3 registered binary not confirmed; defect = unpinned baseline ensemble | **empirical** (registered outcome, verbatim) |
| query-shaped ≫ window-shaped admission marginal (≈24×) | **empirical (unregistered)** |
| moveloc verb mining (det ≥ 0.9, support ≥ 20 → {journeyed, moved, travelled, went, went back}) | **empirical** (unit-tested) |
| qa1 compositional binding vs memorization | **open** — needs a config-holdout bench (slice #80) |

## Handoff to slice #80 (advisor-vetted; pre-registered separately before any numbers)

Per the split-slice ruling: the slice that diagnosed the bench does not also deliver its
replacement's headline. #80 builds a **config-holdout qa1 bench** — freshly generated
stories with held-out **entity×location combinations** as the primary test block (unseen
joint configs of seen atoms: the compositional-binding question), plus a novel-entity-names
block as a *secondary diagnostic only* (open-vocabulary lexicon coverage, confounded by
token-keyed tiers and teacher weakness — gated conclusions never rest on it). H1/H2 rules
carry over unchanged against the residual that reappears there. The moveloc family,
mining, splits, and B0 instrumentation from this slice are the reusable substrate.

## Provenance / honesty

moveloc is hand-authored (`template_fixed`-class); `frac_induced` unaffected. All scoring
teacher-forced on deployment query tails; story-level splits asserted disjoint. B0 was the
on-disk package (regeneration not attempted — inventory and mtime reported by the
implementation lane; discrepancy vs the doc investigated and resolved as staleness plus
bench saturation, above).
