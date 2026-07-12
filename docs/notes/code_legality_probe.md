# Code compute-register probe — bracket legality does NOT recover code residual (DEAD)

**Status:** measured + registered, **cross-vendor confirmed** (2026-07-12). Lead-directed
measure-first slice (trunk re-assessment item 1: test the computed route on code). Pre-registered
in `PIL_CODE_REGISTER_PREREG.md` BEFORE numbers — thresholds fixed there, never tuned.
`experiments/campaign_code_legality_probe.py` (+ tests; + `experiments/verify_code_legality_gate.py`,
the independent codex cross-check).

## Setup (a clean #89-style headroom test)
The code cover (`WYLY_DS=code`, mined/cover/sw, `LABELS=corpus`) admits only STATISTICAL rules
(induction/kgram/mined-frames — **no bracket rules**), while `mate_feature` (the innermost
unclosed opener's forced closer) is **already Datalog-certified** (256/256 proved,
`wyly_mate_certify.py`). So: does a crisp, certified legality register recover residual the
statistical cover misses?

## Result: DEAD (cross-vendor identical)
| | grok probe | codex independent |
|---|---|---|
| regenerated core_sw agree | 0.5742 | 0.5739 |
| n_residual (te) | 5109 | 5113 |
| **n_mate_recoverable** | **76** | **76** |
| **GATE** = recoverable / all-residual | **0.0149** | **0.014864** |
| **verdict** (FIRES≥0.30 / DEAD<0.10) | **DEAD** | **DEAD** |

Two independent implementations (different lanes, different classifier code, independent
regeneration) recovered **exactly 76** rows. codex sanity check "every recoverable row is
bracket-relevant" PASS.

## Why DEAD is honest, not a sparsity artifact (registered scoping numbers)
- residual_rate = 0.4257 (5109 rows) — far above the 0.02 abort; plenty to measure.
- Even within bracket-TOUCHING residual, mate recovers only a minority — 11.7% (76/649, grok's
  gold-or-wrong-pred-is-bracket denominator) to 25.9% (76/293, codex's narrower denominator).
  The denominator definition is a secondary-lens judgment; **both readings are DEAD and neither
  touches the primary gate.** So bracket legality is a NARROW register (it only forces the
  matching closer at forced positions), and code residual is dominated by non-bracket
  (identifier / operator / semantic) mass.
- CONTROL: the 76 recoverable rows are not statistically separable either (composite #88 router
  AUC 0.48, perm-p 0.48) — consistent, though moot at n=76. `teacher_consensus` axis omitted
  (no 8-teacher code ensemble) — reported, not faked.

## Interpretation (tag-disciplined)
| Claim | Tag |
|---|---|
| `mate`/`depth` force/flag correctly | **proved** (Datalog certificates, prior slices) |
| bracket legality recovers the code cover's residual | **REFUTED** — GATE 0.0149 DEAD, cross-vendor identical (76/≈5110) |
| the code cover has recoverable legality residual (via brackets) | **empirical negative** |
| "code compute-registers don't work" / "the computed route doesn't reach code" | **NOT shown** — ONLY brackets were tested; scope-legality and indent-consistency registers are UNTESTED (they'd need building) |

**Bottom line:** the *cheapest, already-certified* code register (brackets) is a dead lever on
code residual. This does **not** refute the broader computed-route-on-code hypothesis — it prices
the cheapest probe of it as negative and localizes the open question to richer registers
(scope legality, indent consistency), which remain unbuilt/untested.

## Process note (lane balancing)
Built by grok; independently re-implemented and confirmed by codex (its own mate-recoverable
classifier, reusing the cached cover state). First application of the standing lane-balancing
rule ([[lane-balancing-rule]]): a load-bearing negative gets a cross-vendor cross-check before
it is trusted. Both lanes agreed to the row.
