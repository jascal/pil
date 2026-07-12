# Consolidation — PRs #76–#90 against the open-walls ledger

**Written 2026-07-12.** One note tying the #76–#90 arc back into
`WYLY_LM_ENDGAME_REVIEW_FABLE.md`'s open-walls list: what closed, what moved, what stays
open. Per-slice detail lives in the individual `docs/notes/*.md`; this is the ledger.
(The FABLE doc carries §10–§12 as the in-ledger integration and points here.)

## Closed

- **Join inducibility at the atom-language ceiling** (#77). The judge + SIG pipeline
  recovers planted joins *exactly at the atom language's expressibility bound* (regime S:
  comp 0.950 = bound 0.950, per-topology exact). The machinery is exonerated; the boundary
  is the atom **language** — per-var incidence atoms cannot compose held-out **star/hub**
  signatures (bound 0.000). CFQ (#75/#76) is closed for a compounding reason: its joins are
  underdetermined at the tested granularity (majority, lexical, and mined-type-merge all
  measured out; the merge family fails at its own oracle-occupancy ceiling).

- **Routing safety — the arc's single strongest result, negative at four granularities**
  (#81→#88). The real wikitext routing gain (gated rescue admits flat-declined families:
  **+0.0088, p≈5e-08**) is **not safely extractable** by any measured mechanism: claim-time
  gating leaves a leaky frontier (#82, no admissible point, safety costs ⅔ of the gain);
  admit-time key-retreat is defeated by gain/leak entanglement at the same keys (#83, 6.5%
  ban coverage); mine-time threshold relaxation clears the judge but carries regressions
  everywhere (#86, no zero-regression selection); and a multivariate discriminator over the
  coupling's own axes plateaus (#88, CV AUC 0.618 < 0.70, 0/100 rescued). **Contract
  answer:** zero-regression trusted-tier growth is not achievable at this granularity —
  price an explicit regression budget next to the certificate (field specced in #87; to
  ship on the next package-emit slice).

- **Admission determinism, shipped as contract** (#84/#85). "Run-variance" (#81 Finding 0)
  attributed: deterministic given (code, device, internal seed) — CPU≡GPU bit-identical,
  external seeds no-ops; the tier is a 4-rule stable core + a ~13-name internal-seed fringe
  worth ~0.006. Shipped (#85): `WYLY_SEED` exposed (default-preserving, guarded by a
  standing reproduction invariant on every future `wyly_lm_v5.py` edit), and per-rule
  `admission_stability` annotations in the manifest (domain-gated, null-never-0,
  round-tripped through the real serving loader).

- **The certifiability ladder to 500×** (#78): core 0.285→0.224 over 14M→6.9B inside the
  pre-registered band, crystallization 86.6%, library {1,2,3} scale-stable, dtype confound
  measured (gold Δ 0.0002 fp16↔int8 at 2.8b). 6.9B is this machine's hardware ceiling.

- **qa1 as an instrument** (#79): saturated by construction (1.000 even on the deduplicated
  non-verbatim subset; 65/110 prefixes verbatim in train). Replaced by the config-holdout
  generator (#80), which delivered the **binding-vs-memorization proof**: an admitted
  binding family recovers 100% of a held-out-configuration residual that the full memorizing
  tier covers at 0.000.

- **Mined frames "at scale"** (#86): existence-wise a *miner* fact (the interaction gate,
  not support, binds at both 410m and 2.8b; frame signal exists at 2.8b) — but unharvestable
  under the safety coupling. The "scale kills the family" reading is dead.

- **Hub-capable atoms / the #77 hub-star wall** (#93/#94). #93's CONSTRUCT arm bridges the
  typed output-vocabulary wall (fire-time assembly of train-mined slot tables generalizes
  on regime-S stars where memorized SIG plateaus). #94's LSTRUCT bridges the remaining
  lexical star wall via a position-based topology template. The count-aggregate framing
  from the earlier ledger is superseded by constructive assembly + structural templates.

- **khop schema gap** (#95 + rosetta #49). The certified khop / 2-hop program now emits as a
  package with mir==served parity; the rosetta-side schema lands in #49.

## Moved

- **The stage-invariant gain/regression coupling — the arc's central new object.** New
  marginal signal, wherever harvested, arrives entangled with regressions on
  previously-correct rows. #87 *located* it (coupled rows sit at the incumbent's decision
  boundary: gain∪regression vs untouched AUC 0.763, p<1e-4) and *characterized* its
  separability (gains vs regressions differ weakly-but-significantly on four axes, ~0.60,
  none ≥0.65; #88 showed they composite to only 0.618). So: real, boundary-located, not
  separable at the four measured axes. This reframes "trusted-tier growth on real text"
  from a mechanism search into a substrate property at this granularity — and prices the
  contract accordingly.

- **The constraint/legality register — triggered and building** (#89/#90). The first
  register that COMPUTES admissibility (a count aggregate over one variable's incidence
  set), the program's first concrete step from retrieval toward derivation (family-4
  ergo/claymore). #89 fired the trigger on sudoku (union-recoverable 0.980); #90 built the
  legality feature (parity-proven vs the numpy oracle) and froze the rule inventory to
  NAKED-ONLY. Honest scope: the sudoku dataset presents **only late-reveal cells** (index
  60–80), so the demo covers the near-solved endgame, not the solve; early/mid capability
  is untested. The constraint *scope* is hand-authored (like `mate`/`depth` for brackets) —
  only the judge's admission is learned. Novelty over mate is precise: the legality
  register's *answer* is computed, where mate derives a feature but retrieves the answer.

## Still open (recipe plateaus; achievability open, none "unbridgeable")

- **The relative/strata gate** — the one routing variant the #82 absolute-τ sweep bracketed
  but never measured; revisit only if constraint-register work makes sub-key features cheap.
- **A richer discriminator feature family** than the four #87 axes (#88 closed the composite
  over *those* axes).
- **Automatic cross-domain constraint discovery** — mine the constraint scope from data
  rather than hand-author it; the count-aggregate form + cert + judge are reused, scope
  discovery is the missing piece. Two known-answer testbeds (recover sudoku's scope; recover
  the bracket pairing+nesting). The big prize the hand-authored register is a step toward.
- **Wake-SGD-only 2-hop** (sleep-compile delivers the capability; a purity question).
- **The certifiability ladder past 6.9B** (compute-gated, not method-gated).
- **A uniform-reveal-index sudoku dataset** (lead-priced, not chased): would convert the
  near-tautological endgame regime into a real difficulty axis (the 0.19/0.30/0.59 gradient
  #89 measured) — a genuine early/mid-cell test of the legality register.

## Process ledger (standing lessons, all bit before)

Pin baseline ensembles *and* the val distribution (#79); pin admission-time vs eval-time
ensembles separately (#80); condition registered contrasts on non-degeneracy (#80); register
the val *selection objective*, not just the sweep grid (#82); a certified package is a
*specific rule set*, so reproducibility is a first-class contract (code, device, seed) with
per-rule stability annotated (#84/#85). Conventions: tolerance (not `==`) for cross-device
numeric compares; CPU-index contracts for cross-device tensors (each bug class hit ≥ twice).
Four consecutive registered non-definitive results (#82/#83/#86/#88, plus #87) are the
discipline recording thin, safety-coupled signal at true size — not four failures.
