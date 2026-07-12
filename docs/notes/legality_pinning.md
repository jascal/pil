# Constraint-register build 1: legality feature + inventory pinning (slice #90)

**Status:** measured (2026-07-12), registered rules applied verbatim; freeze verified,
feature parity-checked against the #89 oracle. First slice of the lead-approved
constraint/legality register — the legality **derived feature** (a count aggregate over a
cell's peers) built as a tensor computation, and the registered measurement that **freezes
the register's rule inventory**. No Soufflé certificate yet (#91); no battery hub arm yet
(#92). Sudoku only, `WYLY_LABELS=corpus`. **These are domain-solving numbers, NOT
natural-text; no natural-text/wikitext comparison appears in this slice** (lead condition 3).

## Registered outcome: inventory = NAKED-ONLY

| metric (sudoku, LABELS=corpus) | value |
|---|---|
| feature parity vs #89 numpy oracle | **9078 rows, 0 mismatches** |
| residual error rate (solution cells) | 0.845 |
| recovered_naked (fraction of residual errors) | 0.943 (n=1084) |
| recovered_union (naked∪hidden) | 0.980 (n=1126) |
| validity floor (union ≥ 50 errors) | passed |
| **FREEZE: naked ≥ 0.90 × union?** | 0.943 ≥ 0.882 → **NAKED-ONLY** |
| proposer measured ValFiredAccuracy (not hardcoded 1.0) | 1.000 |
| proposer held-out marginal (descriptive, non-gating) | +0.094 |

**The register's rule inventory is frozen to NAKED SINGLES.** Condition 2 (the lead's
lane-0.94-vs-architect-0.70 discrepancy) resolved by measuring the *recovered-fraction*
gap, not the classification-count gap: naked singles recover 96% of what naked∪hidden
recovers, so hidden singles are not needed. The register computes naked-single legality
only — the simpler construct.

## The headline caveat — the dataset only tests the near-solved endgame

The reveal-index distribution of *every* sudoku solution-cell prediction target
(n=9078): **min 60, max 80, median 70 — 100% in the last third.** Early and mid cells
**never appear as prediction targets** (an independent architect reconstruction found the
same: 59–80 across the first-6000 and a strided whole-dataset sample). So:

- The register is measured **only on cells where ~73 of 81 are already filled** — the most
  trivially-forced sub-task. Recovery ≈ 0.98 is near-tautological at that reveal depth.
- Its capability on **early/mid cells (where forcedness is genuinely low) is UNTESTED by
  this data.** The stratification the lead mandated to quantify the easy-win could not
  stratify — because the data contains only one stratum. That *is* the maximally honest
  form of the easy-win caveat: the demo covers the endgame, not the solve.
- **Not a bug and not a blocker** for the machinery: #91's certificate proves the count
  feature ≡ its Datalog program window-by-window *regardless of which cells*, so it is a
  valid proof of the machinery; the CLAIM is scoped to "recovers the near-solved endgame."
  A genuine early/mid-cell test needs a harder dataset (a possible later slice, not
  required for #91 or #92).

## What is / isn't being shown (framing, per condition 3 + the hand-authored precision)

- **Shown:** certified-derivation machinery — a count-aggregate legality feature, a
  forced-move proposer that slots into the existing judge with zero new arbitration
  (measured fired-accuracy 1.0), recovering residual the memorizer fails (its 0.845 error
  is the domain it cannot solve; the register solves the endgame of it). This is the ergo
  bet's first concrete step: an answer *computed*, not retrieved.
- **NOT shown:** solving sudoku (only the near-solved endgame is in the data); learning the
  constraint (the row/col/box scope is HAND-AUTHORED, like `bracket_sets` for mate — only
  the judge's admission is learned); early/mid-cell capability (untested).
- **Independent of this slice's data limitation:** the hub-atom proof (#92) tests the same
  count-aggregate atom against planted stars in the join battery — the sudoku late-cell
  quirk cannot touch it.

## Tags

| Claim | Tag |
|---|---|
| tensor legality feature ≡ #89 numpy oracle (0 mismatches / 9078) | **empirical** |
| register inventory = NAKED-ONLY (recovery ratio 0.943/0.980 ≥ 0.90) | **empirical** (registered freeze) |
| dataset tests only late-reveal cells (index 60–80, 100% last third) | **empirical** — scope caveat |
| register recovers the near-solved endgame the memorizer fails | **empirical** (late cells only) |
| register solves sudoku / handles early-mid cells | **NOT shown** — untested by this data |
| constraint scope is hand-authored, not learned | **stated** (framing) |

## Honesty notes

Feature computed, not yet certified (cert = #91, so proposer confidence is *measured*
1.0, not asserted). Sudoku only; `LABELS=corpus`; no natural-text comparison. Freeze
decided by the recovery ratio alone; admission marginal descriptive, non-gating. Small
run-to-run residual-count variance from cover retraining (verdict unaffected).
