# Constraint-register build 2: legality certificate + admission (slice #91)

**Status:** measured + **proved over the stated domain** (2026-07-12), registered rules
applied verbatim, all promotion conditions independently reproduced by the architect. The
constraint/legality register is now **certified**: its legality feature is proved equal to
an independent Datalog program window-by-window, and its forced-move proposer's confidence
is promoted to certified-constant-1.0 under the strict registered gate. This is the
program's **first certified compute-register** — a rule whose *answer* is computed and
proved, not retrieved. `experiments/campaign_legality_certificate.py` (+30 tests; live
souffle test guarded for souffle-less environments). Built by the codex lane (its Soufflé
program authored independently of #90's tensor feature — a genuine cross-check).

## Registered outcome: PROMOTED (all three gates cleared)

| gate (registered) | result |
|---|---|
| three-way certificate: numpy oracle ≡ tensor feature ≡ **Soufflé**, full tuple incl. abstain rows | **9078/9078 = 100.000%, 0 mismatches** |
| corpus validity scan (every solution grid complete + all-different) | **23,461 scanned, 0 invalid — clean** |
| admission through the judge — held-out marginal / **regressions** | +0.090 / **0 regressions** |
| **CONF PROMOTION** (cert=100% ∧ validity clean ∧ regressions=0) | **PROMOTED — final_conf = 1.0** |

Independently reproduced by the architect (souffle on PATH, GPU host): identical numbers.

## What is proved, scoped exactly

- **`proved` OVER THE STATED DOMAIN** = row∪col∪box sudoku legality, these windows. The
  Soufflé program (integer division for cell-index recovery, per-unit candidate-count
  aggregation, `val()` token→digit normalization) agrees with the tensor feature on the
  full per-row output tuple including the −1/abstain rows (where parse bugs hide). **Not**
  "sudoku" universally; **not** natural-text.
- **The certificate covers the concrete instantiation, not the pluggable hook.** The scope
  (row∪col∪box) is a pluggable input, but a future *mined* scope reuses the harness and
  must earn its **own** cert run. Pluggability is never "certified for any scope."
- **conf=1.0 is proved *conditional on the validity scan.*** The cert proves feature ≡
  Datalog; that the forced value equals the corpus label additionally needs valid,
  uniquely-solvable grids (then naked-single value = solution value by all-different math).
  The scan (0 invalid over 23,461) discharges that premise — so the constant-1.0 confidence
  rests on a checked artifact, not an assumption.

## The honest scope caveat (propagated verbatim from #90, per lead condition 3)

The sudoku dataset presents **only late-reveal solution cells** (reveal index 60–80, 100%
last third). So the certified register recovers the **near-solved endgame** the memorizer
fails (0.85 solution-cell error → +0.090 marginal, 0 regressions), but its capability on
early/mid cells is **untested by this data**. The certificate proves the *machinery*
correct regardless of which cells; the *claim* is scoped to the endgame. A
uniform-reveal-index sudoku dataset (lead-priced, not chased) would make it a real
difficulty test.

## What this is / is not

- **IS:** the ergo bet's first certified concrete instance — a served answer that is
  *computed* (a count aggregate) and *proved* (Soufflé ≡ tensor ≡ oracle), memorization-free,
  admitted through the existing judge with zero new arbitration and zero regressions.
- **IS NOT:** solving sudoku (endgame only); learning the constraint (row∪col∪box scope is
  hand-authored, like `mate`/`depth` — only admission is learned); a natural-text result
  (sudoku-only, `LABELS=corpus`, no blending).

## Tags

| Claim | Tag |
|---|---|
| legality feature ≡ Datalog program, 9078/9078, over the stated domain (row∪col∪box sudoku) | **proved** (Soufflé certificate) |
| corpus solution grids valid (complete + all-different), 0/23,461 invalid | **proved** (exhaustive scan) |
| certified forced-move proposer: +0.090 marginal, 0 regressions on the sudoku cover | **empirical** |
| conf=1.0 proved conditional on the validity premise | **proved (conditional)** |
| register recovers the near-solved endgame; early/mid untested | **empirical** — scope caveat |
| register solves sudoku / handles early-mid / transfers to natural text | **NOT shown** |

## Process notes

Codex authored the Soufflé program independently of #90's tensor feature (a real
implementation cross-check, not a mirror of the same code). Its sandbox lacks GPU, so it
correctly declined to fabricate the admission/regression numbers and the wrapper verifier
ran that step on the host — no faked results. A benign environment bleed: the main-loop's
post-merge artifact-reminder surfaced inside the codex subagent as a spurious
"system-reminder"; codex correctly disregarded it (not its task). No action needed —
the reminder is scoped to the main loop.
