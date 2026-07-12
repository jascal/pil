# Sudoku forced-move trigger probe (slice #89) — trigger FIRES; but read it honestly

**Status:** measured (2026-07-12), registered gate applied verbatim; verdict cross-checked
by an independent grid reconstruction. The measure-first probe gating the constraint/legality
register build. `experiments/campaign_sudoku_forced_move.py` (+17 tests). No register built,
no certified feature — plain-numpy forcedness from the in-window solution grid.

## Registered outcome: FIRES

| metric (LABELS=corpus, n=1359 solution-cell rows) | value |
|---|---|
| cover solution-cell error rate (Step-0 abort bar 0.02) | **0.846** — cover barely solves sudoku |
| board-level train/test overlap (Step-0 contamination) | 0.0003 (negligible) |
| base rate: union (naked∪hidden) forced over ALL rows | 0.980 |
| residual union-forced fraction | 0.980 |
| **GATE: absolute union-recoverable fraction of residual errors** | **0.980 → FIRES (≥0.30)** |
| non-overlapping-board subset | same → FIRES |

**Independent architect check** (reconstructed grids from `corpus_sudoku.txt`, peers =
puzzle clues ∪ revealed row-major prefix): union ≈ **1.000** (naked 0.70 / hidden 1.00),
confirming the lane's ~0.98 union. The trigger is robust.

## The honest reading — not "special residual structure"

The base rate equals the residual rate (both 0.980): the residual is **not enriched** for
forced moves; almost *every* solution cell is forced, whether the cover gets it right or
wrong. That reframes the result:

- **The memorizer cannot do sudoku** (0.846 error on solution cells under corpus gold; it
  wins its 0.35 aggregate on the copyable PUZZLE echo/headers, not the solve).
- **The generated puzzles are single-step-solvable in reveal order.** Revealed row-major on
  top of the clues, each next cell is logically forced by what is in-window — the reveal
  order does the propagation, so naked/hidden singles nearly suffice (no search).
- So the register would recover ~all of the cover's error mass — **the ergo bet's cleanest
  demonstration**: a certified, memorization-free count aggregate DERIVES answers the
  memorizer fails wholesale. But it is an **easy win**: generator-easy puzzles + reveal
  order, not a hard residual pocket. State it as "derivation replaces the memorizer on a
  solvable task," never "patches special structure."

## Build scoping (from the breakdown, as registered)

- **naked-vs-hidden is unsettled between the two measurements**: the lane read naked=0.94
  (naked nearly suffices); the architect check read naked=0.70, hidden→1.00 (hidden needed).
  The two differ on peer-set/hidden-single definitions. **Pin this at build time** — it
  decides whether the register computes naked singles only or naked∪hidden.
- **Reveal-position is the real difficulty axis.** Naked-only without clues rose 0.19 →
  0.30 → 0.59 across reveal thirds (architect check): early cells are far less forced. The
  register's genuine test is early/mid cells, and a harder puzzle distribution (requiring
  chains) would move it toward the DEAD regime. The current corpus is the easy end.

## Tags

| Claim | Tag |
|---|---|
| trigger FIRES: union-recoverable ≈ 0.98 of residual errors (lane + independent check) | **empirical** |
| base rate = residual rate: forcedness is uniform, not residual-enriched | **empirical** |
| memorizer fails sudoku solving (0.846 error, corpus gold); board overlap negligible | **empirical** |
| the register would work here as certified derivation replacing memorization | **empirical** (probe-level; the build would prove it) |
| the win is "easy" (generator-easy puzzles + reveal order); harder distributions untested | **open** |
| naked-only vs naked∪hidden sufficiency | **open** — pin at build time |

## Build decision (escalated — a commitment boundary)

FIRES → the register is justified by the registered rule. But given (a) the easy-win
framing, (b) that this is a large 3-axis build (2D count feature, new Soufflé program,
battery `expand()` change for the hub side), and (c) that steering parked this thread
pending exactly this trigger — the go/no-go on the build is surfaced to the research lead /
user with the honest number, not auto-started. Recommendation: **build, as the ergo-bet
demonstration and the shared count-aggregate vehicle for hub-capable joins** — but scope
the build's value test to early/mid reveal cells (where forcedness is real, not trivial)
and pin the naked-vs-hidden question first.

## Honesty notes

Plain-numpy forcedness, no certified feature; `WYLY_LABELS=corpus` (the domain note's 0.520
core is teacher-imitation — wrong gold for a constraint question); overlap-adjusted subset
reported; v5 untouched. The prereg lives in the workspace log (PIL_CFQ_JOIN_PREREG.md), not
a repo file — the lane correctly printed the registered classification defs verbatim in the
campaign as the in-repo record.
