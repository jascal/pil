# Frontier-rows characterization (slice #87) — half the hypothesis confirms; the flips are not pure noise

**Status:** measured (2026-07-11), registered rules applied verbatim (workspace prereg
log). Steering item 1: test whether the stage-invariant gain/regression coupling
(#82 claim / #83 admit / #86 mine) is explained by new marginal signal living on
decision-boundary rows where any change flips rows both ways.
`experiments/campaign_frontier_rows.py` (+18 tests incl. AUC fixtures with known ground
truth, runner-up-trace parity, and a test_te-aliasing guard). Measurement only; frozen
B0′; #86's test_te untouched (stage 3 = val only, spy-asserted).

## Registered outcome: MIXED — and the decomposition is the finding

**Stage 1** (#81's gated1 on test_te; the only voting stage — 143 gains / 62 regressions,
cross-checked against #81's b/c counts row-by-row):

| clause | result |
|---|---|
| boundary clause (gain∪regression vs untouched, margin gap) | **AUC 0.763, p < 1e-4** — strongly holds |
| axis-uniformity clause (gain vs regression indistinguishable, ≤ 0.55 on all axes) | **fails** — AUC 0.597–0.613 on ALL FOUR axes (confidence, gap, teacher-consensus, decile), each p < 0.05, well-powered |

**The hypothesis splits in half.** (1) *Where* the coupling lives is confirmed: harvestable
signal concentrates hard at the incumbent's decision boundary — untouched rows are far
from it (0.763 separation). (2) *What the flips are* is refuted in its strong form: gains
and regressions are **not** symmetric noise — a weak but consistent, statistically solid
signal (~0.60 on every axis) distinguishes them. Not enough to clear the registered
discriminator bar (≥ 0.65 on any single axis), so per the registration: **MIXED**, and the
contract corollary correctly fires **no recommendation** in either direction.

Supporting stages, descriptive as registered: Stage 2 (cell-18, same mechanism at key
granularity) shows consistent point estimates (boundary 0.68) but is underpowered (N=18/13).
Stage 3 (#86's pooled judge-clearing grid points) missed the voting floor by one row
(19 gains vs the ≥20 floor — the floor doing its job) — and, notably, its **boundary AUC
is 0.488**: no boundary concentration on the *mining*-stage rows. The boundary story may
be claim-mechanism-specific; with 19/7 rows this is a descriptive flag, not a claim.

## What MIXED means here (and does not mean)

- The three registered negatives (#82/#83/#86) now share a **measured location**: their
  gains and regressions live at the incumbent's decision boundary. That much of the
  unification holds and goes into the ledger.
- The "any change flips rows symmetrically — safety is hopeless at this granularity"
  strong form is **not supported**: a ~0.60 four-axis signal separates paying rows from
  breaking rows. No single registered axis crosses 0.65 — but four weakly-informative
  axes at ~0.60 each raise the obvious priced question: does a **composite discriminator**
  (multivariate over the same axes, fit on val with its own registered test and an
  overfitting guard — ~200 rows is thin) clear the bar a single axis cannot? Priced, not
  built; trigger: any decision to re-attempt safe harvesting.
- Contract question: undecided by registration (MIXED ⇒ no motion). The regression-budget
  field stays specced-but-unrecommended; zero-regression growth is neither proven
  achievable nor priced impossible.

## Tags

| Claim | Tag |
|---|---|
| coupling rows concentrate at the decision boundary (stage 1: AUC 0.763, p<1e-4) | **empirical** |
| gains vs regressions weakly-but-significantly separable on all four axes (~0.60, p<0.05, powered) | **empirical** |
| registered verdict MIXED; no contract recommendation | **empirical** (verbatim) |
| boundary concentration at the mining stage | **open** (stage 3 descriptive: AUC 0.488 at 19/7 rows) |
| a composite discriminator clears 0.65 | **open** (priced; overfitting guard required) |
| stage-2/3 corroboration | descriptive only, as registered (mechanism duplication; floor) |

## Process notes

Third instance of the cross-device bug class this arc (CUDA index tensors vs CPU axis
tensors), caught — as before — only by running the real GPU campaign; fixed by contract
(extractors return CPU indices) and the lane proactively fixed a second un-flagged
instance. Convention extended: device-boundary contracts for index tensors, alongside the
tolerance rule for numeric compares. Meta-observation for the consolidation: #82, #83,
#86, #87 — four consecutive registered designs landing nuanced, well-powered,
non-definitive results. The arc is in a low-signal, safety-coupled regime, and the
registered-bands discipline is functioning exactly as intended: real effects get recorded
at their true size instead of being promoted or buried.
