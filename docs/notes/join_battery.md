# Synthetic join battery — planted topology, existing judge (slice #77)

**Status:** measured (2026-07-11). Direct successor to the CFQ modality-gap probes
([#75](cfq_join_headroom.md), [#76](cfq_typed_join.md)): CFQ could not say whether the
induction machinery *could* recover joins, because CFQ's joins were not demonstrably there
to recover. This battery plants them. Generator `pil/join_battery.py` (regime **S** is
*self-certifying*: every emitted query asserts planted signatures == type-merge closure),
campaign `experiments/campaign_join_battery.py`. Admission runs through the **existing**
`ResidualFamily.admit` held-out-marginal judge (`celf=False`, `thresh=1e-4`,
`max_rules=128`) over five proposer arms encoded as `MapDict` join atoms; scoring reuses
`parse_sparql_joins` / `join_f1` verbatim (the generator emits CFQ-format SPARQL).
Worlds satisfy a printed-and-asserted **expressibility invariant** (distinct planted
signatures ≤ 100 ≤ budget), adopted as a pre-verdict correction after a budget-confounded
v1 smoke (recorded in the workspace prereg log before any resized-world numbers).

## Pre-registered rule — reported verbatim

> "Machinery CAN induce planted joins" iff ≥1 arm in its matched regime (S:
> majority/typed/SIG; L: lexical/SIGW) reaches test join_f1 ≥ 0.9 IID AND ≥ 0.7
> compositional, AND that arm's N-control is clean (n_admitted ≤ 1 AND N test gain ≤ 0.02).

**Outcome: NOT met.** S:SIG passed both thresholds (1.000 IID / 0.950 comp) but no arm
had a clean N-control; L's matched arms failed the compositional leg. No verdict flip is
made — the leg-by-leg decomposition below is where the information is, and each diagnosis
is *measured*, not asserted.

## Scoreboard (seed 0; join_f1; empty-maps baseline = 0.000 everywhere)

| regime | arm | n_adm | IID | comp | comp bound | N gain | perm adm | perm gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S | majority | 74 | **1.000** | 0.000 | — | — | 23 | 0.011 |
| S | typed | 74 | **1.000** | 0.000 | — | — | 23 | 0.011 |
| S | lexical | 19 | 0.370 | 0.197 | — | — | 14 | 0.049 |
| S | **SIG** | 66 | **1.000** | **0.950** | 0.950 | — | 27 | 0.025 |
| S | SIGW | 66 | **1.000** | 0.825 | 0.950 | — | 27 | 0.025 |
| L | majority | 15 | 0.498 | 0.000 | — | — | 12 | 0.041 |
| L | lexical | 3 | 0.142 | 0.000 | — | — | 3 | 0.072 |
| L | SIG | 21 | 0.580 | 0.334 | 0.508 | — | 19 | 0.080 |
| L | **SIGW** | 28 | **0.930** | 0.429 | 0.508 | — | 22 | 0.077 |
| N | *inventory ceiling* | — | *0.645* | — | — | — | — | — |
| N | majority | 20 | 0.517 | — | — | 0.517 | 13 | 0.034 |
| N | SIG | 28 | 0.589 | — | — | 0.589 | 15 | 0.071 |
| N | SIGW | 39 | 0.624 | — | — | 0.624 | 17 | 0.072 |

(typed in L/N admits nothing — the T=1 world gives the assembler no types to mine;
its S row is structurally ≡ majority, a designed redundancy check: self-certification ⇒
zero within-key ambiguity ⇒ assembler output = per-key majority.)

## The four measured findings

**1. Where the atom language can express the planted joins, the judge+SIG pipeline
recovers them at the language's ceiling.** S: SIG 1.000 IID; compositional 0.9500 vs
expressibility bound 0.9500 — saturation is exact *per topology* (chain 0.9385 = bound
0.9385; cycle 1.0000 = bound 1.0000). Majority memorizes (1.000 IID / 0.000 comp), SIG
composes — the designed contrast. **Empirical.**

**2. The atom language has a measured expressibility boundary at hub variables.**
A per-var incidence atom *is* its variable's full incidence multiset, so a star (hub)
signature spans the whole predicate combination — a held-out combo's star signature
cannot exist in train. L compositional: star rows bound 0.0000 (0/492 rows expressible),
chain rows bound 1.0000; overall bound 0.5080 vs SIGW's measured 0.4286 (≈84% of
ceiling, chains at 0.8437). Per-var incidence atoms compose across queries for low-degree
variables (chains, cycles), not hubs. **Empirical.** (S's high comp is consistent, not a
counterexample: its holdout drew no star combos — 813 chain / 187 cycle rows.)

**3. The registered N-control was mis-specified — it measures partial-structure recovery,
not hallucination.** Regime N randomizes topology given the predicate multiset, but the
signature *inventory* per multiset is still determinate. The full-inventory predictor
(per-key multiset-max of train gold) scores 0.6449 on N test; SIG (0.5891) and majority
(0.5170) sit *under* that ceiling — the judge recovered real determinate substructure,
which the registered control then counted against every arm. The rule's "CANNOT" outcome
therefore localizes to control design, not machinery. **Empirical** (ceiling printed by
the campaign as the `inventory-ceiling` row).

**4. Judge selectivity at `thresh=1e-4` is weak against compact partial-credit worlds
(permutation addendum).** The addendum answers a different question than the N columns —
*does the judge admit structure that is not there?* — with val/test gold permuted across
rows (fixed seeds). Its expectation, registered before the numbers (n_admitted ≤ 2 AND
perm gain ≤ 0.05 per arm): **NOT met** — most arms admit 12–27 rules under permuted gold,
and several exceed the gain leg too (up to 0.080). Mechanism: `join_f1` gives partial
credit, and in a compact world (e.g. 66 signatures / 74 keys) frequently-correct
signature fragments overlap permuted gold often enough to clear a 1e-4 marginal.
Magnitude separation from real signal stays large (real val gains 0.38–1.00, i.e.
~5–40×), but rule-count selectivity is poor. **Empirical.** Implication (open): admission
thresholds should be calibrated to the gate metric's permutation noise floor rather than
fixed at 1e-4; a per-run permuted-gold floor is cheap to compute.

## Tags

| Claim | Tag |
|---|---|
| generator self-certification (S planted == type-merge closure), expressibility invariant, atoms/judge wiring | **empirical** (17 unit tests; asserts at generation time) |
| judge+SIG recovers planted joins at the atom-language ceiling in S (exact bound saturation) | **empirical** |
| per-var incidence atoms do not compose for hub/star signatures (L star bound 0.000) | **empirical** |
| regime-N gain = determinate-inventory recovery (ceiling 0.6449 bounds all arms) | **empirical** |
| judge admits 12–27 spurious tiny-marginal rules under permuted gold at thresh=1e-4 | **empirical** |
| pre-registered rule verdict ("not met") | **empirical** (reported verbatim; no flip) |
| principled threshold calibration (permutation-floor-scaled) fixes selectivity | **open** |
| hub-composable atom representation (e.g. typed hub abstraction) closes the L comp gap | **open** |

## Provenance / honesty

All five arms are fixed templates over train-mined content — `template_fixed`-class per
the `candidate_provenance` precedent. **No battery result changes `frac_induced`**, and
this note deliberately says the machinery *recovers* planted joins ("induced" is a
reserved provenance term in this program). Atoms are mined from train only; val/test gold
flows only into `join_f1`; the expander reads only (maps, question tokens, predicate key).

## What this means for the program

- **The CFQ negatives are now contextualized by a capability result.** The same judge,
  gate metric, and atom machinery that scored ~0 on CFQ joins recover planted joins at
  the representation ceiling when determinate structure exists. CFQ's join failure is a
  property of the data at the tested feature granularity (underdetermination, #75/#76)
  plus the hub-expressibility limit (finding 2) — not a judge failure. `frac_induced=0.00`
  on CFQ reads accordingly: a measurement of the data/representation pair, not an
  indictment of the admission machinery.
- **The incidence-native atom (SIG) was the strongest arm in every regime** — consistent
  with the PIC incidence-prior hypothesis *at the atom level* in a synthetic world;
  transfer to real text remains **open**.
- Two concrete, measured levers for any future slice (not this one): hub-composable atom
  representations, and permutation-floor threshold calibration for `ResidualFamily.admit`.
  Per the standing queue, the battery slice stops here and work re-anchors to the endgame
  trunk (real-text trusted-tier families; the 6.9b teacher ladder rung).
