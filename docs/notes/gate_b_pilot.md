# Gate (b) pilot — FIRES: beam-energy separates the forced continuation (cross-vendor) (slice #101)

**Status:** measured + registered, **cross-vendor confirmed** (2026-07-13). The decisive
measure-first pilot of the SequenceEnergyScore arc (gate (a) SIGNED). Pre-registered in
`PIL_GATE_B_PILOT_PREREG.md` (SIGNED; CORRECTION v2 = propagation-aligned beam, v3 = pinned
turnstile-margin energy — both before any valid numbers). `experiments/campaign_gate_b_pilot.py`
(grok) + `experiments/gate_b_pilot_codex.py` (codex, independent) + tests. **Verdict: FIRES →
the beam build is justified; the thread does NOT end.**

## What was tested
Does the signed EnergyStep — an M-step legality-pruned beam with the **pinned PIC turnstile margin**
(`⟦legal⟧(v) − max⟦legal⟧(w)` = +1 forced / 0 guessed, weakest-link, det_rank tie-break) — commit
the known-forced sudoku value where a step-1 per-token confidence baseline cannot, stratified by
constraint-propagation depth `d`? The beam and the ground-truth `d` share one `propagate_one_pass`
primitive (whole-board naked+hidden singles), so they cannot drift.

## Result — FIRES, cross-vendor
Both lanes' per-depth `beam_acc(M)` show the registered ground-truth signature: **beam_acc → 1.0
exactly when M reaches the cell's `d`**, and collapses to baseline on never-forced cells.

| d | M=1 | M=2 | M=3 | M=4 | M=5 |
|---|---|---|---|---|---|
| 2 | 0.78 | **1.00** | 1.00 | 1.00 | 1.00 |
| 3 | 0.40 | 0.78 | **1.00** | 1.00 | 1.00 |
| 4 | 0.43 | 0.45 | 0.80 | **1.00** | 1.00 |
| 5 | 0.64 | 0.64 | 0.73 | 0.82 | **1.00** |
| never | 0.46 | — | — | — | 0.47 |

| lane | population n | conf_acc | deciding M | SEPARATION | verdict |
|---|---|---|---|---|---|
| codex (held-out train prior) | 779 | 0.78 | 2 | **+0.221** | FIRES |
| grok (full-corpus prior) | ~9,678 | 0.84 | 3 | **+0.165** | FIRES |

Both clear `SEPARATION ≥ 0.15` and `beam_acc ≥ 0.50` with the rising signature. Robust across
independent implementations, populations, and baseline constructions. codex fully self-verified
(5 unit tests incl. `d=2 → beam(M≥2) forces gold`; byte-identical re-run; hand-verified prune);
grok's identical 1.0-at-M=d signature + pinned-margin/whole-board-propagation spot-verified.
Fast (~34s, no cover regeneration). ruff clean; 427 tests pass.

## Honest scope + caveats (tags)
| Claim | Tag |
|---|---|
| the pinned-energy propagation beam commits gold on 100% of forced-within-M cells, beating per-token by ≥0.15 (rising with M) | **empirical**, cross-vendor — FIRES |
| the sequence view (M-step lookahead) captures multi-step forcing per-token misses | **empirical** — validated on ground truth |
| the win is a *surprise* over a weak baseline | **NO** — the sudoku per-token baseline is already strong (0.78+); this is validation on known ground truth, with the beam's edge largest on deeper cells (per-token ~0.4) |
| the certified register (hard term) is *necessary* for separation | **NOT shown** — the UNCONSTRAINED arm (wikitext, no hard prune) also separated (+0.22–0.28), BUT over a weak baseline (conf 0.08) and tiny n=97; a caveated side-finding ("soft margin energy may suffice — worth a real test"), not a conclusion. The legality prune fired 0× within M≤5 (working-but-unexercised, verified). |

## Consequence (per the prereg's FIRES branch)
The beam build is escalated as a commitment boundary (the gate-(a) algebra → code, #90-style
discipline). This unlocks energy-mode admission (`DECIDE_ENERGY`) and the three-levers batching
generalization. The committed token would be certified over the M-step legal lookahead (bounded
beam, not optimal).

## Process (lane balancing)
Raced: grok primary + codex independent, both from scratch. Run 1 was confounded by a spec bug
(row-major beam vs any-cell propagation-depth ground truth — my error, caught by the race +
verification, no valid numbers read); the lead's mid-slice note caught that the energy must be the
pinned turnstile margin, not an ad-hoc proxy. Both corrected before numbers; the registered run is
this one. See [[lane-balancing-rule]], [[sequence-energy-direction]].
