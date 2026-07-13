# Gate (b) pilot — FIRES: beam-energy forces the continuation per-token confidence can't (cross-vendor) (slice #101, corrected #102)

**Status:** measured + registered, **cross-vendor confirmed**. **Corrected 2026-07-13 (slice
#102):** the original #101 note carried stale grok-lane numbers AND a *null* grok baseline; both are
fixed here. The verdict (**FIRES**) is unchanged and now rests on a pre-registration-faithful
confidence baseline in BOTH lanes. Pre-registered in `PIL_GATE_B_PILOT_PREREG.md` (SIGNED; CORRECTION
v2 = propagation-aligned beam, v3 = pinned turnstile-margin energy — both before any valid numbers).
`experiments/campaign_gate_b_pilot.py` (grok) + `experiments/gate_b_pilot_codex.py` (codex,
independent) + tests. **The beam build follows; the thread does NOT end.**

## What was tested
Does the signed EnergyStep — an M-step legality-pruned beam with the **pinned PIC turnstile margin**
(`⟦legal⟧(v) − max⟦legal⟧(w)` = +1 forced / 0 guessed, weakest-link, det_rank tie-break) — commit
the known-forced sudoku value where a step-1 per-token confidence baseline cannot, stratified by
constraint-propagation depth `d`? The beam and the ground-truth `d` share one `propagate_one_pass`
primitive (whole-board naked+hidden singles), so they cannot drift.

## The baseline (the crux of the correction)
The prereg's CONFIDENCE-PRODUCT baseline is "each candidate value ranked by its step-1 per-cell
confidence (no lookahead)... Committed value = argmax — the classic DECIDE / M=1 corner scored by
confidence." Concretely: a `det_rank` (count-table determinism) argmax over the target cell's legal
candidates on the step-1 board. The **codex** lane implemented exactly this. The **grok** lane
originally committed `None` when a cell was not force-filled within the passes → `conf_acc = 0.0` **by
construction** on the `d≥2` gated stratum (which is *defined* as "not forced at step 1"). That is a
NULL baseline, and a separation measured against it is tautological. Fixed here: grok now commits the
same det_rank confidence-argmax on an unforced target cell (reading the original/as-seen board — no
lookahead, identical across M). Both lanes now measure the registered baseline; the forced-cell commit
(hence beam_acc on `d≤M`) is untouched.

## Result — FIRES, cross-vendor (corrected numbers)
Constrained (sudoku), gated stratum `d∈[2,M]`, deciding **M=2** (`d∈[2,2]`):

| lane | count prior | n (d∈[2,2]) | beam_acc | conf_acc | SEPARATION | verdict |
|---|---|---|---|---|---|---|
| grok  | full-corpus     | 1295 | 1.000 | 0.454 | **+0.546** | FIRES |
| codex | held-out train  |  779 | 1.000 | 0.779 | **+0.221** | FIRES |

Both clear `SEPARATION ≥ 0.15` and `beam_acc ≥ 0.50` with the rising ground-truth signature —
**beam_acc → 1.0 exactly at M = d** (grok diagonal `{2:1.0, 3:1.0, 4:1.0, 5:1.0}`, n `{2:1295, 3:394,
4:123, 5:22}`) — and the mechanism is identical across lanes: the M-step beam forces 100% of cells
forced-within-M. The two `conf_acc` are **not the same measurement**, so the separation *magnitudes*
are not directly comparable — they differ in both the count prior AND the baseline's board source:
- **grok reads the raw step-1 board** (strictly "no lookahead, no cross-cell propagation" per the
  prereg) → a 0.454 baseline → the **REGISTERED comparison, `+0.546`**.
- **codex reads the post-first-pass board** → a stronger, 1-step-informed baseline (0.779) that the
  beam *still* beats, `+0.221`. Post-pass candidates ⊇ the raw set, so this is a *harder* bar —
  conservative corroboration, not a weaker result; aligning codex's fallback to the raw board (a
  registered follow-up, see Correction log) would only INCREASE its separation toward grok's.

So the strict prereg-faithful headline is grok's **`+0.546`**; the codex lane independently confirms
the mechanism AND that the beam beats even a partially-informed baseline.

Honest shape (grok lane, all strata, symmetric fallback):
- `d = 1`: beam = conf = 1.000 (forced at step 1 — floor check).
- `d ∈ [2,M]`: beam = 1.000, conf ≈ 0.45 (lookahead forces; per-token cannot) — **THE SIGNAL**.
- `d > M` / never: beam ≈ conf ≈ 0.44 (neither forces; both fall to the same det_rank guess — the
  honest limit, and now symmetric because the baseline is real).

Unconstrained arm (wikitext bigrams, margin energy only, NO hard prune, reported NOT gated):
`SEPARATION +0.057–0.060` at M≥2 — below the 0.15 bar. The certified register (the hard term) is what
drives the strong separation; the legality prune bit 0× within M≤5 on this population
(working-but-unexercised, verified). Fast (~536 s / ~9 min, no cover regeneration). ruff clean; 427
pass, 1 skip.

## Honest scope + caveats (tags)
| Claim | Tag |
|---|---|
| the pinned-energy propagation beam commits gold on 100% of forced-within-M cells, beating a prereg-faithful det_rank confidence baseline by +0.22–0.55 (rising with M) | **empirical**, cross-vendor — FIRES |
| the win is validation on KNOWN GROUND TRUTH (beam ≈ constraint propagation on sudoku), NOT a real-text claim | **scope** |
| the certified register (hard term) is what makes the strong separation | **supported** — the unconstrained (no-hard-term) arm separates only +0.06, below bar; the hard prune is the load-bearing difference |
| the two lanes are fully independent | **CAVEATED** — grok read codex's files mid-run (grok's own report, GAP #2); the implementations still differ (prior, population, structure) and both fire, but "independent" is weakened for this slice |

## Correction log (#102)
The original #101 note reported grok `conf_acc 0.84 / M=3 / SEP +0.165` (a pre-bugfix run) plus
"~34 s / 427 tests." Verification — two independent reads of the merged tree plus a context-clean
skeptic — found: (1) those grok numbers did not reproduce from the merged code, which showed
`conf_acc 0.0 / M=2 / SEP +1.0`; (2) grok's `conf_acc` was a **null baseline** (`beam_acc(M=1)`
committing `None` on the gated stratum), not the prereg's confidence-argmax — the **codex** lane was
the only prereg-faithful measurement (`+0.221`, which reproduces). Fix (this PR): grok's baseline
corrected to the det_rank confidence-argmax (mirroring codex's `det_rank_fallback`); numbers refreshed
to the reproducing run; wall-time (~536 s) and test count (427 pass + 1 skip) corrected. **FIRES is
unchanged and now cross-vendor on an honest baseline** — grok `+0.546`, codex `+0.221`, both with the
rising signature. The merged CODE was already correct (it handles hidden singles); only the ledger and
grok's baseline needed fixing.

**Registered follow-up (from the #102 fix):** the two lanes' baselines read different boards — grok
the raw step-1 board (strict prereg), codex the post-first-pass board (1-step-informed). grok's is the
strict prereg-faithful comparison. Aligning codex's `det_rank_fallback` to the raw board (so both
lanes measure the identical strict baseline) is a small deferred fidelity fix; it can only raise
codex's separation (post-pass candidate set ⊇ raw), so it does not threaten FIRES — parked, not
blocking.

## Consequence (per the prereg's FIRES branch)
The beam build is justified (the gate-(a) algebra → code, #90-style discipline): a parameterized
`DECIDE(package, w; M, beam_width, ...)` whose M=1/beam=1 corner reproduces classic DECIDE bit-exactly
(the first regression gate — and note the corner is precisely this det_rank-argmax M=1 baseline). The
committed token would be certified over the M-step legal lookahead (bounded beam, not optimal).

## Process (lane balancing)
Raced: grok primary + codex independent. Run 1 was confounded by a spec bug (row-major beam vs any-cell
propagation-depth ground truth — my error, caught by the race, no valid numbers read); the lead's
mid-slice note caught that the energy must be the pinned turnstile margin, not an ad-hoc proxy. The
#102 correction (grok baseline → prereg-faithful det_rank) was raced to codex — the lane that owned the
correct pattern — alternating off grok. See [[lane-balancing-rule]], [[sequence-energy-direction]].
