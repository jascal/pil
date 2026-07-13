# SequenceEnergyScore arc — consolidation (#101–#108): the beam's value is the certified hard term

**Committed ledger wrap-up (2026-07-13), per the lead's step (b).** Covers the whole energy arc + the
register/battery context it sits on. The arc is **built + validated end-to-end**; it now PAUSES for the
lead's what's-next reassessment (candidates in `WYLY_TRUNK_REASSESSMENT.md`).

## What the arc was
A user-designed `SequenceEnergyScore`: ONE parameterized `DECIDE(package, w; M, beam_width, ...)` whose
degenerate M=1/beam=1 corner is exactly classic DECIDE (so admit⟺serve stays near-tautological). Energy =
the PIC turnstile MARGIN (integer, float-free, Datalog-exportable), NOT −log(conf). The
constraint/legality register (#89–#92 — the program's first CERTIFIED compute-register, the ergo bet's
concrete instance) is the first HARD energy term: legality prunes the beam, soft `det_rank` margins rank
survivors = constrained beam decode. The whole point was to test the **split-by-hard-term-availability
decomposition** — does M-step lookahead help, and does its help come from the hard term or the soft margin?

## The build (#103–#106) — each objective-gated, cross-repo, corner-preserving
- **#103 corner:** a new `cover:"energy-beam"` serve mode whose M=1/beam=1 reduces to `serve_sw`
  BIT-EXACTLY (gap 0.0 / 11,477 windows). The admit⟺serve near-tautology made real; protects every prior
  result.
- **#104 schema:** additive raw `(cnt,tot)` + `alpha` + `schema_version=3`, gated to energy-mode emits;
  dual-runtime (pyspoke + sgiandubh C++), all four lead conditions met, zero decision change. (The
  integer `det_rank` needs raw counts; shrunk floats can't be un-shrunk.)
- **#105 engine:** a SUBSTRATE-AGNOSTIC `beam_decode(oracle, M, beam_width, rule_id, seed)` in rosetta
  (`expand`/`extract_commit`/`fallback`; `margin:float`, `counts:None`⇒tie; determinism contract).
  Reproduces the #101 pilot BIT-IDENTICALLY via a sudoku oracle (0 mismatches / 173,070). The seam the
  text + neural-LM oracles plug into.
- **#106 serve-integration:** `DECIDE(M>1)` live over a real package via a `TextOracle`, committing with
  `cert_kind="M-step-lookahead"` (condition 2); `serve_sw` refactored into `enumerate_candidates`
  bit-exactly. Functional, NOT a text-FIRES claim.

## The validation (#101/#102, #107, #108) — the decomposition, with a SIGN
| regime | slice | verdict | reading |
|---|---|---|---|
| sudoku, propagation-forced | #101/#102 | FIRES | validation-on-ground-truth (beam ≈ constraint propagation); ledger self-corrected (grok null-baseline → codex prereg-faithful +0.221) |
| sudoku, guess-requiring (must BRANCH) | #107 | **FIRES** | the energy RANKS the right guess: ~63–71% vs 0.33 baseline at M=8, prune biting; a confounded signature caught + corrected, a 1/1011 strict split recorded honestly; scope: shallow-`r`, M≥7 |
| real text (wikitext), soft-margin only | #108 | **REGRESSION** | soft-margin lookahead HURTS: Δagree −0.015 @M=2 → −0.007 @M=5, never positive, 2–10× latency; cross-vendor BIT-IDENTICAL; non-degeneracy verified |

**The result:** **the beam's value is the certified HARD term (legality), not soft-margin lookahead.**
Where the hard term is present and forces/prunes (sudoku), M-step lookahead commits answers per-token
can't reach. Where there is no hard term (real text), soft-margin lookahead sacrifices next-token accuracy
for trajectory confidence and is net-negative. The all-hard / all-soft endpoints bracket it decisively.

## The payoff design: the HARD-TERM-GATED beam
The regression is not a dead end — it's the deployment constraint. Blanket M>1 is net-negative because
real text is soft-dominated; the fix is to GATE: fire the M-step beam only where the hard margin
discriminates within M, else commit the M=1 corner. The gate signal is free (the margin +1/0 IS the
per-token hard/soft indicator). Then **gated-beam ≥ greedy everywhere, > greedy on hard tokens** — a free
option whose gain scales with a domain's hard-token density, i.e. exactly the bounded experts (family 4,
sgiandubh/claymore) the stack targets, and whose k·M cost is table lookups (no model/GPU/float). Full
design + the next experiment (a gated beam on a MIXED domain — code: hard brackets/indent, soft
identifiers) is in the what's-next escalation. **UNTESTED** — motivated by the two endpoints, not proven.

## Honest limits on record
- #107 FIRES is on SHALLOW-`r` cells (≤3; ~0.18% stall+unique yield from the uniform generator) and needs
  DEEP lookahead (M≥7); sudoku GROUND-TRUTH only — the beam ≈ constraint propagation there.
- #108 REGRESSION is on soft real text; a hard-term-gated beam on a mixture is the untested claim.
- The certified hard term today is sudoku legality only (code produced none, #99) — register density is
  the lever the whole payoff depends on.
- Tags: FIRES/REGRESSION are **empirical** over their stated domains; the gated-beam usefulness is **open**.

## What's next (escalated to the lead → the arc pauses)
Candidates (`WYLY_TRUNK_REASSESSMENT.md`): the **hard-term-gated beam** (the design the regression bought,
user-flagged) · atomic-sequence tier (condition 4) · ergo-claymore spoke wiring (family-4 payoff; composes
with a gated-beam spoke) · the deeper-search generator lever (#107 scope) · the (M,W) structural-rule
resweep + the neural-LM lookahead attribution control ([[beam-arc-followups]]) · back to trunk opens.
See [[sequence-energy-direction]], [[beam-arc-followups]], [[lane-balancing-rule]].
