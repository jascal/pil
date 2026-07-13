# Hard-term-gated beam — Slice A: the gate + the #108 regression ELIMINATED (verified)

**Status:** built + verified (2026-07-13). The energy arc's payoff design (lead-named, WYLY_STEERING.md
17:07), Slice A = the gate mechanism + the free-safety proof. The wikitext pilot (#108) showed blanket
M>1 REGRESSES on soft text (Δagree −0.015); sudoku (#107) FIRES on hard. The gate makes the beam a **free
option**: fire M>1 only where the hard term engaged; else fall back to the bit-exact M=1 corner. Cross-repo
(rosetta engine + serve, + a pil proof).

## What was built
- **`rosetta/beam_engine.py`:** additive `max_committed_margin` on `beam_decode`'s return dict =
  `max(s.margin for s in winner.steps, default=0.0)` (0.0 on total-wipeout) — the clean per-token
  "did the hard term engage on the committed trajectory?" signal. (`prune_events` was a leaky proxy — it
  fires on text dead-ends too.)
- **`rosetta/serve_package.py::serve_energy` (M>1):** the gate — `if max_committed_margin <= 0.0: return
  serve_sw(...)` with `cert_kind="per-token"` (the bit-exact M=1 corner); else the beam commits as before
  (`cert_kind="M-step-lookahead"`). The M=1/beam=1 branch, `serve_sw`, `serve`, `decide` byte-unchanged.

## Result — the free-safety property, verified
On the existing text energy package (`wyly_expert_package_v5_energy_pilot_grok`, 57,325 rules), where
`TextOracle` pins every `margin=0.0`, the gate ALWAYS falls back to the corner:
- **Gated `DECIDE(M∈{2,3,4,5})` is BIT-EXACT to the M=1 corner** — `Δagree = +0.000000`, `Δcover =
  +0.000000` over the full held-out `te` split (n=11,477), 45,908 comparisons, 0 mismatches — vs the
  recorded UNGATED Δagree −0.015/−0.013/−0.009/−0.007. **The #108 regression is ELIMINATED.**
- `#101` sudoku reproduction still BIT-IDENTICAL (0/173,070; the field is additive). Corner parity still
  gap 0.0. rosetta 68 pass, pil ruff clean, +3 new gated-beam tests.
- The 3 prior serve tests that asserted un-gated M>1-on-text behavior are updated (strengthened with an
  explicit `result == corner` bit-exact assertion) — the correct by-design consequence: on text, M>1 IS
  the corner now.

## What it means — half the claim, proven
`gated-beam ≥ greedy EVERYWHERE` is now proven on the worst case (all-soft text → bit-exact corner, no
regression). The other half — `> greedy on HARD tokens` — needs the beam to actually FIRE, which requires
a served hard-term package. **NONE EXISTS** today (the #92 legality register is a Soufflé/tensor
certificate, not a servable manifest). So Slice B (the substantive follow-up) = emit a served
sudoku/legality expert (register→manifest + a manifest-driven legality oracle + a Python pyspoke, since
the C++ sgiandubh has no energy-beam branch), where the gate FIRES and demonstrates the `> greedy` half.

## Honest scope + tags
| Claim | Tag |
|---|---|
| the gate eliminates the #108 soft-margin regression on text (gated M>1 == corner bit-exact) | **proved** — 45,908/45,908 bit-exact over the held-out split |
| the corner + #101 are unaffected by the gate/field | **proved** — corner gap 0.0, #101 0/173,070 |
| gated-beam ≥ greedy everywhere | **proved on the worst case** (all-soft) — the free-safety half |
| gated-beam > greedy on hard tokens | **NOT shown** — needs a served hard-term expert (Slice B); untested |
| the gate signal (`max_committed_margin>0`) captures all hard-term wins | **partial** — captures forced-step wins; a served hard-term expert may also need a legality-PRUNE signal (vs text dead-ends) for pruning-driven wins — flagged for Slice B |

See [[sequence-energy-direction]], [[beam-arc-followups]] (iv), [[lane-balancing-rule]].
