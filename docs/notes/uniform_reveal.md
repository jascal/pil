# Uniform-reveal sudoku — the legality register GENERALIZES across difficulty, with a gradient (slice #100)

**Status:** measured + registered, **cross-vendor confirmed** (2026-07-12). Steering item 3
(difficulty-axis finding; answers #90's untested early/mid caveat; the gate-(b) testbed).
Pre-registered in `PIL_UNIFORM_REVEAL_PREREG.md` BEFORE numbers. `experiments/campaign_uniform_reveal.py`
(+ tests; + `experiments/verify_uniform_reveal.py`, codex's independent cross-check). The
#89/#90/#91 campaigns and their frozen results are untouched.

## The cheap path worked — no regeneration
The late-reveal bias (index 60–80, all third-2) was a PARSING artifact, not a generator property:
the oracle's `find_subsequence` matched the FIRST `SOLUTION\n` header in the 256-window, which for
an early-cell target is the *previous* block's header → the cell was mis-parsed and dropped. Fix:
anchor to the LAST `SOLUTION\n` header before the target (the target's own block). This unlocked
the early/mid cells **already present** in `wyly_nexttoken_sudoku_L256.pt` — no new generator, no
teacher/mined-state rebuild (~4s run, pure parsing).
- classifiable cells 9078 → **34614**; distribution now spread (third 0/1/2 = 33.1/33.1/33.8%).
- **board-validity: 0 invalid** across all 34614; every reconstructed puzzle has **exactly 36 clues**
  (81−45 holes) — a wrong anchor would corrupt this, so 36-on-all proves correct anchoring.
- the OLD anchor reproduces the frozen #90 result (9078, all third-2) exactly.

## The difficulty gradient (cross-vendor exact)
Forced-recovery (`forced_value == gold`) per reveal-third — grok and codex (independent extraction) agree to ≤0.001:

| reveal-third | n | recovered_naked | recovered_union |
|---|---|---|---|
| 0 (early) | 11461 | 0.551 | **0.736** |
| 1 (mid) | 11468 | 0.694 | 0.869 |
| 2 (late) | 11685 | 0.857 | **0.950** |
| overall | 34614 | 0.701 | 0.852 |

## Interpretation: GENERALIZES WITH DEGRADATION (neither pre-registered binary fit)
The pre-reg named two readings (HOLDS if ≈ flat; endgame-only CONFIRMED if it collapses to the
~0.19 naked-only floor). **Neither fits** — reported transparently rather than forced:
- **Not endgame-only.** Even the hardest early-reveal cells recover a clear majority (union 0.74,
  naked 0.55) — far above 0.19. The 0.19 was itself a double artifact: the first-header clamp (only
  third-2 windows ever survived) and a without-clues external reconstruction. With the in-window
  clues + revealed prefix, the certified register is already strong early. **#90's caveat is
  answered POSITIVELY: the register is not confined to the near-solved endgame.**
- **Not flat.** A real, monotonic gradient (union +0.21, naked +0.31 early→late) — the signature of
  constraint propagation: more filled peers per cell ⇒ more cells pinned to one value.
- The honest object: the register's **certified correctness holds everywhere** (the feature ≡
  Datalog cert is reveal-index-agnostic); its **coverage** — the fraction of cells single-cell
  legality can force — is what degrades with difficulty.

## Gate-(b) payoff
The ~26% of early cells the single-cell register CANNOT force are exactly the ambiguity a
multi-token beam-energy lookahead could separate — the difficulty axis gate (b) needs (does
beam-energy separate a gold continuation better than the product of per-token confidences?).

## Tags
| Claim | Tag |
|---|---|
| per-third recovery gradient (0.74→0.95 union) | **empirical**, cross-vendor exact |
| the certified register generalizes across difficulty (not endgame-only) | **empirical** — #90 caveat answered positively |
| coverage (single-cell forcing rate) degrades with difficulty | **empirical** (constraint-propagation signature) |
| the late-reveal bias was a parse artifact; early cells were mis-parsed, not absent | **method correction** (36-clue + 0-invalid + frozen-#90 reproduction) |
| the "0.19 floor" difficulty reading | **superseded** — a clamp + without-clues artifact |

## Process (lane balancing)
grok built the parse-fix; codex confirmed the gradient from a from-scratch extraction path (numbers
≤0.001 AND independent framing agreement) — [[lane-balancing-rule]]. Both lanes, numbers and reading.
