# German R3-minimal — oracle rescore — LABELED-dependency gating confirmed (as a CEILING); aux_verb settled

Pre-reg: germandata `PREREG_GERMAN_EXPERT.md` §R3 (the lead's oracle-attachment reframe, steering 01:08/06:30).
Raced grok + codex. Measure-first UPPER BOUND: feed GOLD attachment/deprel, does German case reach 0.90 / does
aux_verb move? NOT serve-honest — an oracle ceiling. This is the terminal reconciliation of the R1→R2→R3 arc.

## Headline (fable-corrected): the lever is LABELED DEPENDENCY, not attachment
Gold **heads alone** (the governor-only oracle, no labels) scored **0.778 — BELOW the ~0.79 baseline**, both
lanes. Bare attachment does not help. What closes case is the **labeled deprel** (nsubj→Nom, obj→Acc,
adnominal-nmod→Gen, obl:arg→Dat, cop→Nom, det/amod agreement-inheritance, `case`-child→prep register). So the
justified subtask is a **labeled-dependency (deprel) predictor**, NOT an attachment/head predictor.

## CASE — the full deprel→case oracle CEILING
| lane | baseline | full deprel→case oracle (CEILING) | vs 0.90 |
|---|---|---|---|
| grok | 0.7779 | **0.9217** | clears |
| codex | 0.7893 | 0.8753 | near-miss |
Per-class (grok, full oracle): Nom 0.831, Acc 0.848, Dat 0.935, **Gen 0.950** (from 0.29 baseline — genitive is
adnominal `nmod`, pure labeled-structure). Cite PER-LANE deltas (baselines differ 0.778 vs 0.789).

## This is a CEILING, not a realized number (the load-bearing caveat)
The oracle is fed GOLD deprel. A REAL labeled-dependency parser at ~92% LAS would DISCOUNT the 0.92 ceiling by
its label accuracy → **plausibly BELOW 0.90 serve-honest**. So: the oracle ceiling clears the bar; this does NOT
establish that a served German-case expert reaches 0.90. A labeled-dependency subtask is justified to CHASE the
bar, but the realized number is ceiling × LAS and is marginal — do not tell the lead "case is closed."

## The cross-vendor straddle (grok 0.92 / codex 0.88) = rule-completeness, not an attachment wall
From the artifacts: codex's agreement-inheritance is pass-1 only (n=2886) vs grok's 2-hop fixpoint (n=3685,
~+0.04); grok added `obl:arg→Dat` (GSD annotates bare datives as `obl:arg`; literal `iobj→Dat` fires on 0 test
tokens; +0.005 @ 98.8% TRAIN precision) and restricted `cop→Nom` to nominals (it mis-fired on predicate
adjectives, which carry no case). grok's 0.9217 is the fair complete-cascade number (both fixes are correct GSD
linguistics, TRAIN-validated, disclosed as added after the partial-oracle miss). codex's 0.8753 is the same
cascade minus the fixpoint + obl:arg.
**Deliberately NOT re-run to cross 0.90:** re-running codex on the test set until it passes = adaptive test
reuse on a read-once protocol → would weaken the gating claim. The legitimate cross-vendor headline, if needed,
is a ONE-SHOT verbatim port of grok's cascade (no tuning loop, one test read) — deferred, not done.

## aux_verb — attachment/deprel is NOT the lever (settled cross-vendor)
Both lanes: gold attachment adds ~nothing (grok n_flips=0 — every oracle fire already matched the memorizer;
codex oracle == its clause-heuristic). aux_verb's ~0.87 ceiling is surface-form-subsumed, NOT labeled-dependency.
So a labeled-parser subtask helps the R1 case bar, NOT the R2 aux_verb bar.

## Named residuals (deferred — NOT done, to avoid test reuse)
- codex verbatim-port of grok's cascade (agreement fixpoint + obl:arg→Dat) — one test read, for a clean
  cross-vendor ceiling.
- `verb_government` register precision (grok residual bucket 0.103) — German verb→object-case is lexically
  idiosyncratic; a residual limiter, likely not a quick fix.

## Arc summary + tags
Partial oracle (heads + 2 registers) → "attachment dead, re-diagnose" → skeptic caught it undertested → full
labeled deprel→case oracle → "labeled-dependency clears the ceiling." Measure-first + the skeptic pass reversed
a wrong conclusion twice. Tag: empirical, **upper-bound (oracle)** — realized gain is ceiling × parser LAS.
