# German R1 — per-token transduction + register hard-layer — in-between (cross-vendor); case is R3-gated

Pre-reg: germandata `PREREG_GERMAN_EXPERT.md` §R1 (bars verbatim). Raced grok + codex — each a full independent
SOFT=0 build (count-table + register arbitration, no wake/SGD) — measured in two cuts: a conservative first cut
(registers fire only on form-unambiguous hits) and a fairer "narrow-and-arbitrate" enrichment (register = hard
prune to its candidate set, soft tier picks the survivor; NP-span government; a gnn register tier). Trained on
GSD **gold only**, test read once, no teacher dumps. This is the German instantiation of the agreement register
that #113 / Probe A licensed on templated English.

## Verdict: in-between (empirical), cross-vendor robust
Both lanes, both cuts:
- **pos ~0.886** (grok 0.888, codex 0.885) — clears the 0.85 halt floor, misses the 0.95 fire bar.
- **morph_case ~0.72–0.80** — well under the 0.90 fire bar.
Does not fire, does not halt. morph_gnn ~0.80 (not gated). Null floor (case gold "-") = 0.529.

## The register marginal is small and PARSE-FRAGILE (the load-bearing finding)
| cut | grok case marginal | codex case marginal |
|---|---|---|
| conservative (unambiguous-only) | +0.003 | +0.020 |
| enriched (narrow-and-arbitrate + NP-span government) | **−0.053** | **+0.013** |

The sign is NOT stable across lanes/cuts — it flips on a government **span-stop** detail. grok's span (stops at
punct / next-prep / conjunction, NOT at a verb) over-reaches under German **V2 word order** (a fronted PP —
"In den Garten geht er." — runs the span past the NP and hard-prunes the verb/subject to the preposition's case;
55% precision) → −0.053; codex's span stops at the first non-NP token (the verb) → marginally positive. →
**we cannot establish a clean positive register value on GSD case under parse-free heuristics.**

## The per-class structure IS robust (both lanes) — the register value is localized to government
Enriched, case ON: government-determined cases IMPROVE (Dat → 0.86–0.90, Acc up), form-ambiguous + null DROP
(Nom, Gen, "-") from hard-pruning an incomplete paradigm. The registers genuinely help exactly where case is
government-determined, and hurt where the paradigm is ambiguous/incomplete — net a wash-to-negative.

## Conclusion: case is R3-gated
The genuine case lever is government done RIGHT = a real attachment parse (R3), not a heuristic span. Form-
decidable declension registers cannot net-lift case (German syncretism: "die"=Nom/Acc, "der"=four cases), and
parse-free government is too fragile (V2). Strong cross-vendor evidence that **case accuracy is R3-gated** —
matching the prereg calling R3 "the honest hard rung where the semantic core lives." R2 (det_pron / aux_verb,
binary context-gated) is next since R3 has no data yet.

## Honest scope + owned gaps
- Government is a parse-free heuristic (lower bound). The two enrichment gaps that bias the marginal DOWNWARD
  are the architect's spec gaps, not lane bugs: (i) the span-stop rule is V2-blind (over-reaches on fronted PPs);
  (ii) an ungated bigram score out-votes the dev-tuned context tier inside a correct candidate set. A verb-aware
  span-stop is a named follow-up to pin the marginal sign — it does not change the verdict or the R3-gated
  conclusion.
- morph_gnn's register tier existed only in the enriched cut (the first cut had none — an architect spec gap);
  its marginal (~−0.003 to −0.004) is small-negative, same over-pruning pattern.
- The majority-per-form memorizer is strong (pos ~0.885, case ~0.75) — most accuracy is pure form-memorization;
  the SOFT=0 context/register tiers add little net. Tag: empirical.
