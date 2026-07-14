# Attachment lever-probe — cheap levers don't close case; the ~79% lever is deprel-LABELING, not head-attachment

Front-door measure-first probe gating the lead-adopted axis-B pivot (build a certified relational attach-primitive
only on the probe's evidence). Raced grok + codex; a half-oracle factorization settles the target. Serve-honest
features (R1-predicted POS, never gold labels), gold-trained, GSD test read once, seed-fixed bilinear. NOT the
certified build.

## The gate: FAILS as pre-registered
No cheap lever closes serve-honest German case. Ladders (GSD test):

| rung | grok UAS/LAS/case | codex UAS/LAS/case |
|---|---|---|
| L0 count-table greedy | 0.378 / 0.334 / 0.757 | 0.484 / 0.427 / 0.741 |
| L1 + tree decode | 0.387 / 0.339 / 0.767 | 0.454 / 0.399 / 0.727 |
| L2 + bilinear soft-scorer | 0.539 / 0.414 / 0.755 | 0.464 / 0.380 / 0.725 |
| L3 both | 0.538 / 0.417 / 0.762 | 0.460 / 0.377 / 0.726 |

No rung reaches the ~0.87 case gate (all ~0.72–0.77). Cheap bilinears cap head-prediction at **~0.46–0.56 UAS**
(they lift over a weak count-table baseline [grok] but HURT over a strong one [codex] — baseline- and
implementation-confounded, so the robust statement is the UAS cap, not "the bilinear lifts heads"). PLATEAU.

## The half-oracle factorization (the decisive test): DEPREL-LABELS are the ~79% lever
Feeding oracle vs predicted heads/deprels through the case cascade (codex, best rung L2; ceiling = codex's own
cascade **0.875**, NOT grok's fuller-cascade 0.92):

| condition | serve-honest case |
|---|---|
| predicted heads + predicted deprels | 0.724 |
| oracle heads + predicted deprels | 0.744 (**+0.019**) |
| predicted heads + oracle deprels | 0.798 (**+0.073**) |
| oracle heads + oracle deprels (ceiling) | 0.875 |

The deprel-oracle swap buys ~4× the head-oracle swap → **~79% of the case gap is deprel-LABELING, ~21% is
head-attachment.** A real but skewed split (both contribute; labels dominate); neither alone reaches the ceiling.
The cheap bilinear even DEGRADED case-bearing-deprel accuracy (L2 0.497 vs L0 0.649 on the relations the cascade
routes on) — it chased the smaller lever and hurt the bigger one.

## Consequence: REDIRECT the primitive (don't kill it)
The attach-primitive as specced targeted HEADS = the smaller lever. The relational primitive German case needs is
a relation-LABELER — a biaffine scorer with the **deprel label** as the priority (heads secondary). Same axis-B
shape, retargeted from "which word governs" to "what relation type." Even oracle labels alone reach only 0.798
(< 0.90) with predicted heads, and cascade completeness caps the ceiling (grok's fuller cascade 0.92 clears 0.90,
codex's 0.875 does not) — so the full close needs labels + heads + a complete cascade. The **R4 LLM-hybrid remains
the honest fallback** if the retargeted primitive doesn't reach the bar.

## Data-side convergence (user-directed): golden-set curation attacks the same lever
The dominant lever being LABELS means the user's golden-set strategy — review Haiku's parses, keep the correct
ones, hand-build golden examples for the wrong ones (the exceedance play; Haiku ~0.39 UAS = many errors to fix) —
directly improves the deprel LABELS. Data-side (golden labels) and model-side (retargeted label-primitive)
converge on the same bottleneck. Sequencing (user): the UD golden set first, then the satzklar **component** parse
(both user-facing; component is higher UI value, user-curated, with germanapp-repo tests — done after UD). A full
golden set is the prerequisite for the next-stage build. See `[[german-golden-set-and-component-strategy]]`.

## Honest scope + lane note
Serve-honest features; gold used only as oracle-swap inputs for the factorization (never training); test read
once. The cheap bilinear was small (rank ≤16, few epochs) — measure-first, NOT the certified primitive; the
ceiling of a full-capacity label-primitive is untested (this probe fenced off the CHEAP head-scorer and located
the label lever). Lane substitution: grok's Build account hit a payment error mid-run; its ladder numbers are
recorded above (independently verified by the wrapping agent), but its pilot file (4 lint errors + a hand-fixture
typo, logic verified correct) was NOT committed — codex's pilot (clean; carries the ladder + the factorization +
case-bearing-deprel metrics) is the shipped artifact. Tag: empirical.
