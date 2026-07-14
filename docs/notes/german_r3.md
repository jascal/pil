# German R3 proper — labeled-dependency predictor — case NOT closable by the null parser (0.74 vs 0.92 ceiling); pivot to a relational attach-primitive

Pre-reg: germandata `PREREG_GERMAN_EXPERT.md` §R3. Raced grok + codex — SOFT=0 greedy per-token head+deprel
predictor, serve-honest features (R1-predicted POS, no gold labels), gold-trained, GSD test read once.

## The load-bearing finding (robust cross-vendor): case is NOT closable serve-honest by this parser
Deliverable B — morph_case rescored under the student's **predicted** head+deprel (the serve-honest number the
#116 gold-attachment oracle could not give):
- grok **0.757**, codex **0.741** — vs the #116 GOLD-ATTACHMENT CEILING **0.92** and the R1 parse-free baseline
  **~0.79**.
Serve-honest case falls far short of the 0.92 ceiling (parse quality is the gap), and is even **below** the 0.79
baseline (codex delta −0.0485, n=7780, ~10 SE — real, not noise): a bad parse's wrong deprels ACTIVELY corrupt
the case cascade. Cross-vendor robust AND implementation-insensitive — case agrees within 1.6 points (0.741 vs
0.757) despite a 9-point student-LAS gap. **Scope: THIS SOFT=0 greedy parser family cannot close case — NOT "no
parser can"** (that is the pivot).

## Deliverable A — the student parser is weak
UAS **0.378–0.484**, LAS **0.334–0.427** (range; codex's build stronger). A SOFT=0 greedy count-table over
relative offsets is a poor head model — no learned relational scorer, no tree constraint (both stated scope
limits). Majority-offset baseline: UAS ~0.32.

## The Haiku comparison — plausibly real emission-mediocrity, but NOT apples-to-apples (no "fires" claim)
Haiku's one-shot whole-tree UD emission (`teacher/ud_served.jsonl`) scores UAS **~0.39** / LAS **~0.34** on the
clean-tokenization overlap (grok 234 sentences, codex 206) — **cross-vendor consistent**, so NOT an alignment
artifact (an earlier intermediate grok run showed a lower number; the shipped/final run agrees with codex). This
is plausibly REAL: one-shot nested-JSON UD-tree emission is a hard format (~180/384 sentences have token-count
mismatches → emission is unreliable). BUT the prereg's "student LAS within 5 of Haiku" FIRES criterion is **not
meaningful here and is NOT claimed**: (a) the overlap (~230/977) is selection-biased toward cleanly-tokenized
sentences; (b) it compares FULL-test student LAS against OVERLAP-ONLY Haiku LAS (different populations); (c)
whole-tree emission ≠ the targeted attachment query an LLM hybrid would actually use — so "student ≥ Haiku-whole-
tree" says nothing for or against the hybrid. **Residual:** a proper CoNLL token-alignment + same-population
comparison (and a word-identity / deprel-only check to settle real-vs-artifact definitively) if the Haiku
baseline is ever needed load-bearing.

## Consequence: the pivot (user-directed) — build a relational attach-primitive, not an LLM hybrid
The serve-honest number decides: case is not closable by the SOFT=0 greedy parser. Rather than the R4 LLM-hybrid,
the direction (user-directed) keeps attachment in the certified tier — a **certified bilinear attach-primitive +
tree decode** (the axis-B relational-primitive growth; a focused, domain-general parsing expert; attachment is
the same bilinear "does i govern j" shape as the already-certified induction head). A staged **lever-probe**
measures whether cheap levers (a tree decode + a bilinear soft attach-scorer) lift LAS enough to make case
closable serve-honest, before committing the full certified build. **Coordination:** this diverges from the
lead's R4-hybrid default ("< 0.90 → hybrid; no re-litigating") — it does NOT re-litigate the R3 number (it
stands); it chooses to BUILD the parser rather than rent one. Flagged to the lead.

## Honest scope + tags
Serve-honest (R1-predicted POS; never gold head/deprel/POS); gold-trained; test read once; greedy per-token
(no tree/MST constraint) — a stated scope limit. `verb_government` register precision (~0.59–0.64) is a residual
on the cascade ceiling. Tag: empirical.
