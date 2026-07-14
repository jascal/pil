# German R2 — det_pron + aux_verb disambiguation — in-between (does not fire); surface plateaus, attachment-sufficiency open

Pre-reg: germandata `PREREG_GERMAN_EXPERT.md` §R2 (fires iff BOTH det_pron ≥ 0.97 AND aux_verb ≥ 0.97 on GSD
test AND admitted rules inspectable + recover the app catalog). Raced grok + codex — SOFT=0 mined-rule students,
serve-honest **surface** features only (no gold neighbor labels), gold-only training, test read once.

## Verdict: in-between / does not fire (cross-vendor)
| task | grok | codex | bar |
|---|---|---|---|
| det_pron | 0.927 | 0.969 | ≥0.97 |
| aux_verb | 0.870 | 0.870 | ≥0.97 |

Neither task clears 0.97 in both lanes. (The lead's "R2 expected to fire" is not borne out — R2 is in-between.)

## det_pron — near the bar, but not a clean fire on EITHER conjunct
codex reaches 0.9686 (0.0014 short) via a **743-leaf** dev-admitted context tree; grok's inspectable-feature
version (next-cap) gets 0.927. So det_pron misses the accuracy bar, AND its near-miss rides a 743-leaf tree
that is **in tension with the "admitted rules inspectable + recover the catalog" conjunct** (a 743-leaf tree is
not an inspectable catalog rule). We do NOT chase the 0.0014 (threshold-gaming a pre-registered bar). The
genuine inspectable win: "followed-by-NOUN ⇒ DET" IS recovered/improved cleanly (both lanes, via next-capitalized
≈ German nouns are capitalized).

## aux_verb — surface features plateau; both lanes fell back to the memorizer
Both lanes score **0.8703 = the majority-per-form memorizer EXACTLY** — because both **failed to find an
admitted rule** (cross-vendor failure-to-find, NOT a replicated ceiling). The catalog rule "paired-with-
participle ⇒ AUX" is **marginal, not recovered**: codex's data check = dev 0.890 vs 0.877 naive, ~3% adjacency
coverage — the participle is often NOT adjacent (German word order puts it clause-final: "Er *hat* das Buch
*gelesen*").

**Honest scope (tag discipline — no necessity-from-plateau):** surface features AS TRIED (adjacency; whole-
sentence has-participle) plateau at ~0.87–0.89 on aux_verb. Whether aux_verb is closeable is **open** — two
untested levers: (a) a clause-bounded, direction-aware **parse-free** participle search (comma-bounded, both
aux…participle and participle…aux orders, ge-/-t/-en shape); (b) the **R3 gold-attachment rescore** (does
knowing the governed participle fire aux_verb?). We do NOT claim "aux_verb requires attachment" — the recipe
(surface features tried) plateaus; achievability is open, tested by (a)/(b). Note: the 0.890 dev ceiling on the
oracle-adjacent flag suggests the participle family itself may cap below 0.97 (modal+infinitive, haben/werden
main-verb senses where no participle exists).

## Through-line (HYPOTHESIS, not a finding — n=2 + a near-miss)
The German ceiling MAY recur at syntax: case (R1) needed the governor; aux_verb (R2) surface-plateaus where the
governed participle is non-adjacent; only det_pron is local enough to nearly fire on surface features. The
**R3-minimal oracle-governor rescore** (queued) is the direct test — extended to rescore aux_verb given gold
attachment too.

## Notes
- Gold-only training; test read once; serve-honest features. Tag: empirical.
- codex's foreground run hit the 10-min cap and resumed at lower reasoning effort; its numbers reproduced
  byte-for-byte on independent re-run.
