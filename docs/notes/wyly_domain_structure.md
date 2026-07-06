# Does crystallization track domain structure? (pythia-70m, 8 corpora)

**Yes — and with a sharp threshold.** One protocol (80k held-out L=256 windows, teacher-decision
imitation, mined library, cover-aware judge, support-weighted arbitration; single seed), one rung
(pythia-70m, fits an 8 GB card), eight corpora spanning formal proof text to German parliamentary
prose. The structure axis is the gzip ratio of the *detokenized window stream* — uniform across
corpora, no reliance on raw-file availability.

```
    corpus   gzip   gold  copy%  student  core_sw  crystal
      code  0.251  0.692  91.9%    0.571    0.605   106.1%
  isabelle  0.271  0.502  93.9%    0.559    0.570   102.1%
        py  0.309  0.571  84.6%    0.473    0.504   106.6%
     legal  0.326  0.422  80.1%    0.469    0.480   102.4%
        de  0.350  0.417  63.0%    0.548    0.510    93.1%
      math  0.356  0.401  72.8%    0.389    0.355    91.2%
     wt103  0.400  0.316  73.1%    0.354    0.341    96.2%
  wikitext  0.402  0.311  71.9%    0.348    0.329    94.5%

Spearman vs gzip (n=8; more structured = lower gzip):
  crystallization rho = -0.69   core_sw rho = -0.93   gold rho = -0.98   copy% rho = -0.81
```

New corpora: `isabelle` (Isabelle2025-2 HOL theory sources — formal math), `py` (CPython stdlib
source), `math` (arXiv abstracts, 8 math.* categories — LaTeX-rich prose), `legal` (Blackstone +
Maine + CFR title 17 — mixed-register legal), `de` (EuroParl v7 German — multilingual). All are
Pile-adjacent domains for pythia. Teachers built in ~3 min each on the 8 GB card.

## Findings

1. **The crystallization threshold.** Every corpus with gzip < 0.33 (code, isabelle, py, legal)
   has certified core > soft student — crystallization 102–107%. Every corpus above (de, math,
   wt103, wikitext) sits at 91–96%. The "code regime" named in the ladders report is not a code
   fact; it is a **structure fact**, and it includes legal text and formal proofs.
2. **Teacher predictability ≈ text compressibility** (gold vs gzip ρ = −0.98). A 70m teacher's
   decision entropy is almost perfectly the corpus's byte-level structure — a strong sanity anchor
   for the axis, and a caution: the axis measures the corpus, and the teacher inherits it.
3. **The absolute certified core tracks structure even more tightly than the ratio**
   (ρ = −0.93): what the certified tier captures is close to a direct function of how much
   pattern the domain has.
4. **The rank inversions are informative.** German crystallizes *worse* (93%) than its gzip rank
   — plausibly rich morphology spreads count mass across forms the table tiers key separately
   (the soft student's geometry shares it); arXiv-abstract math is the worst crystallizer (91%) —
   high topical novelty per window, and abstracts are exactly the register where prose carries
   the load and notation is sparse.
5. **Copy incidence is extreme in formal text** (isabelle 94%, code 92%): the induction/relation
   tier is structurally load-bearing there, consistent with the mined-frame counts admitted per
   sleep (dozens on code-like corpora, singletons on prose).

## Follow-up: concept induction + class frames (build order 12+1)

Table-driven **concept induction** (`ConceptSpace`, refreshed each sleep: count rows with support
≥ 30 and cosine ≥ 0.85 union-find into concepts; cmap → representative) feeding two new judge
candidates under `WYLY_CONCEPTS=1`: a **pooled concept-counts tier** (counts index-added over
cmap; per-key Laplace confidence — the exact row wins where it has support, the pool wins on rare
members) and **mined cframes** (the mined-frame machinery with anchors passed through cmap —
class-anchored frames, discovered not designed). Full 8-corpus ablation vs the baselines above:

- **German, the named target, moved: core_sw 0.510 → 0.525, crystallization 93.1% → 95.1%** —
  closing about half its gap to the prose neighbors. The judge admitted `mined cframes` **on
  German only**; German's ConceptSpace merged 2,296 classes by the last sleep (the most of any
  corpus) — the morphology-fragmentation hypothesis confirmed as roughly half the story.
- `concept counts (pooled)` was admitted **on isabelle only** (+0.001) — pooling pays where
  reference-and-reuse dominates.
- Every other corpus flat within ±0.001, both families correctly declined by the judge: no
  regressions, no noise admissions.
- **The residual German gap (~95% vs ~96–97% expected) now points at word order** — verb-final
  Nebensätze put the predictive evidence beyond any fixed-offset frame; that is the pointer-rule
  item on the build order, not a concept problem.

## Follow-up: pointer rules (build order 2) — the copy headroom, harvested where concepts are crisp

The **pointer kind** generalizes induction: score every in-window source position by exact-suffix
match length ℓ *and* concept-suffix match length ℓc (through the ConceptSpace cmap), predict the
successor of the lexicographic argmax, with a per-(ℓ,ℓc)-cell confidence measured on val each
sleep — C10's calibration premise applied cell-wise. `WYLY_POINTER=1`; induction L is the
ℓc-blind fixed-ℓ special case.

8-corpus ablation vs the concepts baselines:

- **New arc bests on the structured side**: code 0.605 → **0.610**, isabelle 0.571 → **0.573**,
  math 0.355 → **0.365** (+0.010, the largest gain). The judge admitted the pointer on
  code/isabelle/py **ranked first** (the single most valuable rule) and on math third.
- **The off-diagonal (ℓc > ℓ) cells are real where concepts are crisp**: code ℓ=4,ℓc=6 fires at
  **0.89** val accuracy, ℓ=2,ℓc=6 at 0.75, even ℓ=0,ℓc=6 at 0.60; isabelle ℓ=2,ℓc=6 at **0.84**;
  math ℓ=6 diagonal at 0.83. Class-extended copy — match by type/identifier class where the
  exact tokens differ — is a genuinely new, reliable channel on formal text.
- **German stayed flat (0.525), pointer declined** — the discriminator's second half resolves:
  class-matched sources exist but predict the *wrong surface form* (a class-pointer proposes the
  source's successor verbatim; German needs the successor's *class* re-inflected in the target
  context). The residual is a **copy-with-transform** problem (build order 5), not a pointer
  problem. Prose (wikitext/wt103/legal) likewise flat: its off-diagonal cells are unreliable
  (0.04–0.28) — prose concepts are not crisp enough to extend matches through.

Schema note: a `pointer` package kind would ship `lmax`, the cell-confidence table, and reference
the manifest's concepts map — the relation-kind playbook applies once a served use-case wants it.

## Follow-up: transform-composed pointers (build order 5 ∘ 2) — the mechanism validated, the German thread honestly closed

`WYLY_TPOINTER=1`: the pointer finds the source by class-match; the **counts tier re-inflects** —
predict the member of the source-successor's concept that the target's local bigram row supports
most (`argmax_{m∈concept(succ)} counts[last, m]`). A composition of two certified structures with
its own (ℓ,ℓc)-cell confidences. 8-corpus ablation vs the pointer baselines:

- **The mechanism works, uniformly**: the class→form decode roughly doubles pure-class cell
  reliability on *every* domain (de ℓ=0,ℓc=2: 0.11→0.24; code 0.20→0.35 and ℓ=0,ℓc=6 0.60→0.68;
  isabelle 0.21→0.33; math ℓ=0,ℓc=4 0.23→0.45). Re-inflection through local count evidence is
  the right transform.
- **Arbitration keeps each rule in its cells**: on long exact matches the transform *hurts*
  (code ℓ=4,ℓc=6: 0.89→0.70 — verbatim beats re-decoding when the match is strong), and the sw
  cover lets the plain pointer keep those cells while the tpointer offers the low-ℓ ones. The
  judge admitted the tpointer **on code** (where its cells add cover) and correctly declined it
  elsewhere.
- **No core moved** (all 8 within ±0.001): the class-only channel tops out at 0.24–0.45 val
  accuracy — below what wins arbitration against the table tiers on almost every window at this
  corpus scale.

**The German thread, three experiments in**: morphology fragmentation → *fixed* by concept
induction (+0.015, half the gap); word-order copy → the pointer machinery reaches it and the
transform decodes the right form-class, but bigram-row concepts at 6 MB are too coarse to make
distant class-matches reliable. Per the tag discipline: the *recipe* plateaus here —
achievability stays **open**, with two named paths: sharper concept induction (multi-tier
evidence: frame/skip rows, not just bigrams) and more corpus (the support floor on class cells is
the binding constraint).

## Confounds (named)

Single seed per cell; corpus register differs from Pile representation per domain (CFR is
XML-stripped; arXiv *abstracts* are not arXiv *papers*; EuroParl German is one genre of German);
each structured corpus is a single project/source (isabelle = one distribution, py = one stdlib),
so within-corpus self-similarity and possible teacher memorization inflate both axes together —
the gold-vs-gzip anchor suggests the axis is still meaningful, but cross-source replication
(several projects per domain) is the obvious robustness step; gzip is a byte-level proxy and
under-credits morphological structure (the de inversion).

*(Produced 2026-07-05, the domain-structure goal; script `experiments/wyly_domain_structure.py`;
logs in the review artifacts dir. Extends the ladders report and the paper's §7/§8 domain claim:
"more crystallizable the more structured the domain" is now an 8-point measured curve, not a
2-point contrast.)*

## The refreshed matrix: the threshold dissolves (complete stack, 2026-07-06)

Re-run of all 8 corpora under the full stack (concepts + pointers + transform + detection +
derived roles incl. composition + learned member sets + 3-fold admission):

```
    corpus   gzip   gold  copy%  student  core_sw  crystal    (orig core -> now)
      code  0.251  0.692  91.9%    0.584    0.611   104.7%    0.605 -> 0.611
  isabelle  0.271  0.502  93.9%    0.578    0.588   101.7%    0.570 -> 0.588
        py  0.309  0.571  84.6%    0.463    0.501   108.1%    0.504 -> 0.501
     legal  0.326  0.422  80.1%    0.478    0.487   102.0%    0.479 -> 0.487
        de  0.350  0.417  63.0%    0.554    0.539    97.3%    0.510 -> 0.539  (+0.029)
      math  0.356  0.401  72.8%    0.395    0.381    96.4%    0.355 -> 0.381  (+0.026)
     wt103  0.400  0.316  73.1%    0.357    0.350    98.0%    0.341 -> 0.350
  wikitext  0.402  0.311  71.9%    0.343    0.342    99.5%    0.329 -> 0.342

  Spearman vs gzip: crystallization -0.67, core -0.93, gold -0.98 (stable axis)
```

**Five new arc bests** (wikitext, wt103, math, legal, code). The original study's sharp
crystallization threshold at gzip ≈ 0.33 (>100% below, 91–96% above) has become a gentle slope:
**every corpus now crystallizes at 96–108%**. The prose/formal crystallization gap was largely a
RULE-VOCABULARY artifact, not a domain constant — closed by rules over derived roles (mate,
depth, clause, referent, composed roles) and learned member sets, not by more memory. The
biggest movers are exactly the two worst crystallizers of the original study: German +0.029
(morphology → concepts; Nebensätze → clause/clause-succ roles) and math +0.026 (referent +
learned-set roles). What remains open: the absolute ceiling still tracks structure (core ρ −0.93
unchanged) — roles close the *ratio* gap, and the teacher's own predictability bounds the rest.

## The abstraction wing: abstraction breaks the band, morphology does not (12 corpora)

Four corpora added at the user's suggestion — **phil_de** (Kant/Nietzsche, German Gutenberg),
**psy_en** (Freud/James/Jung), **soc_en** (Veblen/Spencer/Sumner), **fi** (EuroParl Finnish, the
agglutinative extreme) — full stack, same protocol:

```
    corpus   gzip   gold  student  core_sw  crystal
        fi  0.345  0.389    0.580    0.558    96.2%   <- morphology extreme: HOLDS the band
   phil_de  0.366  0.323    0.438    0.416    95.2%   <- abstraction x morphology (OOD teacher)
    psy_en  0.390  0.333    0.366    0.338    92.3%   <- abstraction: BELOW the band
    soc_en  0.393  0.347    0.383    0.348    90.9%   <- abstraction: the new floor
  (12-corpus Spearman: crystal -0.61, core -0.91, gold -0.94)
```

1. **Finnish vindicates the roles machinery**: despite morphology far beyond German's, it
   crystallizes at 96.2% — concepts (1,655 merges) + depth/cmember roles carry it. The
   morphology story is closed: agglutination is not the certified tier's enemy.
2. **Abstraction is.** psy_en and soc_en under-crystallize their compressibility rank by 5–7
   points — the same *rank-inversion* signature German once showed, now in the conceptual
   direction. Strikingly, the concept machinery worked *hardest* exactly there (psy 7,884 /
   soc 9,888 classes merged — abstraction vocabularies cluster massively) and the judge admitted
   learned-set and referent roles on all four — **the diffuse tail is not a vocabulary problem
   the current role kinds can reach**. Abstraction-dense argument structure (claims, stances,
   discourse relations) is the next genuinely missing rule register.
3. **Confounds, named**: soc_en is the smallest corpus in the study (2.1 MB; psy 3.8 MB) and de20
   showed support floors matter (+0.7% from 3.5× corpus) — but phil_de at similar size scores
   95.2%, so size alone does not explain the psy/soc dip. German-language Gutenberg is likely
   outside the Pile (phil_de gold 0.323 — the lowest of any corpus — treat its row as
   teacher-OOD); cover on the abstraction wing runs 94–97% vs 99%+ elsewhere.

## The discourse register: admitted, honest about its size — and the 6.9b rung

**Discourse roles** (the register the abstraction wing identified as missing): `since-member`
(position-in-sentence/move), `member-parity` (quotation/attribution scope), plus English
connective, attribution-verb and subordinator sets over the existing kinds — all Soufflé-PROVED
(seven extractor kinds at 192/192). The judges take them, domain-appropriately:

- **attrib gate** admitted on psychology (Freud/James *attribute* constantly);
  **quoteparity FIRST** on sociology *and* German philosophy; **clause (now English) +
  sincedot** on wikitext.
- Movement is real but modest: psy_en 92.3 → **93.9%**, phil_de 95.2 → **96.3%**, soc_en flat,
  wikitext steady at 99.1%. Verdict, tagged honestly: the register is **admitted but not yet
  decisive** — discourse structure is the right direction (judges keep choosing it) but the
  abstraction tail at 70m/small-corpora needs either deeper discourse roles (rhetorical-relation
  spans, claim/evidence pairing) or simply more corpus (soc_en remains 2.1 MB).

**The 6.9b rung — the scaling law extends to 500×** (teacher run entirely on the 8 GB card:
int8, batch 1, 1.9 h; gold top-1 0.497, quantization caveat named):

- fig-comparable base config: fixed core **0.224** vs the series 0.285 (14m) → 0.230 (2.8b) —
  **the plateau holds**;
- full modern stack: core_sw **0.268 @ 99.0%** vs student 0.260 — **crystallization 103% at the
  largest teacher yet**. The roles library doesn't just survive scale; the certified core
  exceeds its own student at 6.9B.

## Deep-discourse roles: the SPAN register works; the pairing register doesn't (yet)

Rhetorical-span and claim/evidence roles, all composed from certified primitives (nine role
programs now Soufflé-PROVED, incl. the claimant `succ=-1` and the claim-echo
`of_shift=-1 → prev-occ → succ=1` chains) plus one new kind (`dgate2`: gates on two features
jointly — the span-pair):

- **The span roles win.** `prevsent-head` ("what did the *previous* sentence start with") was
  admitted on wikitext at **+0.0097 — the largest single dgate marginal of the campaign** —
  taking wikitext to a **new arc best 0.346**; and it finally moved sociology
  (0.345 → **0.351**, crystallization 91.0 → 91.9%, the first real soc gain in three attempts,
  alongside connectives). `senthead` was admitted on psychology (+0.0037).
- **The pairing roles don't (yet).** `sentpair` (dgate2), `attrib-subj` (the claimant) and
  `claim-echo` were built, certified, offered — and declined everywhere. Tagged honestly: the
  claim/evidence *pairing* hypothesis is unsupported at 70m on these corpora; sentence-head
  SPANS carry the recoverable discourse signal. phil_de took none of the deep gates (−0.008,
  run variance).

The abstraction ledger after three discourse goals: psy 92.3→93.9%, soc 90.9→**91.9%**, wiki at
**99.7%** with its best core yet. The tail shrinks by real, admitted, certified rules — a point
a sleep at a time.

## Closing the abstraction tail: corpus does it — and unlocks the pairing register

Two attacks, cleanly attributed (psy20/soc20 = 16 MB multi-source Gutenberg topic corpora, 29
books each — which also removes the single-source confound; psy_en/soc_en re-run with the new
registers isolates the register effect):

- **The relation registers with multiple surface roles** (`dstate` — bucketed
  position-in-sentence × quotation parity, ~10 states — plus `dgate2` combos conn×head,
  dstate×head, conn×prevhead) were built, certified (TEN role programs now, all Soufflé-PROVED),
  offered — and **declined everywhere**. On the small corpora the register-only runs moved
  −0.005/flat. Tagged: multi-role relation gating is unsupported at 70m even with low-cardinality
  sides.
- **Corpus size closes the tail.** soc20: crystallization **90.9% → 97.5%** (core 0.350 vs
  student 0.359) — sociology rejoins the prose band (wt103 98.0%, wiki 99.7%). psy20: **→
  94.1%** (from 92.3–93.9). Same-difficulty teachers (gold 0.331/0.352 ≈ the small-corpus
  golds), 29 sources each — the effect is support, not source memorization.
- **And more corpus unlocked the claim/evidence machinery**: `attrib-subj` — the claimant role,
  declined on every small corpus — was **admitted on soc20** (84 keys), alongside prevsent-head.
  The pairing register wasn't wrong; it was support-starved.

**The abstraction ledger, final for this campaign**: psy 92.3 → **94.1%**, soc 90.9 → **97.5%**.
The 'abstraction breaks the band' finding survives in weakened form: at matched (16 MB) corpus
scale, psychology remains the hardest domain in the study — the residual is real but small, and
every point of it was closed by admitted, certified structure.

## The causal/temporal wing: genre template beats byte structure (15 corpora)

Three causal-reasoning domains (user-proposed): **med** (Gutenberg medicine, 16 MB), **hist**
(analytical history, 16 MB), **postmortem** (disaster investigations, 5.5 MB — the Rogers
Commission Challenger report, Johnstown flood, 1906 San Francisco earthquake). Plus a temporal-
marker register (tempo/tempo-succ gates — declined everywhere, 0/3 folds: point-markers of time
carry no recoverable signal).

```
    corpus   gzip   gold  student  core_sw  crystal   admitted highlights
postmortem  0.390  0.429    0.311    0.334   107.4%   quoteparity, prevsent-head
       med  0.409  0.361    0.359    0.346    96.4%   (marker gates declined)
      hist  0.405  0.333    0.433    0.394    91.0%   DSTATE (first ever!), clause-succ, depth
```

1. **The study's biggest positive rank inversion**: postmortem at prose-level compressibility
   crystallizes at **107.4%** — the core *exceeds* the student, code-style, on text that gzip
   calls unstructured. The investigation genre's rhetorical template (chronology → findings →
   testimony quotation → recommendations) is invisible to bytes and perfectly visible to the
   role registers. The Challenger report crystallizes like a program.
2. **Analytical history is the new hardest domain (91.0%)** — strong student (0.433), lagging
   core: free-running causal narrative with no template. Fittingly, it drew the **first-ever
   dstate admission** (the rhetorical-state register, declined on all twelve prior corpora) plus
   clause-succ — the discourse registers activate exactly where discourse is the only structure,
   they just can't yet carry enough of it.
3. **The gzip axis breaks down at the analytical end** (crystallization ρ −0.53 at n=15, from
   −0.69 at n=8): postmortem and hist differ by 0.015 gzip and 16 crystallization points. At the
   prose end, what predicts certifiability is **genre scaffolding**, not compressibility — a
   better axis would measure rhetorical-template density directly (the admitted-register profile
   is itself that measurement).

Confounds: postmortem is 5.5 MB and 4 sources (small-corpus + genre-homogeneity inflate together
— though the *direction* matches soc20's support finding); Gutenberg medicine/history are
pre-1930 registers; tempo sets were English+German only.

## The constraint/planning wing: the residual IS the constraint semantics (19 corpora)

Four generated, seed-reproducible constraint domains (`wyly_gen_planning.py`): **chess** (13,869
legal games — every continuation move-legal by construction), **sudoku** (23,461 CSP
puzzle→solution grids; a 30-token vocabulary that required a tiny-vocab pad in the PCA
grounding), **sched** (2,567 weeks of no-conflict hospital rosters), **robot** (36,903 gridworld
BFS-optimal path traces).

```
   corpus   gzip   gold  copy%  student  core_sw  crystal   notable admissions
    sched  0.076  0.839  97.7%    0.917    0.877    95.6%   pointer first
    robot  0.176  0.396  99.3%    0.617    0.587    95.1%   pointer; THREE-rule library
   sudoku  0.269  0.307  99.9%    0.544    0.520    95.6%   pointer, sincedot = grid position
    chess  0.377  0.429  85.9%    0.623    0.597    95.7%   induction, CAP-ECHO ON PIECES
```

1. **A tight 95.1–95.7% band across four unrelated constraint notations** — the wing's finding.
   The template/notation layer crystallizes completely (copy% runs 86–99.9%; pointers dominate
   every library); the missing ~4.5% is uniform and is exactly the **constraint semantics** —
   legality, no-double-booking, path optimality — computations no current register performs.
   The first rule register that *computes* a constraint (a legality checker as a derived
   feature) would claim it.
2. **The gzip axis is now dead for crystallization** (ρ −0.18 at n=19; it still predicts the
   absolute core, −0.78): sched is the most compressible corpus in the study (0.076) yet sits at
   95.6%, while postmortem at 0.390 hits 107.4%. Across 19 corpora the domains cluster by
   structure **type**: template genres (code, proofs, investigations) >100%; constraint
   notations ≈95.5%; morphology-rich prose 96–99.7%; analytical/abstraction prose 91–95%.
3. **The registers keep explaining themselves**: cap-echo fires on chess *piece letters* (the
   entity echo reads N/B/R/Q/K); sincedot becomes *grid position* on sudoku; robot's entire core
   is a three-rule library (pointer + mined frames + one concept set) at 95% — planning traces
   are almost pure copy structure.
4. **OOD caveats named**: sched/robot formats are synthetic (outside the Pile) — yet sched's
   gold is 0.839, the highest in the study: templating dominates familiarity. Sudoku's gold
   0.307 shows the constraint content itself is opaque to the teacher; the student *beats* the
   teacher's own next-token predictability by copying grid structure.