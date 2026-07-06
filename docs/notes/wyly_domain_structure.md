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
