# Rule-library × dataset ladders (the library/dataset-extension goal)

**What this is.** The certifiability instrument (self-compiling student → sleep judge → certified
package cover; `experiments/wyly_lm_v5.py`, matrix in `experiments/wyly_ladder2.py`) run across
**two rule libraries** and **three datasets**, with teachers from the pythia ladder. All numbers are
teacher-decision agreement on held-out L=256 windows, 80k windows per dataset, one protocol.

- **Libraries.** `base` = induction L=1,2,3 (the §8.5 library). `ext` adds induction L=4,5; k-gram
  suffix tables k=2,3 (fit once on the train region, support≥2/det≥0.5 pre-gate; the package's
  native `ngram` kind at ctx length k); skip-bigrams at offsets 2,3 (the package `gate` kind with an
  empty frame); and a repetition rule (eq+copy — **no package kind exists**; see design directions).
- **Datasets.** `wikitext` (wikitext-2 slice; the fixed dataset of the whole arc), `code`
  (llama.cpp C/C++ sources — domain shift), `wt103` (a middle slice of wikitext-103 shard 2 —
  same-domain different-slice control).

## Results (matrix emitted by `wyly_ladder2.py`; logs in the review artifacts)

```
DATASET x RULE-LIBRARY LADDERS -- 80k L=256 windows per dataset, teacher-decision agreement

== dataset: wikitext ==
     teacher   gold  copy%   lib   agree  r-marg   core   cover  admitted
   pythia14m  0.244  27.9%   ext   0.380  +0.039  0.273   99.1%  [kgram k=3, induction L=1, kgram k=2, induction L=2, skip o=2]
   pythia70m  0.311  28.4%  base   0.353  +0.035  0.276   99.2%  [induction L=1, induction L=2, induction L=3]
                             ext   0.344  +0.049  0.269   99.2%  [kgram k=3, kgram k=2, induction L=1, skip o=2, induction L=2, induction L=3]
  pythia160m  0.374  26.4%   ext   0.301  +0.040  0.259   99.2%  [kgram k=3, kgram k=2, induction L=1, induction L=2, induction L=3]
  pythia410m  0.427  25.0%  base   0.284  +0.018  0.245   99.2%  [induction L=1, induction L=2, induction L=3]
                             ext   0.273  +0.032  0.244   99.3%  [kgram k=3, kgram k=2, induction L=2, skip o=2, induction L=1, induction L=3]
    pythia1b  0.447  24.6%   ext   0.277  +0.031  0.246   99.2%  [kgram k=3, kgram k=2, induction L=2, induction L=1, induction L=3]
  pythia1.4b  0.463  23.8%   ext   0.260  +0.037  0.234   99.2%  [kgram k=3, kgram k=2, induction L=1, induction L=2, skip o=2]
  pythia2.8b  0.485  23.5%   ext   0.261  +0.034  0.236   99.2%  [kgram k=3, kgram k=2, induction L=2, induction L=3, induction L=1, induction L=4]

== dataset: code ==
     teacher   gold  copy%   lib   agree  r-marg   core   cover  admitted
   pythia70m  0.692  69.3%  base   0.553  +0.204  0.294   99.8%  [induction L=2, induction L=3, induction L=1]
                             ext   0.567  +0.258  0.362   99.8%  [induction L=2, kgram k=3, induction L=1, induction L=3, induction L=4, induction L=5]
  pythia410m  0.764  66.3%  base   0.514  +0.172  0.294   99.8%  [induction L=2, induction L=3, induction L=1]
                             ext   0.531  +0.232  0.381   99.8%  [kgram k=3, induction L=2, induction L=3, induction L=4, induction L=5, induction L=1, kgram k=2]

== dataset: wt103 ==
     teacher   gold  copy%   lib   agree  r-marg   core   cover  admitted
   pythia70m  0.316  28.8%  base   0.366  +0.040  0.283   98.7%  [induction L=1, induction L=2, induction L=3]
                             ext   0.359  +0.049  0.274   98.8%  [kgram k=3, kgram k=2, induction L=1, induction L=2, skip o=2]
  pythia410m  0.433  25.4%  base   0.301  +0.023  0.257   98.8%  [induction L=1, induction L=2, induction L=3]
                             ext   0.289  +0.040  0.258   98.8%  [kgram k=3, kgram k=2, induction L=1, induction L=2, induction L=3]

wikitext ext-vs-base certified core across the pythia ladder:
     teacher base(s8.5)    ext   delta
   pythia14m      0.285  0.273  -0.012
   pythia70m      0.276  0.269  -0.007
  pythia160m      0.258  0.259  +0.001
  pythia410m      0.245  0.244  -0.001
    pythia1b      0.242  0.246  +0.004
  pythia1.4b      0.231  0.234  +0.003
  pythia2.8b      0.230  0.236  +0.006
```

### Findings

1. **Library saturation on the fixed dataset.** On wikitext the judge admits 5–6 of the 10
   candidates (k-gram k=3 always first, then k=2 / induction L=1,2 / sometimes skip o=2 or L=3) and
   *rejects* induction L=4/5, skip o=3, and repetition at the margin — the library is exhausted
   against this data, which is what "as far as you can" means operationally. Admissions shift with
   teacher scale (pythia-2.8b is the first to admit induction L=4).
2. **The cover-order/judge mismatch (the sharpest design finding).** In the *soft model*, the
   extended library pays more at every scale (rules-marginal up from +0.016…+0.035 to
   +0.031…+0.049). In the *package cover*, ext ≈ base (and slightly worse at small scale): the
   k-gram tier fires before the online counts and largely **displaces** them rather than adding.
   The judge admits by soft-model val marginal; the runtime cover has a fixed priority order — two
   different objectives. The certified tier's bottleneck on natural text is **not** rule supply; it
   is that most windows are already covered by an equally-good bigram answer.
3. **Dataset contrast.** **Code is a different regime.** pythia
   follows the induction pattern on **69%/66% of code windows** (vs ~25% on wiki) and its decisions
   are far more predictable (gold 0.692/0.764). There the library extension genuinely pays in the
   package cover — certified core **0.294 → 0.362 (70m, +0.068) and 0.294 → 0.381 (410m, +0.087)**
   — the judge admits ALL FIVE induction depths plus the k-gram tiers, and, uniquely, the certified
   core **rises with teacher scale** (0.362 → 0.381; wiki declines gently). The displacement problem
   (finding 2) is dataset-dependent: k-grams add real precision over the counts tier on code.
   `wt103` mirrors `wikitext` on every axis (base cores 0.283/0.257 vs 0.276/0.245; ext ≈ base;
   same admissions) — the same-domain control confirms the wiki findings are not slice artifacts.
4. **Scale trend is library-robust.** The §8.5 shape (gentle decline, flattening from 410m) holds
   under the extended library; the ext-vs-base delta at 2.8b is (+) while at 14m it is (−) — the
   bigger the teacher, the more the extra families help, consistent with §8.5's rising
   crystallization.

## Confounds (named, per the tag discipline — everything above is `empirical` over these ladders)

- **Fixed student capacity** (K=192 grounded-frozen concepts, one soft relational stack, V=4096
  teacher classes): falling agreement with scale bounds *this student*, not certifiability.
- **Hand-designed library**: candidates are enumerated, not learned; saturation means *this
  library* is exhausted, not that no rule would pay (the learned-proposer direction below).
- **Judge/cover objective mismatch** (finding 2): admission is measured in the soft mixture, value
  is realized in a fixed-priority cover — the reported `core` numbers understate what a
  cover-aware judge could certify.
- **k-gram tables are fit once** on a stride-12 subsample of the train region and never updated
  online; the counts tier updates every episode. Their relative quality is therefore not
  apples-to-apples across episodes.
- **Class truncation**: top-V=4096 teacher-decision classes per (teacher, dataset) — the kept-window
  sets differ slightly across rows (~1–4%); metrics are fractions of each row's own kept set.
- **Data volume and slicing**: 5 MB text slices, stride-1 windows sampled to 80k, corpus-ordered
  temporal split; `code` is a single project (llama.cpp), so its regularity partly reflects
  intra-project idiom reuse; `wt103` may share articles with pythia's training data (the Pile), so
  teacher memorization inflates predictability on both wiki datasets in unknown measure.
- **Single seed, single student run per cell**; teacher decisions computed once (fp32 ≤160m,
  fp16 above — argmax ties can flip under dtype).
- **Tokenizer edge**: code text required chunked encoding (added-token recursion in `pil.tokens`);
  chunk boundaries are newline-aligned and shared by student and teacher, so internally consistent.

## Design directions

1. **Cover-aware admission**: the judge should score candidates *under the package cover order*
   (or jointly optimize order + admission) — finding 2 says this is where the next certified points
   are. Cheapest version: admit by cover-marginal instead of soft-marginal.
2. **Support-weighted cover**: replace the fixed tier priority (gates → longest ngram → counts →
   induction) with per-rule confidence arbitration (support/determinism already ship in the
   manifest) — the runtime stays host-side and exact.
3. **Learned proposers over frames**: the enumerated families stop at suffixes and single offsets;
   rosetta's `gate` kind supports arbitrary frames ({offset: token} conjunctions). Mining frames
   from the student's *residual errors* (interaction-scored, as in the wyly_lm_bench proposer) is
   the natural next family — and the point where the library stops being hand-designed.
4. **Schema extension**: the repetition rule (and eq/copy relational rules generally) have no
   package kind; the PIC side already has eq-atoms. A `relation` kind (eq-guard + copy action)
   would let the certified tier express what the battery proved learnable.
5. **Online k-gram tiers**: promote the k=2,3 tables from fit-once candidates to online count
   structures like the bigram tier (the wake loop already touches every window).
6. **Per-dataset experts, one hub**: code vs wiki packages differ hugely in shape (cover, table
   sizes); serving them as separate sgiandubh spokes under claymore, with the hub routing by
   abstention, is the deployment-shaped version of this matrix.

## Follow-up: cover-aware admission, implemented (design direction 1)

`WYLY_JUDGE=cover` in `wyly_lm_v5.py`: the sleep judge scores each candidate by its marginal to the
**package cover** on held-out val — the objective the runtime realizes — instead of its vote in the
soft mixture. Two defects surfaced by the first re-run and fixed: (i) a **val leak** — the fit-once
k-gram/skip tables were fit on the train region *including* the val slice, inflating table rules'
val marginals for both judges (tables now fit on the fit-set only); (ii) **greedy threshold
myopia** — per-rule cover slices in a tiered cover are individually small, so the 0.002 threshold
rejected collectively-valuable tiers (threshold now 5e-4 for the cover judge).

Leak-fixed results (8 cells; `ladder2_cov.log`): the cover judge **matches or beats
max(base, ext-soft) in 6/8 cells with half the trusted tier** (2–3 rules vs 5–7): code/70m 0.369
(> soft 0.362, myopia fixed), code/410m 0.379 (≈ 0.381), wt103/70m 0.285 (> both arms),
wikitext/2.8b 0.238 (best of any arm). Residual gap only at the smallest scales (wikitext 14m
0.279 vs base 0.285) — the fit-once table quality bound, not the admission rule. Verdict: the
displacement problem is repaired at its source — admission and realization now optimize the same
objective, and the certified tier gets leaner and better simultaneously.

*(Produced 2026-07-05; cover-aware follow-up same day; see WYLY_LM_ENDGAME_REVIEW_FABLE.md §8.6.)*
