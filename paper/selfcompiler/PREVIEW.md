# The Self-Compiling Learner: Crystallizing Language-Model Behavior into Certified, Served Rule Programs

*Draft preview (markdown twin of `selfcompiler.tex`; figures in `figures/`; every number traced in `NUMBERS.md`).*

## Abstract

We present a continual learner whose consolidation phase is a **compiler**: during wake it learns
soft relational circuits by plain SGD; during sleep it hardens them to discreteness, extracts
their programs mechanically from the weights, certifies the extractions with an exact Datalog
engine, installs the certified rules as gradient-free structure, and returns the freed capacity to
the pool. A certified rule is forgetting-proof because it is no longer weights. On a synthetic
curriculum the compile arm retains all tasks perfectly (1.000/1.000/1.000) where a matched-budget
baseline collapses (0.010/0.002/0.007); the flagship circuit — a discrete induction head learned
by plain SGD — carries a machine-checked certificate of exact agreement with a three-line Datalog
program on its full test domain (2048/2048). Pointed at real text and at the Pythia family as
teachers, the learner's judge admits exact rules only when they pay on held-out data, and the
resulting **certified core** — a count table plus a handful of proved rules, served host-side with
citations and honest abstention — captures a fraction of teacher behavior that does **not**
collapse with teacher scale (crystallization rises to a ~85% plateau across 200× parameters) and,
under support-weighted arbitration, can **exceed the student that discovered the rules** (0.60 vs
0.57 agreement with a code-domain teacher). An independent corpus-statistics decompiler agrees
with the learned rules on 99.2% of the contexts both cover — two-instrument confirmation of what
the teacher is. Methodology is reported as a first-class result: every claim carries a
proved/empirical tag over a stated domain, and the paper's negative results — a conjunction-free
benchmark, admission by val-variance noise, a cover-order bottleneck worth more than every rule
family combined — were each caught by the machinery itself.

## 1. Introduction

Two research programs approach the same artifact from opposite sides. *Mechanistic
interpretability* starts from a trained transformer and mines circuits out of frozen weights;
*neurosymbolic learning* starts from symbols and asks how much behavior rules can carry. This
paper builds the bridge as a single running system: a learner that behaves like a small language
model while it runs, and like a decompiler while it sleeps.

Three design commitments. **Memorization is structural**: a lookup table updated by counting, not
gradient — plain SGD provably underfits lookup tables, and a count table is what our serving
format ships anyway. **Relational computation is a rule form**: a relational head is a T3
threshold gate × a bilinear matcher over raw token codes × *successor routing* (copy what followed
the matched position), with a decode tied to the token-code table — an attention head restated as
a discrete gather, and the form that makes extraction possible. **Consolidation is compilation**:
sleep does not merely protect weights; it anneals a circuit to its hard-routed self, reads the
program out of the weights, certifies the program against the circuit's own behavior with Soufflé,
and replaces the circuit with the program.

**Contributions.**
1. A relational rule head that learns induction to 0.999 under **plain SGD**, hard-routes at
   0.998, and — after a hardening anneal — carries a certificate of exact agreement with a
   three-line Datalog program on all 2048 test windows **[proved, stated domain]**, with
   mechanical extraction (family from the vocabulary match matrix, guards from the certified data
   domain, tie-breaks from the form). (§4)
2. **Sleep-as-compilation**: certified rules installed as preempting structure, soft sources
   recycled. Zero forgetting on a three-task curriculum without replay vs a matched-budget
   baseline that forgets everything; the autonomous version on real text discovers its own rules
   and **beats the hand-built variant**. (§5, §6)
3. **The certifiability scaling law**: across Pythia 14M–2.8B the certified fraction declines
   only gently (0.285→0.230, near-flat above 410M) while crystallization (core/student) **rises**
   from 74% to a ~85% plateau. (§7)
4. **The race**: the learning route and an independent corpus-statistics decompiler agree on
   **99.2%** of contexts both cover, while optimizing opposite operating points. (§7)
5. A rule-library study whose sharpest finding is architectural: with a fixed-priority cover,
   new families mostly **displace** the count tier; **support-weighted arbitration** is worth more
   than every family combined — doubling the certified core on code, where the core then
   **exceeds the soft student that discovered the rules**. (§8)
6. A deployment path: a `relation` schema kind landed across the schema and two runtimes with
   verified parity; per-dataset experts federated under an abstention-routing hub, every answer
   citing the sleep cycle that admitted its rule. (§9)

**Methodology as a result.** Every claim is tagged — **proved** only with a machine-checked
artifact over a stated domain, otherwise **empirical** — and the corrections were produced by the
machinery itself: an exact certificate caught a mirror bug; a val-variance analysis caught noise
admissions; the inherited reference number (0.189 "teacher decode") was exposed as a probe
artifact (true: 0.244).

## 2. Related Work

**Induction heads / circuits** (Elhage et al. 2021; Olsson et al. 2022): we learn the circuit as a
discrete rule, certify it exactly, and measure its durability across scale (teacher copy-pattern
incidence declines only 28%→23.5% over 200×; 66–69% on code). **Rule induction**: our proposers
are deliberately classical — sequential covering and conditional refinement (CN2, FOIL, RIPPER,
ILP) reappear as interaction-scored, error-driven frame mining; that *marginal* scoring cannot
beat a linear path is a corollary the field knew. **Wake/sleep + library learning** (Hinton et
al. 1995; DreamCoder): the consolidation product here is a *certificate*, not a posterior.
**Distillation** (Hinton et al. 2015): we target decisions and ask how much of the teacher
crystallizes into exact structure. **Model decompilation**: the analysis route mines gated
n-grams/causal idioms from corpus statistics and frozen weights; the 99.2% agreement between
routes is, to our knowledge, the first two-instrument cross-validation of a decompiled artifact.

## 3. The Learner

**Substrate.** Token codes `C ∈ R^{V×K}` are **frozen** after initialization from a teacher's
embedding geometry (PCA, z-scored) — grounding buys the linear path +0.023, and gives continual
learning a stationary concept space (a trainable C with a tied decode lets task *k* wreck unseen
tasks' class rows) [empirical].

**Paths.** (i) an online **count tier**; (ii) the **relational head**: T3 gate σ(β(ρ·ℓ−θ)) over
query literals; bilinear matcher `s_i = φ_iᵀ A q/√K + u·φ_i` over **raw** position codes
(near-orthogonality is what the matcher needs — sigmoid-squashed features sit at chance); softmax
routing; **successor routing** (the value written back is the matched position's successor code);
tied decode `logits = qCᵀ`; (iii) **installed rules**: exact programs that preempt.

**Wake/sleep.** Wake: plain SGD over a streaming episode + count updates. Sleep: structural —
harden, extract, certify, install, recycle; propose and judge library candidates. Sleep may use
heavier optimization (consolidation is not the wake learner).

**The judge.** A rule is admitted only if it improves held-out val agreement *under the cover
semantics the runtime will realize*; high-variance families carry stricter thresholds.

## 4. Certified Relational Circuits

Battery (L=128; all baselines lr-piloted; the transformer gets ~4× the student's steps):

| task | old form | transformer 2L (30k) | **head (SGD)** | hard-route |
|---|---|---|---|---|
| induction | 0.022 | 0.023 | **0.999** | **0.998** |
| marker | 0.019 | 1.000 | **1.000** | **1.000** |
| khop2 | 0.008 | 0.005 | 0.010 | — |
| khop2 (Adam) | | | 0.920 | 0.224 |

The transformer is not a strawman (1.000 on marker); it never undergoes the induction phase
transition at this scale/budget, while the rule form has the circuit's shape built in.

**Hardening to proved.** β-anneal (1→24) + straight-through takes induction to soft = hard =
1.000; the certificate — tensor hard-route vs `hit(w,p) ← tok(w,p,t), qtok(w,t)`;
`best = max hit`; `pred = successor(best)` — reaches **2048/2048** [proved, stated domain]. The
same anneal makes khop2 **discrete** (0.987 = 0.987), closing the multi-hop softness wall (Adam in
sleep; wake-SGD-only multi-hop open).

**Mechanical extraction.** Fidelity scan over (layer, head); vocabulary match matrix
`M(a,b) = φ_aᵀ A C_b` diagonal-dominance (token equality); gate saturation; successor routing and
most-recent tie-break from the form. The program is *assembled from findings* and verified
extensionally: extracted ≡ hand-written ≡ tensor on all 2048 windows. The exactness matters: the
Soufflé check caught a real mirror bug (a zero-pad slot colliding with a shifted token id).

## 5. Sleep as Compilation

Installed rules **preempt** (inside their guard = the certified data domain) — an additive vote
fails because soft logits outgrow any fixed weight [empirical]. Compiled heads are **reset**;
capacity returns to the pool.

Curriculum (induction → marker → khop2, one model, no replay): each sleep compiles what wake
sketched — including watching the 2-hop circuit *form* across extended sleeps (0.806 → 0.960 →
0.995, then compile). Final: **1.000/1.000/1.000** with heads reset three times; the
matched-budget no-compile baseline: **0.010/0.002/0.007** (Fig. `curriculum.pdf`). A certified
rule cannot be forgotten because it is not weights — and this is now a machine-checked theorem:
the C-series companion (i-orca `examples/concept_grounding`, Isabelle kernel) proves the
preempting cover's covered-domain behavior independent of the soft function (hence identical at
any two instants of any training trajectory), agreement sets on certified subsets exactly
preserved (`retention_by_compilation`, `certified_accuracy_invariant`), and — with
pairwise-disjoint guards — the firing rule sovereign and installation order irrelevant
(`cover_order_irrelevant`): the obligation the curriculum discharges with disjoint token ranges.
Zero forgetting is a property of the install semantics, not of training dynamics
**[proved, over the stated cover structure]**.

## 6. Real Text

Wikitext, 256-token context: no prior variant had beaten a trained bigram lookup (0.148); the
assembled learner reaches **0.199** online, and the package-expressible **certified core** alone
reaches 0.191 — a certifiability tax of 0.008. Certification produced the sharpest diagnosis: the
learned soft head is a weak approximation (0.022 hard-alone) of the program it approximates
(0.614 on the copy subset), so the certified program was **installed** — and the autonomous
version discovers its rules itself (rejecting all candidates for two sleeps, then admitting
induction L=1 and L=3), ending at **0.201** with a copy-subset marginal of **+0.058 vs +0.042**
for the hand-built variant: *the machine's rule choices beat ours* [empirical].

## 7. Imitating Transformers: the Race and the Scaling Law

**Teachers.** Pythia decisions on the same 80k windows. (The inherited "final-residual decode
0.189" was a linear-probe artifact; the true decode is 0.244 at 12-token context; teacher gold at
L=256 runs 0.244→0.485 up the ladder.) Teachers are *more* rule-shaped than their corpus: copy
incidence 29% vs gold 20% at 70m; all three induction depths admitted within three sleeps; rule
value ~2.5×.

**The race** (same corpus region, same held-out windows, same teacher):

| arm | cover | agree | when-fired |
|---|---|---|---|
| analysis (shipped, det≥1.0) | 3.6% | 0.025 | **0.688** |
| analysis (det≥0.5) | 16.9% | 0.089 | 0.523 |
| **learning: certified core** | 99.2% | **0.276** | 0.279 |
| learning: full student | 100% | 0.352 | 0.352 |

Each route wins its own objective. The deepest result is **convergence**: on the 480 bigram
contexts both routes rule on, they give the same answer **476/480 (99.2%)**.

**The scaling law** (Fig. `scaling.pdf`): certified core 0.285→0.230 (near-flat above 410M) — *no
collapse across 200×* — while crystallization (core/student) **rises** 74%→~85% plateau.
Structure splits as circuit theory predicts: n-gram-shapedness decays fastest (−24%); the
induction bump is durable (−16%). Controls: adjacent teachers agree 0.51→0.75; the core's 0.230
vs pythia-2.8b ≈ 65% of a *full 14M transformer*'s 0.351. Confounds stated: fixed student
capacity/library bound the student, not certifiability [empirical, this ladder and these windows].

## 8. The Rule-Library × Dataset Matrix

Library extended to judge saturation (induction L=1..5, k-gram tables, skip-bigrams, gapped gate
frames — grid-enumerated then **mined**: anchored {offset:token} frames + 2-anchor conjunctions
from residual errors, interaction-scored at anchor granularity; relation rules; online tiers)
across wikitext, a wikitext-103 control, and C/C++ source.

Three portable lessons, each caught by the machinery: (1) **overlapping-window statistics inflate
support** (stride-1 windows repeat each pair ~21×; a naive gate is no gate); (2) **high-variance
candidates re-admit under val noise whenever offered** — pre-gate strength must scale with
variance; (3) **the fixed-priority cover was the dominant bottleneck** — families *displaced* the
count tier.

Fixing (3) is the headline. **Support-weighted arbitration** — every applicable rule fires,
highest confidence wins; per-key Laplace-shrunk determinism `c/(t+α)` for tables (fields the
manifest already ships), held-out fired-accuracy for scalar kinds (one new field) — is worth more
than every family combined: +0.032–0.047 on natural text and a **doubling on code** (0.31→0.60),
where the certified core **exceeds the soft student that discovered the rules** (0.604 > 0.569)
[empirical]. The arbitration guarantee is also now formal: with per-cell *calibrated*
confidences, argmax arbitration dominates every policy — every fixed tier priority included —
and an ε-miscalibrated arbiter is within 2ε·(total weight) of optimal (`argmax_policy_optimal`,
`miscalibration_bound`; kernel-checked) **[proved, over the stated finite cells]**. Mined frames are then the only family still adding on natural text (+0.005–0.007,
arc bests), and correctly decline where recovery is thin. (Fig. `library.pdf`.)

## 9. Deployment

The rules live in an expert-package schema served host-side. We contributed the `relation` kind
(eq-guard + copy; the learned repetition rule `eq=[[1,2]], copy=1` admitted at 0.944
fired-accuracy) and the `confidence` field to the schema and both runtimes (Python reference +
C++ spoke) with verified parity — the parity test itself caught a routing bug. The spoke serves an
OpenAI-compatible endpoint whose cover miss is an explicit abstention; a federating hub routes
*by* that abstention: wiki → wiki expert, code → code expert, never-seen gibberish → **refused in
code**. Every answer cites the count evidence or the sleep cycle that admitted its rule. Two
deployment lessons: word-level relevance gates never overlap subword citations (BPE splits
identifiers); doubled-token "gibberish" *legitimately* fires the repetition rule.

## 10. The Second Campaign: Roles, Not Memory

An eight-corpus study at the 70M rung found a sharp crystallization threshold at detokenized-gzip
≈ 0.33 (structured domains >100%, prose 91–96%; teacher predictability ≈ compressibility, ρ =
−0.98). The campaign then removed the gap with structure, never memory: concept induction,
calibrated pointer rules, and — decisively — **derived roles**: the package became a two-layer
program (Soufflé-certified extractors + dgate rules; composed and chained roles including the
entity echo; 192/192 per kind), with fold-stable admission. **The threshold dissolved**: every
corpus now crystallizes at 96–108% (wikitext 99.5%), five arc bests fell, and the two worst
crystallizers moved most (German +0.029 via the composed clause-successor role — the verb-final
channel; math +0.026 via referent + learned member-set roles, admitted on 6/8 corpora). The
prose/formal gap was a rule-vocabulary artifact; the absolute ceiling still tracks structure
(ρ = −0.93) — roles close the ratio, teacher predictability bounds the rest. (Fig. campaign2.pdf.)

## 11. Confounds and Limitations

Fixed student capacity; hand-seeded (though data-selected) families; top-V truncation; 5 MB
slices with stride-1 overlap; possible teacher memorization of wiki text (the Pile); single
seeds; fp16 teacher decisions above 410M; the code corpus is one project; the ladder stops at 2.8B
(VRAM). Certification is over *stated domains* — proved never travels past its test set.
Wake-SGD-only multi-hop remains open; the runtimes' shipped cover is still fixed-priority — the
arbitration gains are measured in the learner and proposed to the schema, not yet realized in
serving — **now closed**: both runtimes implement the support-weighted cover, the emitter ships
every admitted family with its arbitration confidences, and replaying all held-out windows
through the served runtime reproduces the learner-measured core *exactly* (0.605 code / 0.329
wikitext, gap +0.0000; C++/Python parity 200/200). Calibration is now a *formal* requirement
rather than an observation: C10's dominance
premise is per-cell calibration on the evaluation measure, and its 2ε envelope is exactly what
low-support confidence inflation (the val-variance episodes) consumes — support pre-gates and
per-family judge thresholds are what keep ε small.

## 12. Conclusion

The loop runs end to end with no transformer in the inference path: **learn (SGD, wake/sleep) →
harden (anneal to discreteness) → certify (Soufflé, proved on stated domains) → extract (weights
to Datalog, mechanically) → install (preempting structure) → package (provenance + confidence) →
serve and federate (citations, honest abstention)**. The durable findings: consolidation can be
compilation, and certified structure is the strongest anti-forgetting mechanism we tested; a
transformer's behavior is more crystallizable than its training text, and more so the more
structured the domain; the certified fraction does not collapse with teacher scale; arbitration
beats rule supply; and two independent routes to the same artifact agree where they overlap — the
closest thing a decompilation program has to external validation.
