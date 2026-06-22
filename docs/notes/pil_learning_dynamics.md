# PIL learning dynamics — design notes

Working notes for *why* the pil loop is shaped the way it is. Companion to
`fieldrun/PIC_PROPOSAL.md` (the inference-time theory) and
`fieldrun/experiments/rule_synth/PIC_LOSSINESS.md` (what is / isn't compressible).

## 1. The two-phase loop

LLMs during training appear to build **massively parallel partial solutions** (features,
circuits, residual contributions) that are then **selected from and gated** (attention,
normalization, margin, loss). PIL models that explicitly:

1. **Generative step** — propose many `d_j ∈ H` in parallel (`propose_parallel_sources`,
   or `pil.fieldrun_io.load_probe_sources` for real ones). Each `d_j` is an *incidence*: a
   signed vote `c_j^v = ⟨d_j, U_v⟩` toward every proposition.
2. **Learn + gate step** — aggregate `L_v = Σ_j c_j^v`, decide by margin / power diagram,
   and update the frame `U` so that more positions become high-margin "retrieved" and fewer
   stay in the diffuse "computed" regime.

The frame `U` is the only thing PIL owns and learns. The sources are *proposed* (synthetic
now; real fieldrun DLAs later). This keeps the experiment honest about which half is being
optimized.

## 2. The decode-side / frame-side split is load-bearing

Every loss term is one of two kinds, and they can trade against each other:

- **decode-side** — depends on the readout: `nll` (fit the target distribution) and the
  hinge `margin` (push the worst-competitor margin past a target). These are exactly the
  quantities fieldrun measures on a frozen model.
- **frame-side** — intrinsic to `{U_v}`: the **frame potential** (mean squared off-diagonal
  cosine of the Gram), driven toward the **Welch floor** `(V−dim)/(dim(V−1))`. This is the
  PIC-T2 "improve the Gram structure" objective: `ρ` off-diagonal → 0 is the move toward the
  diagonal-`G`, classical-incidence-calculus limit.

Why split them: optimizing decode-side margins can **degrade** encode-side (co-firing)
structure. We measured this directly in polygram PR #113 — readout-aligned dictionary
geometry dropped `Spearman(Polygram, co-firing)` 0.64 → 0.27, because co-firing is
encode-side and the readout alignment is decode-side. So PIL never collapses the two into a
single "interpretability" scalar; it reports them separately and lets the experiment show the
trade.

The starter's Gram term penalized `G.pow(2).mean()` *including the diagonal* — i.e. it
penalized frame **norms** (`‖U_v‖⁴`), fighting the margin term. The fix
(`frame_potential`) penalizes **only off-diagonal cosine**, decoupled from scale, with the
Welch bound as the honest floor (you cannot reach 0 when `V > dim`).

## 3. Why PIL is allowed to move the forge tax

The forge tax (composed fraction) is **not** reducible by *frozen* re-expression of a fixed
`U` — that plateau is now kernel-confirmed (i-orca `examples/tropical/HeadTail.thy`: the
compact head reproduces the decode only when it out-values the open-class tail; the tail is
the explicit irreducible residue). But it **is** movable by *retraining the subspace*
(entangled-core: retrained rank-8 bottleneck lossless ~30× below the frozen floor;
sae-forge: train the subspace, not the encoder).

PIL updates `U_v` → it is a subspace-retraining method → it sits on the side where the tax
moves, **at the cost of host-model fidelity**. So pil's headline objective ("raise the
retrievable fraction / lower support number / shrink the composed fraction") is an *open,
achievable-in-principle target*, not a frozen-compression plateau — and not yet a result on
any real model. Stating it any more strongly would violate the no-necessity-without-proof
discipline.

## 4. Diagnostics that tie back to fieldrun's measurements

- **support number** `σ(t)` — smallest source set whose partial sum crosses threshold;
  RETRIEVED ≈ small `σ`, COMPOSED ≈ large `σ` (PIC D4). Conjecture `σ ∼ PR` (PIC O2).
- **participation ratio** — `(Σw)²/Σw²` over `|c_j^t|`; the diffuseness measure of T4. PIL
  tracks whether learning *concentrates* support (lowers PR), never claiming the floor → 1.
- **frame potential vs Welch** — over-completeness-honest conditioning of the Gram.

The split between RETRIEVED/SELECTED/COMPOSED is **model-dependent** (fieldrun
`FINDINGS_PYTHIA.md`: ~25–37% / 50–61% / 8–15% across Qwen-0.5B and the Pythia 70m→1b
ladder), so any pil "we shifted the split" claim must name the substrate and the cost.

## 5. First synthetic finding (2026-06)

Planted-frame synthetic, `dim=32`, `V=64`, `J=24`:

| `frame_reg` | retr% | top1 | margin | fp/welch | support PR |
|---|---|---|---|---|---|
| 0.0  | 98.4 | 100 | 2.45 | 1.63 | 4.96 |
| 0.05 | 98.4 | 100 | 2.44 | 1.61 | 4.96 |
| 0.2  | 98.4 | 100 | 2.45 | 1.54 | 4.96 |

Reading: the frame-side term decorrelates the Gram monotonically at **no** decode-side cost
here, but the effect is small — this regime is too easy (the margin term alone already yields
a near-incoherent frame at 1.6× Welch, and support PR is pinned by the planted sparsity).
The frame objective is expected to matter only when the frame is genuinely stressed:
`V ≫ dim`, high planted synonymy `ρ`, or real fieldrun-seeded sources where the Gram is dense.

## 5b. Hard regime — the frame-potential term is decoupled from the decode (negative result)

`experiments/hard_synthetic.py`: over-complete (`dim=24`, `V=192`, 8×) with planted synonym
clusters (24 clusters, `spread=0.10` ⇒ planted within-cluster `|cos| = 0.82`, Welch floor
0.037). The regime *worked as designed* — `syn_comp ≈ 0.88` (the strongest competitor is a
synonym 88% of the time), and synonymy compressed margins so hard that `retr ≈ 0.40` while
`top1 ≈ 0.99`. Sweep over `frame_reg` (3 seeds, 1500 steps):

| `frame_reg` | retr | top1 | margin | nll | fp/welch | wc_cos | align(U,U_gt) |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.395 | 0.99 | 1.240 | 1.949 | **1.346** | 0.291 | 0.634 |
| 0.10 | 0.400 | 0.99 | 1.241 | 1.949 | 1.244 | 0.286 | 0.637 |
| 0.50 | 0.395 | 0.99 | 1.242 | 1.948 | **1.094** | 0.278 | 0.642 |

The frame term **does** what it claims — drives the frame potential 1.35 → 1.09× Welch and
lowers learned synonym coherence — but every decode outcome (retr/top1/margin/nll) is **flat**.
The predicted non-monotone optimum did **not** appear.

Confirmed across the whole hardness plane (`Δretr = retr(fr=0.2) − retr(fr=0)`, single seed,
1000 steps; baseline retr in brackets):

| spread＼noise | 0.04 | 0.08 | 0.16 |
|---|---|---|---|
| 0.05 | +0.000 [0.02] | −0.002 [0.06] | −0.006 [0.38] |
| 0.10 | +0.004 [0.32] | +0.000 [0.36] | +0.000 [0.56] |
| 0.20 | +0.002 [0.80] | +0.010 [0.78] | −0.004 [0.79] |

`|Δretr| ≤ 0.01` in every cell, while the baseline swings 0.02 → 0.80 with the *real* levers
(synonymy `spread`, source SNR `noise`). Note these are **in-sample** retr (no holdout) — the
easiest possible case for a regularizer to look good — so the null is *conservative*: a term
that can't help even in-sample won't help out-of-sample.

**Conclusion (descriptive).** The frame-potential regularizer is **redundant where the margin
term already shapes the frame, and impotent where the bottleneck is source distinguishability.**
When synonyms have near-identical *sources* (`d_j`), the achievable margin is bounded by source
SNR, and no readout-frame (`U`) objective can manufacture signal that isn't in the sources —
the information bottleneck is on the generative side, not the frame side. This is the PIL-local
echo of the decode-vs-encode lesson: shaping the readout doesn't fix an upstream representational
limit. **Roadmap consequence:** the lever for the confusable / forge-tax regime is the
**generative proposer** (item 3 — propose sources that distinguish synonyms: SAE features,
Gram-orthogonal complements, real fieldrun DLAs), *not* frame regularization. Keep `frame_reg`
as a cheap, harmless conditioner; stop expecting it to move retrievability.

Open variant not yet tried: a **targeted** frame penalty (only on confusable pairs `ρ > τ`)
rather than the global frame potential — though the surface above suggests even that cannot beat
a source-SNR bound.

## 5c. Compositional regime — generation IS the lever; targeting barely matters (positive + null)

`experiments/compositional_pil.py`. The §5b negative result said the bottleneck is generative,
not the frame — so this tests it directly. Synonyms share a one-hot **topic** atom (collide on
`r`); a fraction of clusters are **hard** = the synonym parity is the **XOR** of two code-atoms
(provably not linearly separable in `z`), the rest **easy** = parity read directly. A *frame-only*
model is a linear readout of `z`; a generated rule is a ReLU hidden unit `relu(<w,z>+b)` emitted as
a source. Three arms matched on total steps — `frame` (no rules), `untargeted` (M random rules),
`targeted` (M rules seeded at the min-margin clusters' code-atoms, the clusters **discovered** from a
frame-only warmup, not the planted flags). **Eval is held-out 25%** (the §5b in-sample caveat, fixed).

Headline = hard-cluster within-synonym accuracy (the XOR a linear readout cannot do):

| arm (signal=True, held-out) | hard_within_acc | hard_within_margin |
|---|---|---|
| frame (M=0) | **0.454** | −0.024 |
| untargeted M=8 | 0.790 | 2.294 |
| targeted M=8 | 0.765 | 2.024 |
| untargeted M=16 | 0.899 | 3.305 |
| targeted M=16 | 0.910 | 3.311 |
| untargeted M=32 | 0.911 | 3.373 |
| targeted M=32 | **0.934** | 3.812 |

Control `signal=False` (parity is noise), held-out: frame 0.537, targeted 0.530, untargeted 0.490 —
**all at chance.**

**Two findings, honestly separated:**

1. **POSITIVE — generation is the lever (theory's main claim confirmed).** Frame-tuning is pinned at
   chance on the XOR (0.454, margin ≈ 0 — linear *provably* can't); adding rules lifts hard-cluster
   accuracy to **0.93**. When the bottleneck is non-linear (composed / PIC-T3 weighted-threshold)
   structure, the **generative step** moves it where no frame objective could (§5b). The held-out
   control confirms the honest boundary: generation recovers signal only when it exists — it cannot
   manufacture an absent discriminator (vs the in-sample 0.64 mirage from rule overfitting).

2. **NULL — min-margin targeting barely beats spraying.** Targeted gives only a marginal high-budget
   edge (0.934 vs 0.911 at M=32) and is slightly *behind* at low budget (0.765 vs 0.790 at M=8). The
   mechanism is clear: **SGD already allocates rules to the at-risk facets** (the loss gradient drives
   hidden units to the hard clusters), so explicit targeting adds an *initialization* head-start, not
   better *allocation*. The frame-only warmup also can't rank *among* hard clusters (all ≈ chance), so
   targeting collapses to "put rules on hard clusters" — which SGD does for free. At low budget the
   rigid "fully fix a few clusters" (targeted) and the flexible "partially help many" (untargeted SGD)
   average to the same hard-cluster accuracy.

**Reading for the tropical principle (§ of the design conversation).** The decode-side *diagnosis*
tropical algebra offers — which facets are at-risk, which rules are dead — is real, but in a
differentiable setting **gradient descent already performs the allocation that targeting would
hard-code**. So the tropical contribution here is *understanding / pruning / the capacity ceiling*,
not a better optimizer than SGD for placing rules. Targeting should only win where SGD's credit
assignment fails — very low budget with many hard facets and weak gradients — which this regime does
not stress. That sharper regime is the open test.

## 5d. The generative lever is training the feature, not selecting it (the decisive 2×2)

`experiments/scored_proposer.py`. Tests the propose-score-select blueprint, crossing **selection
method** × **frozen/trainable input-weights** to disentangle "selection quality" from "trained vs
frozen feature." A rule is `relu(⟨w,z⟩)`; `ambiguity` selects `w` by an η²-on-ambiguous-examples score,
`random` by nothing; `sgd` is random-init trainable. Held-out, 3 seeds — `acc` = hard-cluster
within-synonym accuracy (chance 0.5); `code` = clusters with **both** synonyms confidently decoded
(the realized γ-separated code, `DecodeCapacity.thy`; max 24):

| budget | sgd | ambig-frozen | ambig-train | rand-frozen | rand-train |
|---|---|---|---|---|---|
| 8  | 0.722 / 24 | 0.532 / 21 | **0.741 / 24** | 0.540 / 21 | **0.750 / 24** |
| 16 | 0.859 / 24 | 0.579 / 23 | 0.852 / 24 | 0.632 / 24 | 0.854 / 24 |
| 32 | 0.905 / 24 | 0.684 / 24 | 0.891 / 24 | 0.668 / 24 | 0.873 / 24 |

**The split is entirely along frozen vs trainable, not selection vs random.** Making the scored arm's
weights trainable (`ambig-train`) jumps it 0.53→0.74 at M=8, **matching SGD**, and `ambig-train ≈
rand-train ≈ sgd` at every budget. So:

1. **Selection is not the lever (falsifies "proposal quality is the bottleneck").** Once rules are
   trained, theory-guided selection is indistinguishable from random init. Once they are frozen, no
   selection method rescues them. The §5d-v1 "negative" was the frozen-vs-trainable axis, not selection.
2. **Capacity is not the binding constraint here (refutes the separation-lemma-as-bottleneck reading).**
   The structural metric **saturates**: even frozen rules at 0.58 accuracy realize the full γ-separated
   code (24/24 clusters have both synonyms confidently decoded *somewhere*). So the bottleneck is **not**
   "creating new γ-separated high-margin tokens" — that capacity is already realized. The gap is
   **per-example routing**: getting the *right* decode for *each* input, which is a *training* problem
   (the input→firing map), not a capacity or selection one.

**Quantified (`experiments/capacity_diagnostic.py`, Grok diagnostics #2/#3).** The saturation is *not* a
capacity ceiling — it is the metric's max (24 = n_clusters). For the trained model (`dim=32`, `V=48`):
the realized γ-code is `48/48` tokens, while the packing bound `(1+2ρR/γ)^d` is `~1e59` (γ=1) / `~1e51`
(γ=1.8) — **capacity slack by ~50–59 orders of magnitude** — and the min frame separation inside the code
(`0.90`) exceeds the effective γ by `17–31×`. So all 48 tokens are γ-decodable with enormous slack; nothing
is near the packing limit. The binding constraint is entirely per-example routing, not capacity.

This completes the triangulation across **three** independent angles — frame regularization (§5b), rule
allocation under SGD (§5c), proposal scoring (§5d) — none beats plain end-to-end gradient training of
the rules. The tropical/PIC theory's value is the **capacity limit** (kernel-proved `DecodeCapacity.thy`),
which bounds *which* tokens can be confidently decoded — but within that capacity, the realized accuracy
is set by trainable per-example routing, which the theory does not optimize. The blueprint's heavier
MILP / tropical-LP proposers were deliberately skipped: a 2×2 already shows no selection beats SGD, and
the binding constraint is routing, not selection — so a better *selector* cannot help.

## 5e. Realization cost is M-dominated, not per-routing-decision (a surprise vs the collapse hypothesis)

`experiments/routing_complexity.py`. To operationalize "realization complexity" (Grok's TRC) before
formalizing it, hold cell capacity fixed (slack ~1e59, §5d) and vary the **number of independent
non-linear routing decisions** `n_hard` (XOR-coded clusters), measuring hard-cluster accuracy vs the
rule budget `M`. Hypothesis: accuracy is governed by `M / n_hard` (rules per routing decision), so the
curves collapse against that ratio. **The data refuted it.** Held-out, 2 seeds:

| n_hard ＼ M | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| 4  | 0.465 | 0.518 | 0.808 | 0.915 | 0.935 | 0.950 |
| 8  | 0.496 | 0.529 | 0.757 | 0.840 | 0.900 | 0.917 |
| 16 | 0.498 | 0.521 | 0.745 | 0.889 | 0.908 | 0.947 |
| 24 | 0.535 | 0.595 | 0.759 | 0.858 | 0.915 | 0.948 |

**Accuracy depends on the absolute budget `M`, essentially not on `n_hard`.** Read the columns: at fixed
`M` the accuracy is ~flat across 4→24 routing decisions (M=8 → ~0.76 for all; M=32 → ~0.91 for all). The
collapse-against-`M/n_hard` fails (spread 0.40 at ratio 1), and the minimal `M` to reach 0.85 is ~16–32
**regardless of `n_hard`** (so `M*/n_hard` falls from 4.0 to 0.67). So **24 disjoint XOR routings need no
more rules than 4** — realization cost does **not** scale with the number of routing decisions.

**Interpretation — superposition.** The `M` ReLU rules represent the `n_hard` routing features **in
superposition**, amortized across decisions: 16 rules route 24 disjoint XORs at 0.86, 32 at 0.91, with
accuracy set by the rule budget (how many features can be disentangled) and the interference, *not* by
the decision count. This is the same superposition the rest of the program studies (Welch bound, the
entangled core), now appearing on the *routing* side. So **TRC is a superposition-capacity quantity — how
many routing features a budget of `M` rules can pack with tolerable interference — not a per-decision
count.** That reframes the theory target (see §6 / Q2′): the realization bound should be Welch-flavoured
(packing routing features into `M` rules), not a circuit-style "one gadget per XOR" count.

Caveats: 2 seeds, mild non-monotone noise in the `n_hard` direction; the `M`-dominance is the robust signal.

## 5f. The interference is mild and lives in ℝ^M — coherence tracks the Welch floor only when n > M

`experiments/interference_probe.py`. The open routing-side conjecture says: at fixed rule budget `M`
and slack dimension `d`, packing more routing features `n_hard` into the rank-≤min(M,d) subspace should
degrade the margin via Welch-floored cross-talk. We test it with the margin (not just accuracy, which
§5e showed saturates) plus the **realized coherence** `coh_μ` = mean off-diagonal |cosine| of the
routing features `f_c = E[h | 2c] − E[h | 2c+1]` (the per-cluster routing direction in **rule-activation
space** `ℝ^M`), against the Welch floor `√((n−M)/(M(n−1)))`. Held-out, 3 seeds, `d=32`:

| M | margin (n=1 → n=24) | coh_μ | welch_floor (n=24) | verdict |
|---|---|---|---|---|
| 8  | 2.10 → 1.83 (mild, only once n>8) | ~0.28, flat | 0.295 | weak degradation; `coh_μ ≈ welch_floor` for n>M |
| 16 | 2.93 → 3.11 (flat/rising) | ~0.19, flat | 0.147 | no degradation; n≤M leaves rank room |

**Three honest findings:**

1. **The clean Welch-degradation curve is not present.** M=16 is flat; M=8 declines only ~13% and
   non-monotonically. No cliff, no strong graceful-degradation signal.
2. **But the dimension in the conjecture was wrong — it's `M`, not `min(M,d)`.** The routing features
   `f_c` live in **rule-activation space `ℝ^M`**, so the relevant Welch packing is `n` features in `M`
   dimensions. Interference appears precisely when **`n_hard > M`** (M=8: margin dips and `coh_μ ≈ 0.28`
   sits at the Welch floor `0.21–0.30`); for `n ≤ M` there is rank room, `coh_μ` stays low, margin holds.
   This is also *why* §5e was M-dominated: `M` is the feature-packing dimension.
3. **Training packs near-optimally, so the degradation is gentle.** Realized `coh_μ` tracks (does not
   wildly exceed) the Welch floor, and margins stay well above 0 even at `coh_μ ≈ welch_floor` — the
   learner finds a near-Welch-optimal routing code, so exceeding `M` costs only mild margin, not failure.

**Reframed conjecture (for the next round).** TRC's interference term is a Welch packing of routing
features into **`M` (the rule count)** dimensions: feasible margin `γ(n, M) ≈ γ₀(1 − c·√((n−M)/(M(n−1))))`
for `n > M`, and `≈ γ₀` for `n ≤ M`. The `i-orca` target is therefore a Welch bound on the coherence of
`n` vectors in `ℝ^M` (the realized routing features) — directly an instance of the existing Welch
machinery, now on the rule-activation side. The mildness (near-optimal packing) is the empirical caveat.

## 5g. A generator-side coherence regularizer is also a null — the symmetry closes

`experiments/coherence_reg.py`. Grok's algorithmic idea: add a soft-Welch loss penalizing the
rules' firing coherence (`activation_decorrelation_penalty`, label-free — mean squared off-diagonal
of the rule-pattern correlation) to reduce cross-talk in the overpacked regime. Tested at `n_hard=24`,
`M ∈ {8,12}` (2–3× overpacked, where coherence is forced), sweeping the reg weight. Held-out, 3 seeds:

| M | reg | margin | acc | coh_μ |
|---|---|---|---|---|
| 8 | 0.0 | 1.83 | 0.75 | 0.289 |
| 8 | 2.0 | 1.86 | 0.75 | 0.279 |
| 12 | 0.0 | 2.73 | 0.83 | 0.242 |
| 12 | 2.0 | 2.74 | 0.81 | 0.230 |

The regularizer **works on its target** — it lowers the realized coherence `coh_μ` — but **margin and
accuracy are flat** (|Δmargin| ≤ 0.04, |Δacc| ≤ 0.02, within seed noise), even at 2–3× overpacking. So
the generator-side coherence regularizer is a **null**, exactly like the frame-side `frame_potential`
(§5b). The residual coherence the regularizer removes is not what limits the margin.

**This closes a symmetry.** Four theory-guided knobs have now been tested against plain end-to-end SGD:

| knob | side | result |
|---|---|---|
| `frame_reg` (decorrelate `U`) | frame / decode | null (§5b) |
| min-margin rule allocation/targeting | generator | null (§5c) |
| propose-score-select of frozen rules | generator | null (§5d) |
| `coh_reg` (decorrelate rule firing) | generator | null (§5g) |

None beats just training enough rules end-to-end. The consistent lever is **`M ≥ (effective rank of the
routing features)`**, with SGD packing them near the Welch floor on its own. This is also why the
coherence→margin *degradation* proof (Grok's Idea 1) is not worth formalizing: coherence is real and
Welch-floored (proved, `RoutingWelch`), but it is empirically **decoupled from the margin** — a tight
`γ ≤ γ₀(1−c·μ)` theorem would contradict the data. The honest theory boundary is: the *structural*
packing bounds are provable (and proved); the *margin consequence* is mild and not a clean law.

## 6. Open questions

- Does refining on **real fieldrun DLAs** raise retrievability above the frozen model, and at
  what measured fidelity cost (KL to host)?  (roadmap 2)
- Is the frame-potential term the right frame-side objective, or should it be a *targeted*
  coherence penalty (only on confusable pairs, `ρ > τ`)?
- Does `σ ∼ PR` hold under learned frames as it appears to under frozen ones (PIC O2)?
- Encode-side guard: add a co-firing-preservation term and measure the decode/encode trade
  explicitly (the polygram #113 axis), rather than hoping the split holds.
