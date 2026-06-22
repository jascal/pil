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

## 6. Open questions

- Does refining on **real fieldrun DLAs** raise retrievability above the frozen model, and at
  what measured fidelity cost (KL to host)?  (roadmap 2)
- Is the frame-potential term the right frame-side objective, or should it be a *targeted*
  coherence penalty (only on confusable pairs, `ρ > τ`)?
- Does `σ ∼ PR` hold under learned frames as it appears to under frozen ones (PIC O2)?
- Encode-side guard: add a co-firing-preservation term and measure the decode/encode trade
  explicitly (the polygram #113 axis), rather than hoping the split holds.
