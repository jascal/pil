# pil — Projective Incidence Learning

**Experimental prototyping repo for learning dynamics in Projective Incidence Calculus (PIC).**

This is the dedicated research codebase for exploring *how* the PIC structure can be
**learned and improved** in a transformer-like substrate: by massively parallel
construction of partial-evidence vectors (`d_j`), followed by selection and gating
through margins, the Gram kernel, and progressive refinement of the proposition frame.

It is the experimental companion to the [fieldrun](https://github.com/jascal/fieldrun)
PIC formalization (`fieldrun/PIC_PROPOSAL.md`) and the paper *"What a Transformer
Retrieves and What It Computes"* (J. Allan Scott, 2026).

> **Reproduction vs. improvement.** fieldrun *describes* a frozen model's decision logic.
> pil's goal is the opposite arrow: **change the geometry to make more of the logic
> retrievable** — higher margins, smaller sufficient support, a better-conditioned Gram —
> accepting reduced fidelity to any particular host model as the price. Throughout, every
> objective is labelled **decode-side** (margin / readout) or **frame-side** (intrinsic
> to `{U_v}`), because the two can trade against each other.

## Why this can work where frozen compression can't

The fieldrun program has mapped *where* the "computed" fraction (the forge tax) is movable:

- Under **frozen re-expression** — compress or factor an existing unembedding `U` — the
  composed tail is irreducible in every algebra tried (certified shortlist 0%, SVD
  pr-core ~67%, tropical winning-support ~65%; "no compact-faithful unembed", now
  kernel-confirmed in i-orca `examples/tropical/HeadTail.thy`). So "reduce the forge tax"
  by *compression* is a measured plateau, not a knob.
- Under **retraining the subspace** — change *which directions exist* — the floor moves
  (entangled-core: a retrained rank-8 bottleneck is lossless ~30× below the frozen floor;
  sae-forge "train the subspace, not the encoder"). PIL **updates `U_v`**, so it lives on
  this side of the boundary. Its improvement objective is *achievable in principle*,
  unlike the frozen-compression attempts — at an explicit, measured cost in host fidelity.

This is the precise reason pil is a separate repo from fieldrun, not a fork of its probes.

## Quickstart

pil depends only on `numpy` + `torch` (viz/`tqdm` are optional extras). Any torch-capable
environment works:

```bash
# from the repo root
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"          # numpy, torch, pytest, ruff, matplotlib, tqdm

pytest -q                        # 6 correctness tests (pin the starter bugs)
python experiments/synthetic_pil.py --steps 2000
python experiments/synthetic_pil.py --steps 2000 --visualize --save-dir runs/
```

Measured on the planted-frame synthetic (`dim=32`, `V=64`, `J=24`, 2000 steps, CPU/GPU):

| metric | value | reading |
|---|---|---|
| `retrievable_fraction` | **0.99** | fraction of positions past the margin threshold |
| `top1_acc` | **1.00** | argmax matches the planted target |
| `avg_margin` | **2.45** | mean worst-competitor margin (target 1.8) |
| `mean_support_pr` | **4.95** | effective # contributing sources (lower ⇒ more retrievable) |
| `frame_pot / welch_floor` | **1.61** | frame coherence vs the information-theoretic floor (1.0 = optimal) |

First ablation finding (descriptive, not a headline): the frame-side term decorrelates the
Gram monotonically (`fp/welch` 1.63 → 1.54 as `frame_reg` 0 → 0.2) at **zero** decode-side
cost on this synthetic — but the effect is small because the regime is too easy.

The **hard synthetic** (`experiments/hard_synthetic.py`: 8× over-complete, planted synonym
clusters, `|cos|=0.82`) was built to stress it — and produced a clean **negative result**.
Synonymy compressed margins exactly as intended (`top1≈0.99` but `retr≈0.40`, strongest
competitor a synonym 88% of the time), yet sweeping `frame_reg` moves the frame geometry
(`fp/welch` 1.35→1.09) with the decode outcome **flat** (`|Δretr| ≤ 0.01` across the entire
synonymy×noise plane, in-sample). The frame regularizer is **redundant** where the margin term
already shapes the frame and **impotent** where the bottleneck is source distinguishability:
when synonyms have near-identical sources `d_j`, the margin is source-SNR-bound and no
readout-frame objective can manufacture missing signal. So the lever for the confusable /
forge-tax regime is the **generative proposer**, not frame regularization (see
`docs/notes/pil_learning_dynamics.md` §5b).

That prediction is then **confirmed** in the compositional regime (`experiments/compositional_pil.py`,
§5c): when synonyms are XOR-coded (non-linearly separable — PIC-T3 weighted-threshold), a frame-only
linear readout is pinned at chance (hard-cluster acc **0.454**, held-out) while **generated rules lift
it to 0.93**. Two honest halves: (1) *generation is the lever* — rules recover what no frame objective
could; (2) *but min-margin targeting barely beats random rules* (0.934 vs 0.911) — SGD already
allocates rules to the at-risk facets, so tropical targeting buys initialization, not allocation. The
held-out control (`signal=False` → all arms at chance) confirms generation can't manufacture absent
signal.

## Repository structure

```
pil/
  pil/
    geometry.py      incidences, Gram, margins, frame potential (Welch-floored), PR, power diagram
    learner.py       ProjectiveIncidenceLearner: propose -> gate -> refine; labelled losses
    synthetic.py     hard problems: over-complete + synonym clusters; XOR-coded compositional; diagnostics
    proposer.py      RuleBank (the generative step) + min-margin-targeted rule seeding
    fieldrun_io.py   integration contract for seeding from real fieldrun probe dumps
    viz.py           optional matplotlib helpers (training curve, Gram heatmap)
  experiments/
    synthetic_pil.py    runnable planted-frame demo of the generate/gate/refine loop
    hard_synthetic.py   frame_reg sweep over the over-complete/synonym regime (the decode/frame trade)
    compositional_pil.py  frame vs untargeted vs targeted generation on XOR-coded synonyms (held-out)
  tests/             correctness tests pinning the starter bugs + the generators
  docs/notes/        design notes (learning dynamics; decode/frame split)
```

## Key concepts (naming matches `fieldrun/PIC_PROPOSAL.md` §2)

| symbol | code | side | meaning |
|---|---|---|---|
| `d_j ∈ H` | `sources` | — | parallel partial-evidence vectors (circuits) |
| `U_v ∈ H` | `model.U` | frame | proposition directions (unembedding rows) |
| `c_j^v = ⟨d_j,U_v⟩` | `incidences` | decode | direct logit attribution (DLA) |
| `L_v = Σ_j c_j^v` | `logits_from_incidences` | decode | aggregated logit |
| `G_vw = ⟨U_v,U_w⟩` | `gram_matrix` / `cosine_gram` | frame | non-truth-functionality kernel (PIC T2) |
| margin | `margin_to_worst` | decode | `L_t − max_{v≠t} L_v` (Laguerre facet distance) |
| frame potential | `frame_potential` | frame | mean sq. off-diagonal cosine; floor = Welch bound |
| PR | `participation_ratio` | — | effective support size (PIC T4 diffuseness diagnostic) |

## Roadmap

1. **Synthetic floor (done).** Planted-frame recovery + the hard over-complete/synonym
   regime; the generate→gate→refine loop; decode-vs-frame metrics. ✔ runs, 7 tests green.
   Finding: frame regularization is decoupled from the decode; the bottleneck is source-side.
2. **Smarter proposers (validated as the lever; `RuleBank` shipped).** Generated rules recover
   non-linear (composed) structure no frame objective can (§5c). Open: targeting that *beats* SGD
   allocation (only matters in the weak-gradient / very-low-budget regime), and richer proposers —
   SAE features (polygram / sae-forge), induction/number-mover templates — that distinguish
   confusable propositions on real data rather than synthetic XOR.
3. **Seed from fieldrun.** Implement a `--pil-dump` emitter on the fieldrun side that writes
   the `pil.fieldrun_io` contract (real DLA sources `d_j`, frame `U`, per-position targets),
   then refine on real residuals. Question: does retrievability improve over the frozen model
   at a stated fidelity cost?  Add a train/holdout split here (the synthetics are in-sample).
4. **Targeted geometry.** Domain/token-class-specific frames; measure forge-tax reduction
   per class against the cross-substrate baselines (it differs bio vs econ vs LM).
5. **Export.** Retrievable fragment → semiring-Datalog (fieldrun's LOGIC_EXPORT path);
   COMPOSED positions flagged as "no compact formula" (PIC O4).

## Relationship to other jascal projects

- **fieldrun** — the PIC theory + the probes that supply real `d_j`, `U`, margins, PR.
- **polygram / sae-forge / {bio,econ,sm}-sae** — sources of structured partial solutions
  (SAE features, MPS dictionaries) for the generative step.
- **i-orca** — formal backbone; the `examples/tropical` head/tail decode certificate is the
  kernel-checked statement of the boundary pil is trying to *move*, not merely re-express.

## Status

Greenfield prototype (v0.0.1). Synthetic loop runs and is tested; the fieldrun seam is a
documented contract awaiting the emitter. Nothing here claims a forge-tax reduction on a real
model yet — that is roadmap item 2, stated as an open question, not a result.
