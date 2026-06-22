# Real-model routing properties

*Paper-ready section: the real-model validation of the PIC/PIL routing theory. Companion to
`pil_learning_dynamics.md` §5h–§5i (the experiments) and the kernel-checked theory in i-orca
(`examples/tropical/{DecodeCapacity,RoutingRank}.thy`, `examples/superposition/RoutingWelch.thy`).*

## Setup: the seam

The synthetic PIL results are validated on real models through **`fieldrun --pil-dump`** (merged in
fieldrun #98). For each natural-text position it emits the per-block **direct-logit-attribution (DLA)
incidence matrix** `contrib[block][cand] = ⟨d_block, U_cand⟩` over the top-K candidate tokens, where the
blocks are the model's `nb = 2L + 1` additive residual components (one attention output and one MLP
output per layer, plus the embedding). In the PIC framing each block is a **rule**, its per-candidate
contribution the **incidence**, and the decode logit is their additive sum `logit_v = Σ_b contrib[b][v]`.
`pil/experiments/real_dla_analysis.py` and `real_dla_sweep.py` run the analyses.

The seam is **faithful**: across every model, corpus, and language tested, the emitted incidences
reconstruct the model's decode argmax 1.00 of the time and the recomputed margin equals the model's
exactly. The analysis is of the model's *own decode* (argmax vs runner-up), not of correctness.

Models: Qwen2.5 base 0.5B / 1.5B / 3B and 0.5B-Instruct / Coder-0.5B (`nb = 49 / 57 / 73`). Corpora:
English prose, code, rare-word jargon, shuffled prose, and Chinese / French / German / Spanish.

## Claim 1 — Routing is rank-concentrated, across scale, difficulty, and language

The decode is driven by a **small effective number of blocks** (source participation ratio
`(Σ_b c_b)²/Σ_b c_b²` over the per-block contributions), far below `nb`:

| axis | conditions | effective blocks | of `nb` |
|---|---|---|---|
| scale | 0.5B / 1.5B / 3B | 11 / 14 / 12 | 49 / 57 / 73 |
| language | en / zh / fr / de / es | 7–11 | 49 |
| difficulty | easy prose → shuffled | 11 → 9 | 49 |

Strikingly, the effective count is **~constant (~12) regardless of scale**, so as `nb` grows the rank
*slack* grows (PR/nb 0.22 → 0.16). This is the empirical face of `RoutingRank` (the rule adjustments
live in a `≤ min(M,d)`-dimensional subspace): real models use only a fraction of the available routing
dimensions, and that fraction shrinks with scale.

## Claim 2 — Real models route with substantial coherence slack (2–3× the Welch floor)

Treating each position's *decode-vs-runner-up direction in block space* as a routing feature, their
mutual coherence `μ` (mean off-diagonal |cosine|) sits **2–3× above the Welch floor**
`√((N−nb)/(nb(N−1)))` in every condition (μ ≈ 0.26–0.41). This is the one synthetic finding that does
**not** transfer: a task-specialized SGD packs its routing features near the Welch-optimal coherence
(~1× the floor), whereas a general pretrained model, on this imposed routing-feature basis, is **2–3×
looser**. The structural bound (`RoutingWelch`: `n > nb` forces coherence ≥ Welch) holds — the floor is
real — but pretrained models leave slack above it. The gap between general and task-specialized routing
is the clearest quantitative signal in the real-model data.

## Claim 3 — Difficulty modulates slack via fallback, increasing it

We tested whether the slack is an easy-corpus artifact by varying input difficulty (the model's own
decode margin as the proxy). The result is the **opposite** of the naive expectation:

| corpus | decode margin (↓ = harder) | effective blocks | μ |
|---|---|---|---|
| prose (easy) | 1.24 | 11 | 0.27 |
| shuffled prose (hard) | 0.48 | 9 | 0.41 |

Harder inputs produce **more** slack — fewer effective blocks and looser packing — because a confused
model **falls back to a few generic high-prior blocks**. (Code is a red herring: its rigid syntax makes
the model *more* confident, margin 2.56; rare-word jargon is mostly retrieval-style subword completion.)
So the rank-slack and looseness are **intrinsic and grow with difficulty**, not artifacts of predictable
text. Non-English mirrors this — slightly more concentrated and looser, consistent with marginally lower
fluency producing more fallback.

## Honest scope

The **structural** theory transfers cleanly: cell capacity (`DecodeCapacity`) is hugely slack in
practice, and generator rank (`RoutingRank`) is realized as the measured rank-concentration. The
**fine-grained interference** behavior differs between the synthetic and pretrained regimes — pretrained
routing is 2–3× looser than Welch-optimal — and we deliberately do **not** assert a margin-degradation
law (the coherence→margin link was measured mild, §5f–§5g; forcing it into a theorem would overclaim).
The result stands as three model-scale empirical properties of real routing, grounded in a faithful
decompilation seam, with the structural half certified in i-orca and the interference half left as a
measured observation.
