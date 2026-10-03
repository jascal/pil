# TPR projection as margin widening — outcome

**Pre-registration:** [`tpr_projection_margin_prereg.md`](tpr_projection_margin_prereg.md), written 2026-10-03
before the dump. **Run:** `experiments/tpr_projection_margin.py`.
- Dump: GPT-2 small, 12,000 contexts; 8,000 train / 4,000 test.
- Evaluation on GPU with lm-sae's CUDA torch venv. pil code is imported from source and is unchanged.
- `runs/tpr_margin/summary.json` is git-ignored like all pil runs; the numbers below are copied from it.
- Tag: `empirical`.

**Verdict: Q1 fails, Q2 fails, Q3 half-holds.** At GPT-2 small's decode input, the paper's "the fit is more decodable
than the host" effect does **not** appear. A TPR fitted by MSE (DISCOVER's own objective) loses the decision.

## Results (test, n = 4,000)

| arm | gold acc | agrees with real | KL from real | mean margin | retrievable γ=0 / 1 / 2 |
|---|---:|---:|---:|---:|---|
| `real` | 0.998 | 1.000 | 0.000 | 3.80 | 0.998 / **0.979** / 0.914 |
| `oracle` (d_F = 8, MSE) | 0.054 | 0.054 | 4.27 | −2.13 | 0.054 / 0.008 / 0.000 |
| `cleanup` (d_F = 8) | 0.077 | 0.077 | 4.28 | −2.29 | 0.078 / 0.019 / 0.004 |
| `proj-tpr` (rank 240) | 0.996 | 0.996 | 0.077 | 3.39 | 0.996 / 0.971 / 0.867 |
| `proj-pca` (rank 240) | 0.998 | 0.998 | 0.003 | 3.84 | 0.998 / 0.982 / 0.918 |
| `oracle` (d_F = 32, MSE), secondary | 0.840 | 0.839 | 1.39 | 1.17 | 0.841 / 0.562 / 0.243 |
| *post-hoc* TPR (d_F = 8) trained on KL | 0.988 | 0.986 | 0.48 | 3.07 | — / 0.947 / — |

Further measurements:
- **Fit quality:** the d_F=8 oracle has FVU 0.25 and still decides correctly only 5% of the time.
- **Cleanup unbinding:** per-slot list accuracy is 0.42 and query-length accuracy 0.999, but σ is recovered exactly in
  only 0.6% of contexts.
- **Span diagnostic:** only 44% of `‖U_gold − U_rival‖²` (top-5 rivals) lies inside `span(W)`. So T6(a)'s hypothesis, that the subspace contains every readout difference, is **false** at this site.

## Decision rules

- **Q1 — limitivism as margin: FAILS.** The oracle *lowers* the γ=1 retrievable fraction, from 0.979 to 0.008 at
  d_F=8 and to 0.562 at d_F=32.
- **Q2 — snapping is deployable: FAILS.** Cleanup is no better than the oracle. At this site the list words are mostly
  *not* linearly recoverable from `u` (42% per slot), so σ̂ is nearly always wrong.
- **Q3 — projection is not the lever: half.**
  - Holds: `proj-tpr` is within 0.011 of the rank-matched `proj-pca` (0.971 vs 0.982). The TPR subspace is not
    special beyond its rank, and neither projection widens margins over `real`. This is **not** a test of T6(a): its
    hypothesis fails here (see the span diagnostic), and the margin loss is what removing out-of-span components gives.
  - Fails: the pre-registered second clause ("neither beats `oracle`"). Both projections beat the MSE oracle by a
    wide margin.

## Reading

1. **MSE is the wrong objective at a decode site.** The decision lives in a few directions (`U_gold − U_rival`).
   Squared error spends capacity on the high-variance rest. That is how FVU 0.25 coexists with 5% accuracy.
2. **The TPR form itself is not the obstacle.** The post-hoc KL-trained arm (same d_F=8 form) carries the decision at
   0.988. But its margins (3.07) are *below* the host's (3.80). No widening: here the bound structure is a lossy
   *re-expression* of the host's decode, not a cleaner one.
3. **The site matters.** The paper's denoising gain (0.71 → 0.96) was at mid-layer sentence-final periods, read by a
   learned nonlinear decoder. Here, at the final position just before a copy, `u` mostly encodes the *next token*,
   not the whole list (42% per slot). The structure the role scheme assumes is largely absent from `u`. That is a
   finding about the site, not a refutation of the paper.
4. **Relevance to pil's lever.** This joins the frame-regularizer result (`pil_learning_dynamics.md` §5b): a
   readout-side re-expression of a frozen host does not manufacture margin. Q3 is the clean part. Linear projection
   onto any rank-240 subspace leaves the decision nearly intact, but with *less* margin. Widening would need a change
   to the generator, not the readout.

## Status of the theory this note cites

T5 and T6 (i-orca `examples/pic_binding/PROPOSAL.md`, jascal/i-orca#26) are **open** conjectures with no `.thy`
lemma. The pre-registration's phrase "last-layer substitution is certified by the margin theorem" is kept as written,
since it is a dated document, but it overstates. Nothing in this note is certified by T5.
- The MSE oracle's collapse (gold accuracy 0.054, mean margin −2.13) is an empirical instance of what T5(a) would
  price: a fit whose error exceeds the margin loses the decision.
- The post-hoc KL arm's agreement with the host (0.986) at reduced margin (3.07 vs 3.80) is the same kind of
  instance.

Neither is a citation of a theorem.

## Caveats

- One model, one task, one site. Supervised pair roles; MSE without the paper's L2,1 regularizer.
- GPU nondeterminism moves pre-registered arms in the third decimal between reruns (e.g. oracle d_F=8 retrievable
  0.007 vs 0.008). No verdict depends on it.
- The KL arm is post hoc and is a fidelity fit, not DISCOVER's representational test. It answers a capacity question
  only.

## Not run (would be a new pre-registration)

The paper's own site: mid-layer period encodings with a nonlinear unpacking decoder. Its lm-sae counterpart is
`lm-sae/docs/TPR_VS_SAE.md`.
