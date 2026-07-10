# J-lens DLA correction — λ-sweep results

Companion to `experiments/jlens_correction_sweep.py`. Records the empirical outcome of the fieldrun→pil J-lens seam:
does routing each per-block DLA contribution through its layer's averaged causal Jacobian `J_l` sharpen pil's mid-stack
read of the residual, versus the plain logit-lens?

**Tag: `empirical` throughout.** `J_l` is a first-order, context-averaged approximation, not a certificate. These are
small-n directional measurements (64–256 positions, one corpus), not a proof of anything.

---

## The question

The plain DLA/logit-lens incidence `c_b^v = ⟨d̃_b, U_v⟩` scores a block's contribution as if it reached the output
UNCHANGED (identity downstream). The J-lens (Anthropic, *"Verbalizable Representations as Global Workspace"*,
transformer-circuits.pub/2026/workspace) instead routes it through `J_l = E[∂h_final/∂h_l]` first, scoring the block's
TOTAL (direct + downstream) linearised effect. `jcorrect_sources` applies this per block; this sweep measures whether it
helps, as a function of the shrinkage λ (`J' = (1−λ)I + λJ`).

## The pipeline (all merged)

| repo | PR | provides |
|---|---|---|
| fieldrun | #124 | `{J_l}` fit (Hutchinson JVP) + `--jlens-export` + the `capture_point` pin |
| fieldrun | #125 | `--tensors-export` → the model unembedding `U` and final-norm gain `γ` |
| pil | #49 | `jcorrect_sources` — the per-block J-correction |
| pil | #50 | this sweep harness (recon / margin / resolve metrics) |
| pil | #51 | the exact **LayerNorm** fold `diag(γ)·P·J·diag(1/γ)`, `P = I − 11ᵀ/d` |

## Method

- **`{J_l}`**: fit on `experiments/jlens/fit_corpus.txt` (300 prompts), `probes=5 max_seq=24 max_src=4 seed=1`, exported
  to `.npz` (`[n_layer, d, d]` f32 + `fitted[n_layer]`).
- **Source-dump**: `fieldrun --source-dump` (per-position raw per-block DLA vectors `d̃_b` + labels + top-`kcand=24`
  candidates), `n=64` unless noted; recon-argmax 1.00 on every dump (the DLA is faithful).
- **Sweep**: for each λ, `Dc = jcorrect_sources(...)` → candidate-restricted incidences `⟨J·d̃_b, U_cands⟩`, reporting
  - `recon` — fraction where the summed read argmaxes to the model decode (**λ=0 ⇒ ≈1.0**, the sanity invariant; a drop
    at λ>0 is expected — the correction is not meant to preserve the decode);
  - `margin` — decoded token vs strongest competitor (higher = cleaner);
  - `resolve` — fraction-of-depth at which the cumulative-by-layer read first locks to the decode
    (**lower = "resolves earlier"** — the paper's claim, and the metric the J-lens is supposed to improve).
- **Fold**: exact per `norm_type` — RMSNorm `diag(γ)J diag(1/γ)` (rope); LayerNorm `diag(γ)P J diag(1/γ)` (neox), the
  `ln_f` bias and `inv_std` being logit-inert.

`margin` degrades across the board — this is expected and *not* a failure: the summed corrected read `Σ_b J·d̃_b ≠ r`
by construction, so margin-vs-decode is the wrong success metric for a per-block correction. **`resolve` is the signal.**

---

## Result — a depth threshold at ~24 layers

`Δresolve` = corrected `resolve` − baseline (λ=0); **negative = resolves EARLIER = the win**. Best over the swept λ:

| model | arch / norm | layers | d | Δresolve @λ0.25 | best Δresolve (λ) | verdict |
|---|---|---|---|---|---|---|
| pythia-14m | neox / LayerNorm | 6 | 128 | −0.003 | −0.003 (0.1) | **null** |
| pythia-70m | neox / LayerNorm | 6 | 512 | −0.006 | −0.012 (0.1) | **null** |
| pythia-160m | neox / LayerNorm | 12 | 768 | +0.006 | — (none <0) | **null** |
| **pythia-410m** | neox / LayerNorm | **24** | 1024 | **−0.042** | **−0.074 (0.5)** | **WIN** |
| **Qwen2.5-0.5B** | rope / RMSNorm | **24** | 896 | **−0.020** | **−0.030 (0.5)** | **win** |
| Qwen2.5-1.5B | rope / RMSNorm | 28 | 1536 | +0.007 | — | inconclusive † |

**The win appears at 24 layers and is absent at ≤12 — in both architectures.** It's a ~7–8%-of-depth (≈2-layer)
earlier resolve at λ≈0.25–0.5, matching the paper's "middle-third of many layers" prediction: the effect needs depth.

† Qwen2.5-1.5B is **not a clean point** — see below.

### The two winners, in full

**pythia-410m (24L neox, exact LayerNorm P):**

```
n=64                              n=256 (4 contexts)
 lam  resolve  Δresolve            lam  resolve  Δresolve
0.00   0.544   +0.000             0.00   0.560   +0.000
0.10   0.506   −0.038             0.10   0.545   −0.015
0.25   0.494   −0.050             0.25   0.518   −0.042
0.50   0.461   −0.083  ← best     0.50   0.485   −0.074  ← best
0.75   0.524   −0.020             0.75   0.529   −0.031
1.00   0.630   +0.086             1.00   0.621   +0.061
```

Robust to 4× more data across 4 different contexts — not 64-position noise.

**Qwen2.5-0.5B (24L rope, exact RMSNorm γ), n=64:** `resolve` 0.728 → 0.708 (λ0.25) → **0.698 (λ0.5, Δ−0.030)** → 0.746
(λ0.75). The same λ≈0.25–0.5 sweet spot fieldrun's own `--jlens-eval` (whole-`h_l`-through-`J` read) found on this model.

---

## The γ-conjugation is load-bearing (within-model ablation)

On Qwen2.5-0.5B, the win **requires** the γ-fold. Same model, same positions, exact-γ vs direct-`J`:

| λ | exact γ (Δresolve) | direct J, no γ (Δresolve) |
|---|---|---|
| 0.25 | **−0.020** | +0.014 |
| 0.50 | **−0.030** | +0.026 |
| 0.75 | +0.018 | +0.111 |

Direct `J` (γ≈const) never resolves earlier. This isolates the folded-basis conjugation `diag(γ)J diag(1/γ)` as
necessary — a clean causal claim (one variable), not a cross-model confound.

## Confounds ruled out

1. **LayerNorm fold approximation (#51).** Hypothesis: Pythia (neox) nulled only because its γ-fold was approximate.
   Re-running the ladder with the **exact** mean-centering `P` moved `Δresolve` by **<0.01** and still showed no win —
   so the fold approximation was *not* the cause. (E.g. pythia-160m @λ0.25: +0.006 approx → +0.006 exact.)
2. **Depth vs architecture.** Deep-neox (410m) wins *as well as / better than* deep-rope (Qwen-0.5B), so the driver is
   **depth/scale, not architecture** — neox is fine once deep enough (with the exact `P` fold).
3. **Corroboration.** fieldrun's own `--jlens-eval` (a different read — whole `h_l` through `J`, not per-block) agrees
   on the small Pythia rungs: 70m resolve flat then worse; 160m a marginal earlier-resolve but *more* across-depth
   flips. Two independent readers, same null on the shallow models.

## Inconclusive: Qwen2.5-1.5B (a compute wall)

The 28L point that would extend the curve above the threshold **could not be cleanly measured here**. The full
300-prompt fit for Qwen2.5-1.5B runs at ~8.4 min/prompt (d=1536 crosses a cache threshold) → **~42 h**. Swept off the
~80-prompt **checkpoint** `J`, it shows no win (Δresolve `+0.005 / +0.007 / +0.019` at λ 0.1 / 0.25 / 0.5) — but the
shape (recon collapsing fast, no improvement even at λ=0.1) is the **noise-dominated / under-converged-`J`** signature,
not the deep-win pattern. So this neither confirms nor refutes the depth trend; a clean 28L point needs the full fit.

## Honest bounds

- **Small-n** — 64 positions (256 for the 410m confirmation), one corpus. Directional, not definitive.
- **Depth is confounded with width** in the Pythia ladder (`d`: 128→512→768→1024), so the claim is "**scale**, threshold
  ~24L / ~400M params," not depth-isolated-from-width.
- `resolve` is the signal; `margin` degrades by construction (see Method).
- `empirical` — a probe, never a certificate.

## Reproduce

```bash
# fieldrun (per model): fit {J_l}, export it, export U+γ, dump per-block DLA
fieldrun --bundle <M> --text x --recursion-explain --jlens-fit \
    --jlens-corpus fit_corpus.txt --jlens-out <M>.jlens \
    --jlens-probes 5 --jlens-max-seq 24 --jlens-max-src 4 --jlens-layers all --jlens-seed 1
fieldrun --jlens-export <M>.npz --jlens-in <M>.jlens
fieldrun --bundle <M> --text x --recursion-explain --tensors-export <M>.tensors.npz
fieldrun --bundle <M> --recursion-explain --source-dump <M>.source.jsonl --text "<ctx>" --n 64 --kcand 24

# pil: sweep (fold auto-detected from the tensors meta's norm_type)
python experiments/jlens_correction_sweep.py <M>.source.jsonl \
    --jlens <M>.npz --tensors <M>.tensors.npz --lams 0,0.1,0.25,0.5,0.75,1.0
```

For >64 positions, concatenate several short-context source-dumps — the dump is O(L²) in context length, so many short
contexts are far cheaper than one long one.
