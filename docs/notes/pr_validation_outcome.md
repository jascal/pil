# Does Participation Ratio Predict Anything? — Outcome: Claim A SUPPORTED, Claim B PARTIAL

Pre-reg: [`pr_validation_prereg.md`](./pr_validation_prereg.md) (SIGNED 2026-09-25). Primary data: the #126 source
dumps (Qwen2.5-0.5B-Instruct, 1280 positions, 40 prose + 40 code units). Replication: fieldrun
`certified_prune_step0` pil-dumps. Numbers: `results/pr_validation.{txt,json}`. Full-vocabulary reconstruction 1.000.

## Claim A — `r_eff` tracks how many blocks a position needs, beyond margin: **SUPPORTED** `[empirical]`
Partial Spearman `ρ(r_eff, k_suf | margin)`, 95% cluster-bootstrap CI over units:

| corpus | block level (49) | layer level (25) |
|---|---|---|
| prose | **+0.47** [+0.41, +0.52] | **+0.35** [+0.27, +0.44] |
| code | **+0.39** [+0.31, +0.46] | **+0.45** [+0.38, +0.51] |

All four cells clear the pre-registered bar (≥ 0.3, CI > 0). The plain correlations are +0.37 to +0.51, and margin
alone is −0.34 to −0.40. So `r_eff` carries information about the needed coalition that the margin does not.

**Centring is what carries it.** Raw PR `r_raw` barely correlates with `k_suf` (ρ = +0.05 to +0.10) in every cell.
The informative quantity is the architecture-fair, candidate-centred `r_eff`, which is exactly what `pic` §8
reports. The raw PR that `pic` §4.1's counterexamples target is the uninformative one.

**Replication** (candidate-restricted, not a gate): Qwen2.5-0.5B science (+0.30 / +0.49), 0.5B code (+0.43 / +0.36)
and Coder-0.5B (+0.46 / +0.38) all repeat the direction at ≥ 0.3. **Qwen2.5-7B is weaker:** +0.18 [−0.06, +0.41]
at block level and +0.27 [+0.04, +0.48] at layer level, with n = 80. The signal may thin with scale. One 7B dump
of 80 positions can't settle that.

## Claim B — the top `round(r_eff)` blocks decide the token: **PARTIAL** `[empirical]`
PR-head sufficiency is **0.60** on prose and **0.50** on code (code sits exactly at the 50% boundary). The scale is
off, too: median `r_eff` ≈ 13–14 blocks, while median `k_suf` = **3**. So `r_eff` is a useful **ordinal** signal
(more diffuse → needs more blocks) but **not a count**. Reading "`r_eff` ≈ 13" as "13 blocks are needed"
overstates by about 4×. Even the `r_eff` largest-by-`|cc|` blocks decide only about half the time, because
centring ranks blocks by how far they separate `t` from the *average* candidate, not from the runner-up.

## Controls and descriptive reads
- **Split control:** splitting one position's largest block into 4 raises `r_eff` by a median **×1.45–1.55**, with
  no change to the model. That is the non-invariance `pic` §4.1 states, now measured on real positions. `r_eff` is
  only meaningful at a fixed decomposition.
- **Merge control:** Claim A holds at both block and layer granularity, with no systematic direction between them.
- **Spectral rank** of the block matrix is low (median `erank` ≈ 2) and **unrelated or negatively related** to
  `r_eff` (ρ = −0.03 prose, −0.34 code). `r_eff` is not a geometric rank, as `pic` §7 says; the identification with
  `τ★` stays open.

## Predictions vs outcome
| prediction | outcome |
|---|---|
| `ρ(margin, k_suf)` strongly negative | ✓ moderate (−0.34 to −0.40) |
| partial ρ ≈ 0.1–0.3 → Claim A **FRAGILE** | ✗ — 0.35–0.47, **SUPPORTED** |
| split control ≈ +30–60% | ✓ (+45–55%) |
| Claim B **PARTIAL** (≈ 55–75%) | ✓ PARTIAL (50–60%, lower than predicted) |
| `erank` weakly related to `r_eff` | ✓ (or negatively, on code) |

## What this changes
- The review's worst-case critique stands: PR is not a coalition size, and raw PR tracks nothing here. But the
  **centred `r_eff` is a real, margin-independent ordinal predictor** of how many blocks a position needs on
  Qwen2.5-0.5B, at both granularities and across three 0.5B dumps. `pic` §4.1/§8 should say this precisely: an
  ordinal predictor of greedy sufficient-coalition size, not a count, not a rank, and fixed-decomposition only.
- Open: whether it survives at scale (7B is weaker), on LayerNorm models (Pythia), and against **causal** coalition
  size (ablation), which is out of scope here and needs fieldrun `--block-ablate` runs.

## Scope
Qwen2.5 (RMSNorm), 0.5B primary plus 0.5B/Coder/7B replication; direct-effect `k_suf` (greedy upper bound);
wikitext-2 prose and C/C++ code.
