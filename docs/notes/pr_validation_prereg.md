# Does Participation Ratio Predict Anything? — Pre-Registration

**Status: DRAFT — awaiting signature. Decision rules are fixed BEFORE numbers. None of the quantities below has been
computed on these dumps.** Item 4 of the external PIC review: *"establish whether participation predicts anything
beyond attribution."*

## 1. The question
`pic` §4.1 now says PR is **not** a sufficient-coalition size (explicit counterexamples) and **not** invariant under
splitting a source. Those are worst-case statements. `pic` §8 still reports `r_eff` (centred PR) as its main
empirical quantity. So the practical question is whether, **on real positions**, `r_eff` tracks anything besides
itself:
- **Claim A:** does `r_eff` track how many blocks a position actually needs, *beyond what the margin already tells
  you*?
- **Claim B:** does PR's face-value reading hold, i.e. do the top `round(r_eff)` blocks by contribution decide the
  token?

## 2. Quantities (per position)
Blocks `j`, target `t` = the model's decision, incidences `c_j(v) = ⟨d̃_j, U_v⟩` (exact logit contributions).
- **`r_eff`** — the `pic` §8 / `tau_star_entropy.py` definition: centre each block's incidence across the candidate
  set, `cc_j(v) = c_j(v) − mean_{v∈K} c_j(v)`, then `r_eff = PR(|cc_j(t)|)`. Candidate set `K` = the top-24 tokens
  by full-model logit (fieldrun's default `--kcand`). `r_raw = PR(|c_j(t)|)` is reported too.
- **`k_suf`** — greedy sufficient-coalition size, the `min_blocks_to_argmax` definition: order blocks by
  `c_j(t) − c_j(v₂)` (`v₂` = the full model's runner-up), and take the smallest prefix whose partial sum has `t`
  as argmax. **Primary:** argmax over the **full vocabulary**. **Replication:** over the dump's candidates.
  `k_suf` is an upper bound on the minimal sufficient coalition and a direct-effect quantity (the `pic` §3
  caveat).
- **`margin`** — full-model `L(t) − max_{v≠t} L(v)`, the covariate every claim is measured *beyond*.
- **`erank`** — spectral effective rank of the block matrix `D` (`nb × d`, folded coordinates):
  `(Σσ²)² / Σσ⁴`. Descriptive only.
- **PR-head sufficiency** — whether the top `round(r_eff)` blocks by `|cc_j(t)|` have `t` as their full-vocabulary
  argmax (the Claim B test).

## 3. PINS
- **PIN A — primary data:** the #126 source dumps (Qwen2.5-0.5B-Instruct, fieldrun `d17438c`), regrouped by
  corpus. **PROSE** = all 40 prose units (CAL-prose + EVAL-PROSE, 640 positions). **CODE** = all 40 code units
  (640). No parameter is fitted, so no held-out split is needed. Clusters = units (16 positions each).
- **PIN B — replication:** fieldrun `experiments/certified_prune_step0/*.jsonl` `--pil-dump`s: Qwen2.5-0.5B
  science and code, Coder-0.5B science, Qwen2.5-7B science. These are candidate-restricted (16 candidates), so
  `k_suf` is over candidates, with no `erank` and no PR-head test. Single-text dumps, so plain bootstrap.
- **PIN C — granularity (merge control):** block level (`nb = 49`), and layer level (25 sources: embed plus each
  layer's attn + mlp summed). Every Claim A statistic is computed at both.
- **PIN D — split control (descriptive):** split each position's largest-`|cc_j(t)|` block into 4 equal parts and
  report the median ratio `r_eff(split)/r_eff`. `k_suf` in original-block units is unchanged by construction.
  This quantifies PR's non-invariance on real data and is not a decision variable.
- **PIN E — statistics:** Claim A = **partial Spearman** `ρ(r_eff, k_suf | margin)`: Pearson correlation of the
  residuals after regressing the ranks of `r_eff` and of `k_suf` on the rank of `margin`. 95% CIs from 2000
  **cluster-bootstrap** resamples over units (seed 0) on the primary data, and a plain bootstrap on the
  replication data. Plain Spearman `ρ(r_eff, k_suf)` and `ρ(margin, k_suf)` are reported alongside.

## 4. Decision rules (FIXED BEFORE NUMBERS)
**Claim A — `r_eff` tracks coalition size beyond margin (primary data):**
- **SUPPORTED:** partial ρ ≥ 0.3 with CI lower bound > 0, on **both** PROSE and CODE, at **both** granularities.
- **FRAGILE:** that holds for at least one (corpus, granularity) cell but not all four.
- **NOT SUPPORTED:** it holds in no cell.

The replication records whether the primary verdict's sign and magnitude (≥ 0.3) repeat on each step0 dump,
including 7B. It is not a gate.

**Claim B — face value (primary data, block level):**
- **HOLDS:** PR-head sufficiency ≥ 80% on both corpora.
- **PARTIAL:** 50–80% on either corpus.
- **FAILS:** < 50% on both.

The two claims are reported independently. A `pic` §4.1/§8 update follows either way.

## 5. Architect's predictions (recorded, not decision variables)
- `ρ(margin, k_suf)` is strongly negative: small-margin positions need more blocks.
- Plain `ρ(r_eff, k_suf)` is positive, but much of it is margin. Partial ρ is around 0.1–0.3 → **Claim A
  FRAGILE**, likelier at block level than layer level.
- The split control moves `r_eff` by a median of about +30–60%.
- Claim B **PARTIAL** (roughly 55–75%). `r_eff` is dominated by a few large late blocks, which often do decide,
  but centring rewards blocks that separate `t` from the *mean* candidate, not from the runner-up.
- `erank` correlates weakly with `r_eff`.

## 6. Controls
- Full-vocabulary reconstruction on the primary dumps (the #126 control: 1.000).
- The split and merge controls (PINs C–D).
- The margin covariate on every Claim A statistic.
- Cluster bootstrap over units.
- pil `ruff` + `pytest`, including unit tests of `k_suf`, `r_eff` and the partial Spearman on constructed
  examples, among them the pic §4.1 counterexample `c_a = (1,2,0)`, `c_b = (0,−2,½)`.

## 7. Risks / open
- **Direct effect only.** `k_suf` counts blocks in the DLA sum, not causal necessity. Intervention (ablation) effects
  are the review's other suggested target and are **out of scope** here: they need new fieldrun `--block-ablate`
  runs.
- **Greedy is an upper bound.** A true minimal sufficient coalition can be smaller. If PR tracks the greedy order's
  quirks rather than true minimality, this test cannot tell.
- **One architecture family** (Qwen2.5, RMSNorm). The Pythia/LayerNorm contrast in `pic` §8 is not re-tested here.
- **Candidate-set dependence.** `r_eff` is centred over the top 24 here and the top 16 in the replication. That
  difference is reported, not corrected.

## 8. Scope fences
Measurement only, on existing dumps. No new fieldrun runs, no retraining.
