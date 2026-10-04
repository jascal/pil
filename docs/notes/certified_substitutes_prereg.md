# Certified compressed substitutes for GPT-2's last layer — pre-registration

**Status: written 2026-10-03, committed and pushed before any code for this study existed.**

**Context.**
- jascal/pil#134 / #135: the T5(a) **uniform** certificate covers 0% of contexts for every TPR fit. The
  **pairwise** certificate (i-orca #28, kernel-checked, an exact iff) covers exactly the contexts where the
  substitute keeps GPT-2's decision.
- jascal/lm-sae#234: GPT-2 small binds fillers to roles **conjunctively**, not as a systematic TPR.

**The question.** At matched parameter budgets, which compressed substitute of GPT-2's decode input is *provably*
faithful on the most held-out contexts? The headline prediction, from lm-sae#234: a **conjunctive pair-code**
substitute certifies more per parameter than a TPR or a rank-k additive one.

## 1. Tasks, model, site

GPT-2 small, frozen, float32. The site is the decode input `u = ln_f(h_L)` at the final position, with logits
`U u` (tied, no bias). **The decision is GPT-2's own argmax `t`.** Certificates are about faithfulness to the host,
not about the gold token.

- **LIST (list copy).** `Here is a list of words: w1, …, w5. Again: w1, …, wp,` with p ∈ {1,2,3,4}.
  - Nouns as in `tpr_projection_margin.py`: single-token, first 100.
  - Structure σ: list fillers at roles `(slot i, p)` (20 roles) and query fillers at roles `(query slot j, p)`
    (10 roles). 30 roles in all.
- **SVO (passive probe).** `The S V the O. The O was V by the`.
  - S ≠ O from the occupations list, single-token, capped at 40.
  - V from the verbs list, restricted to regular past forms (ending in `ed`, so the participle equals the past) and
    single-token, capped at 16.
  - Structure σ: S at subject, V at verb, O at object. 3 roles.

**Data.** Stimulus seed **21** (fresh; never used before). 12,000 LIST and 12,000 SVO contexts, split 60% train /
10% validation / 30% test by context, split seed 22. Every (filler, role) pair in the test split must also occur in
train; this is checked, and offending test contexts are dropped and counted. The pair-code family cannot represent an
unseen pair, and this study is about compression, not generalisation.

## 2. Substitute families

Every family maps the symbolic structure σ to a decode-input estimate `û(σ) ∈ ℝ^768`. Fillers and roles are
embedded per family.

| family | form | params |
|---|---|---|
| `additive-k` | `b + W·Σᵢ (e_F(fᵢ) + e_R(ρᵢ))`, `e ∈ ℝ^k`, `W ∈ ℝ^{768×k}` | `k(n_F + n_R) + 768k + 768` |
| `tpr-8` | `b + W·Σᵢ e_F(fᵢ) ⊗ e_R(ρᵢ)`, `d_F = 8`, `d_R = n_R` | `8 n_F + n_R² + 768·8·n_R + 768` |
| `tpr-32` | as `tpr-8` with `d_F = 32` | `32 n_F + n_R² + 768·32·n_R + 768` |
| `paircode-k` | `b + W·Σᵢ c(fᵢ, ρᵢ)`, a learned code `c ∈ ℝ^k` per seen (filler, role) pair | `k·|pairs seen| + 768k + 768` |

**Budgets** are pinned to the two TPR sizes, per task:
- **B₈** = params(`tpr-8`) and **B₃₂** = params(`tpr-32`).
- At each budget, `additive-k` and `paircode-k` take the largest integer `k` whose param count is ≤ the budget.
- So each budget compares three families: `additive`, `tpr`, `paircode`.

## 3. Objectives

Two objectives, every family, every budget. Adam, lr 3e-3, 3,000 steps, batch 512, fit seeds {0, 1, 2}.
- **`mse`**: `‖û − u‖²` (the baseline, as in earlier pil runs).
- **`cert`** (certificate-aware): `0.1·‖û − u‖² + H_K + 0.1·T`, where
  - `H_K = mean_{v ∈ top-32 rivals of the host logits} relu(1 − (L̂(t) − L̂(v)))`, a hinge on the substitute's
    pairwise slack over the host decision. Here `L̂ = U û`, and `L̂(t) − L̂(v)` is exactly the pairwise-certificate
    slack;
  - `T = relu(2·max_v |⟨u − û, U_v⟩| − m)`, with `m` the host margin: the uniform-certificate violation, taken over
    the full vocabulary.

## 4. Metrics (test split; certificate verdicts are Soufflé query results from `substitution_certificate.py`)

- **Primary: pairwise certified coverage.** The fraction of test contexts whose substitute is certified by
  `substitution_pairwise_iff`. This is exact and per-context, on contexts never used in training.
- **Secondary:** uniform certified coverage (`substitution_certified_max`), and hybrid with K = 32.
- **Score:** each (task, budget, family, objective) cell reports the seed mean and range. The **family score** in a
  cell is its better objective's seed-mean coverage. Both objectives are always reported; the best-of-two is fixed
  here, before any run.

## 5. Decision rules (fixed now)

- **H1 (headline): the pair-code wins.** In at least **3 of the 4** (task × budget) cells, `paircode`'s family
  score exceeds both `additive` and `tpr` by **≥ 0.03** (pairwise coverage).
  - **Another family wins** if it beats both others by ≥ 0.03 in ≥ 3 of 4 cells.
  - Otherwise: **no clear winner**.
- **H2 (the objective matters).** The `cert` objective raises `tpr` pairwise coverage over `mse` by **≥ 0.10** in at
  least 3 of 4 cells. If it fails, the hinge does not buy certification beyond what MSE gives.
- **H3 (uniform coverage, secondary).** The same H1 rule applied to uniform coverage, reported as secondary.
- **Best certified coverage per budget** is reported per task: the max family score, with its family and objective.

## 6. Not claimed

- Coverage is `empirical` and per context on this finite test set. A certificate is a proof only for the context
  it was issued on. Nothing extends to unseen contexts without a domain bound, which is T5(b) and `open`.
- Compressing GPT-2's decode input from symbolic structure is not a compression of GPT-2 on natural text.

## 7. Artifacts

- Script: `experiments/certified_substitutes.py`.
- Outcome: `docs/notes/certified_substitutes_outcome.md`. This file is not edited after the first run.
