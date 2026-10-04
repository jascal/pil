# Certified compressed substitutes — outcome

**Pre-registration:** [`certified_substitutes_prereg.md`](certified_substitutes_prereg.md).
- Pushed before any code as `0642ecb` (author time 20:17:32). It was rebased onto the pairwise-certificate branch as
  `2cf9d6c`, with content unchanged.
- Script committed before the real run as `9b94d3e` (20:22:48).
- After rebasing onto `main` for the PR, these commits are `828cc96` (pre-registration) and `8c8ff94` (script).
  Author times and content are unchanged. `--smoke` (stimulus seed 999, 600 contexts, 30 steps)
  was used only for bug checks.
- Run: `experiments/certified_substitutes.py`, GPT-2 small on an RTX 5050, about 105 minutes.
- Verdicts are computed by the script's pre-registered rules (`verdicts()`). Certificate verdicts are Soufflé query
  results from `substitution_certificate.py`.
- Tags: coverage is `empirical`. Each certificate is `proved` for its own context, by i-orca
  `substitution_pairwise_iff` / `substitution_certified_max` / `substitution_certified_hybrid`.

## LIST

Fillers 100, roles 30, seen pairs 3000, train 7200, test 3600 (dropped for an unseen pair: 0). Budgets: B₈ = 186,788, B₃₂ = 742,148 params.

| budget | family | size | params | objective | **pairwise** (mean [min, max]) | uniform | hybrid K=32 | agree |
|---|---|---:|---:|---|---|---:|---:|---:|
| B8 | additive | 207 | 186,654 | mse | 0.010 [0.008, 0.012] | 0.000 | 0.000 | 0.010 |
| B8 | additive | 207 | 186,654 | cert | 0.019 [0.015, 0.024] | 0.000 | 0.000 | 0.019 |
| B8 | tpr | 8 | 186,788 | mse | 0.034 [0.024, 0.044] | 0.000 | 0.000 | 0.034 |
| B8 | tpr | 8 | 186,788 | cert | 0.637 [0.628, 0.642] | 0.000 | 0.000 | 0.637 |
| B8 | paircode | 49 | 185,400 | mse | 0.005 [0.003, 0.007] | 0.000 | 0.000 | 0.005 |
| B8 | paircode | 49 | 185,400 | cert | 0.428 [0.424, 0.435] | 0.000 | 0.000 | 0.428 |
| B32 | additive | 825 | 741,618 | mse | 0.022 [0.020, 0.023] | 0.000 | 0.000 | 0.022 |
| B32 | additive | 825 | 741,618 | cert | 0.030 [0.027, 0.032] | 0.000 | 0.000 | 0.030 |
| B32 | tpr | 32 | 742,148 | mse | 0.711 [0.699, 0.720] | 0.000 | 0.000 | 0.711 |
| B32 | tpr | 32 | 742,148 | cert | 0.961 [0.958, 0.967] | 0.000 | 0.000 | 0.961 |
| B32 | paircode | 196 | 739,296 | mse | 0.094 [0.083, 0.104] | 0.000 | 0.000 | 0.094 |
| B32 | paircode | 196 | 739,296 | cert | 0.859 [0.842, 0.871] | 0.000 | 0.000 | 0.859 |

## SVO

Fillers 56, roles 3, seen pairs 96, train 7200, test 3600 (dropped for an unseen pair: 0). Budgets: B₈ = 19,657, B₃₂ = 76,297 params.

| budget | family | size | params | objective | **pairwise** (mean [min, max]) | uniform | hybrid K=32 | agree |
|---|---|---:|---:|---|---|---:|---:|---:|
| B8 | additive | 22 | 18,962 | mse | 0.152 [0.146, 0.166] | 0.000 | 0.000 | 0.152 |
| B8 | additive | 22 | 18,962 | cert | 0.552 [0.547, 0.564] | 0.000 | 0.000 | 0.552 |
| B8 | tpr | 8 | 19,657 | mse | 0.053 [0.020, 0.072] | 0.000 | 0.000 | 0.053 |
| B8 | tpr | 8 | 19,657 | cert | 0.674 [0.648, 0.704] | 0.000 | 0.000 | 0.674 |
| B8 | paircode | 21 | 18,912 | mse | 0.039 [0.033, 0.043] | 0.000 | 0.000 | 0.039 |
| B8 | paircode | 21 | 18,912 | cert | 0.798 [0.783, 0.808] | 0.000 | 0.000 | 0.798 |
| B32 | additive | 91 | 76,025 | mse | 0.557 [0.552, 0.562] | 0.000 | 0.000 | 0.557 |
| B32 | additive | 91 | 76,025 | cert | 0.576 [0.574, 0.580] | 0.000 | 0.000 | 0.576 |
| B32 | tpr | 32 | 76,297 | mse | 0.775 [0.740, 0.792] | 0.001 | 0.000 | 0.775 |
| B32 | tpr | 32 | 76,297 | cert | 0.875 [0.871, 0.882] | 0.000 | 0.000 | 0.875 |
| B32 | paircode | 87 | 75,936 | mse | 0.848 [0.843, 0.853] | 0.006 | 0.000 | 0.848 |
| B32 | paircode | 87 | 75,936 | cert | 0.880 [0.877, 0.881] | 0.004 | 0.000 | 0.880 |

## Verdicts (pre-registered rules)

Family score = the better objective's seed-mean pairwise coverage.

| cell | additive | TPR | pair-code |
|---|---:|---:|---:|
| LIST/B8 | 0.019 | 0.637 | 0.428 |
| LIST/B32 | 0.030 | 0.961 | 0.859 |
| SVO/B8 | 0.552 | 0.674 | 0.798 |
| SVO/B32 | 0.576 | 0.875 | 0.880 |

- **H1 — the pair-code wins: FAILS.** Wins by ≥ 0.03 over both others: {'additive': 0, 'tpr': 2, 'paircode': 1}. The verdict is **no clear winner**.
  - TPR wins both LIST cells.
  - The pair-code wins SVO at B₈.
  - SVO at B₃₂ is a tie within 0.03 (0.880 vs 0.875).
- **H2 — the certificate-aware objective matters: PASSES** (4/4 cells ≥ 0.10). TPR gains, `cert` − `mse`: LIST/B8 +0.603, LIST/B32 +0.250, SVO/B8 +0.621, SVO/B32 +0.100.
- **H3 — uniform coverage: no clear winner.** Uniform coverage is ≤ 0.006 in every cell, so no substitute is certified by a vocabulary-wide bound.

**Best certified (pairwise) coverage per budget:**

| task / budget | best cell | params | pairwise coverage |
|---|---|---:|---:|
| LIST/B8 | B8/tpr/cert | 186,788 | 0.637 |
| LIST/B32 | B32/tpr/cert | 742,148 | 0.961 |
| SVO/B8 | B8/paircode/cert | 18,912 | 0.798 |
| SVO/B32 | B32/paircode/cert | 75,936 | 0.880 |

## Reading (interpretation; not part of the pre-registered verdicts)

1. **"Agrees" became "provably agrees" for most held-out contexts.**
   - On list copy, a 742k-parameter TPR substitute is proved to make GPT-2's decision on 96% of 3,600 unseen
     contexts.
   - On the passive SVO probe, a 76k-parameter pair-code substitute reaches 88%.
   - Each certificate is a per-context proof. Nothing extends to contexts that were not checked: uniform coverage is
     ~0 everywhere, and a domain bound (T5(b)) remains `open`.
2. **The objective, not the family, is the big lever.** Under MSE the families certify 0.5–85%. The
   certificate-aware hinge raises TPR by up to +0.62. This is the positive counterpart of the earlier finding that
   MSE fits the wrong directions at a decode site.
3. **The task decides the family, and the conjunctive prediction is not supported overall.**
   - On LIST, the factored TPR wins clearly: one vector per noun, shared across 30 slot roles, against 3,000 pair
     codes learned from about 24 examples each.
   - On SVO, which has only 96 seen pairs, per-pair codes are affordable, and the pair-code wins or ties.
   - lm-sae#234 is about how GPT-2 *represents* binding. This study is about what *reproduces its decisions*
     cheapest, on seen pairs. The two need not agree, and here they do not consistently.
4. **The tail is the obstacle to certification without enumeration.** The hybrid certificate (top-32 pairwise plus a
   norm tail bound) certifies nothing. Every substitute is only certifiable by checking the full 50,257-token
   vocabulary per context.

**Scope.** One model; two templated tasks; substitutes take the **gold** symbolic structure as input, so this is not
a compression of GPT-2 on text; the 38.6M-parameter unembedding is still needed to decode.

