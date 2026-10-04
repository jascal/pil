# Certification beyond GPT-2 small — pre-registration

**Status: written 2026-10-04, committed and pushed before any code for this study was written or run.**

This re-runs two finished GPT-2-small studies, unchanged in their rules, on two more capable RoPE models:
- **A — certified compressed substitutes:** [`certified_substitutes_prereg.md`](certified_substitutes_prereg.md)
  (pil #136).
- **B — compiled TPR behind n-gram and idiom layers:** rosetta `docs/compiled-tpr-prereg.md` (rosetta #59).

The question: does a more capable model make more of its decisions certifiable by a small substitute, or does the
low-margin residual persist?

## 1. Models (fixed)

| model | HF revision | layers / hidden / vocab | tied unembedding | BOS |
|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | 28 / 1536 / 151,936 | yes | none |
| `meta-llama/Llama-3.2-1B` | `4e20de362430cd3b72f300e6b0f18e50e7166e08` | 16 / 2048 / 128,256 | yes | 128000, prepended to every context |

**Usage.**
- Both models are used as raw completion models: no chat template, including for the Instruct model.
- **Precision: fp32 for both**, for decode-input dumps and for substitute training.
- **Site:** the decode input, i.e. the final-norm output at the last position. Logits are `U u` with no bias,
  because both models tie their unembedding.

**Certificates** are over the **full vocabulary** of each model: 151,936 and 128,256 tokens. The checker is the
merged `substitution_certificate.py` (float64, fixed point 2³⁰, τ = 10⁻⁶), unchanged.

## 2. Tokenization rule

Words are admitted iff `" " + word` is a single token in **that model's** tokenizer. Candidate lists, caps and order
are those of the GPT-2 studies; only the filter changes.

The template words must also be single tokens: `The`, ` the`, `.`, ` was`, ` by`, `,`, ` Again`, `:`, and the
LIST preamble. Templates are tokenized by the model's own tokenizer.

For Llama, the BOS token occupies position 0. Every positional rule is offset by one; COPY's distance-from-end
roles exclude the BOS position.

## 3. Study A — certified substitutes (rules unchanged from pil #136)

Unchanged:
- tasks: LIST copy (100 nouns, `p ∈ {1..4}`) and SVO passive probe;
- families: additive-k, TPR d_F ∈ {8, 32}, pair-code-k;
- objectives: `mse` and `cert` (top-32 hinge plus 0.1 · uniform tail);
- training: 3,000 steps, batch 512, lr 3e-3, fit seeds {0, 1, 2};
- budgets: B₈ / B₃₂ = that model's TPR param counts, with the 768 dimension replaced by its hidden size;
- the 60/10/30 split and the unseen-pair drop rule;
- metrics: pairwise (primary), uniform, hybrid with K = 32;
- decision rules: H1 (pair-code wins: ≥ 0.03 over both others in ≥ 3/4 cells), H2 (`cert` beats `mse` for TPR by
  ≥ 0.10 in ≥ 3/4 cells), H3 (H1's rule on uniform coverage).

**Fresh seeds:** stimulus **41** (SVO uses 41 + 1000), split **42**.

## 4. Study B — compiled TPR (rules unchanged from rosetta #59)

Unchanged:
- tasks: SVO passive probe and closed-pool COPY stress (120 nouns, 3 layouts, 4,000 contexts each);
- layers: dev-filtered train-only n-gram → idiom (SVO: copy-subject with a dev-selected verb whitelist; COPY: rosetta's
  frozen guarded copy circuit, unchanged) → compiled TPR (Datalog role parse, weighted input facts, d_F = 32,
  `cert` objective, fit seed 0, dev-selected margin θ) → abstain;
- test firing domains frozen before test references;
- `dl/equiv.dl` certificates per layer and for the composite;
- the residual split (SVO: `R_obj` / `R_other` / `R_in_sentence`; COPY: `R_ctx` / `R_out`);
- **Q1:** clean TPR certificate **and** additional certified coverage ≥ 0.05 of `R` and ≥ 10 contexts.

**Fresh seeds:** SVO **51**, COPY **52**, split **53**.

**References for `equiv.dl`.**
1. Fresh **f32** fieldrun bundles are converted from the pinned revisions with `fieldrun convert --dtype f32`. The
   existing Llama bundle is f16 and is **not** used.
2. A bundle is admitted as the reference source iff its argmax matches HF fp32 on **≥ 99.5%** of a **frozen parity
   set**:
   - 400 contexts per task, generated with seed 61;
   - disjoint from the study data and never used for selection.
3. If a bundle is not admitted, HF fp32 argmax is the reference source, labelled as such.
4. Parity is reported either way.

## 5. Pre-stated comparison (the GPT-2-small results)

| study | GPT-2 small |
|---|---|
| A, best pairwise coverage | LIST: 0.637 (B₈, TPR) / **0.961** (B₃₂, TPR). SVO: 0.798 (B₈, pair-code) / **0.880** (B₃₂, pair-code) |
| A, verdicts | H1 no clear winner · H2 passes 4/4 · H3 no winner (uniform ≤ 0.006) |
| B, Q1 | **no** on both tasks. SVO: TPR fires 469, agrees 468, certificate fails (`nmiss = 1`). COPY: the idiom pre-empts everything (122 mismatches) |

**Headline rules (per model; fixed now):**
- **"More certifiable than GPT-2"** iff Study A's best pairwise coverage at **B₃₂** exceeds GPT-2's by **≥ 0.05 on
  both tasks**.
  - Budgets are pinned to each model's own TPR sizes, so this compares *at matched substitute structure*, not at
    equal parameters. That is stated, not hidden.
- **"The low-margin residual persists"** iff Study B's Q1 fails on **both** tasks for that model.
- **Both can hold at once.** Each is reported per model, and the conclusion is the conjunction across the two models.

## 6. Not claimed

- Certificates are proofs over these finite test domains only. Coverage numbers are `empirical`.
- Two models do not make a scaling law. "More capable" here means only these two models compared with GPT-2 small.

## 7. Artifacts

- pil: `experiments/certified_substitutes.py` and `experiments/compile_tpr.py`, parameterised by `--model`, with the
  rules unchanged.
- rosetta: `py/benchmark_compiled_tpr.py`, parameterised by `--model`.
- Outcome: `docs/notes/beyond_gpt2_outcome.md` (pil) and the rosetta outcome note. This file is not edited after the
  first run.
