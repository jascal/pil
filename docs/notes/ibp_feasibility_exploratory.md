# Can a model-side bound through GPT-2 certify a nuisance class? — exploratory feasibility

**Status: EXPLORATORY. Not pre-registered.** A feasibility measurement for study (2) after pil #143. It asks whether
a sound bound computed *through* GPT-2 can approach the hull ceiling of #143. Its claims are tagged `empirical`, and
its verdict is a feasibility judgement, not a confirmatory test.

- Code: `experiments/ibp_gpt2.py` (interval bound propagation) and `experiments/ibp_feasibility.py` (driver),
  committed at `359a505` and run from that commit.
- Tests: `tests/test_ibp_gpt2.py` checks the soundness of every primitive (GELU, linear, LayerNorm, attention, MLP)
  by sampling points inside the input box.
- Summary: [`ibp_feasibility_summary.json`](ibp_feasibility_summary.json).

## Setup

- **Nuisance:** one sentence-initial token, `"{Word}, the S V the O. The O was V by the"`, with 23 single-token words
  ("Yesterday", "Today", …). All options tokenize to the same length, so variants are position-aligned and differ only
  at token 0. 100 σ-classes (seed 101); nuisance sets of size k ∈ {2, 8, 23}.
- **IBP:** the token-0 embedding box (the elementwise min/max over the k options) is propagated soundly through all 12
  blocks: interval LayerNorm, attention with an interval softmax, interval GELU. The result is a box for the decode
  input `u`.
- **Sanity, both held:**
  - with k = 1 the box equals the exact float64 forward pass (max deviation 7.1 × 10⁻¹⁴, width 0);
  - every actual residual lies inside its box (300/300; the run aborts otherwise).
- **Measured:** the host-agreement half of #141's certificate (GPT-2's decision `t` beats every rival over the
  region) on three regions:
  1. the IBP box;
  2. the **tightest possible box**: the exact coordinate-wise range of the actual residuals, the best any box-based
     method could ever produce;
  3. the **hull** of the actual residuals (#143's ceiling).

## Result

| k | decision-constant classes | certified on the IBP box | on the tightest box | on the hull | median IBP box width / actual range |
|---:|---:|---:|---:|---:|---:|
| 2 | 92 | **0** | 35 | 92 | 2.0 × 10¹⁴⁹ |
| 8 | 74 | **0** | **0** | 74 | 1.5 × 10¹⁴⁹ |
| 23 | 70 | **0** | **0** | 70 | 1.7 × 10¹⁴⁹ |

- **IBP is vacuous by about 149 orders of magnitude.** At the last position, the box width grows by roughly 10¹² per
  layer: about 10⁹ after layer 1 and about 10¹⁴⁵ after layer 12 (class 0, every k).
- **Any box-based method is capped at 0 for k ≥ 8.** Even the tightest box, which no sound box propagation can beat,
  certifies no class at k = 8 or 23, while the hull certifies all of them.
  - The residuals vary along a few correlated directions. A box ignores correlations, so it is enormously larger than
    the hull in the directions the certificate tests.

## Feasibility verdict (interpretation)

- **Infeasible with box (interval) methods,** for two separate reasons: propagation (a blow-up of about 10¹⁴⁹), and
  the domain itself (even an exact box fails for k ≥ 8).
- **Open, and doubtful, for relational methods** (zonotopes, DeepPoly/CROWN-style linear bounds). These track
  correlations, so they are not ruled out by the tightest-box result. But they would have to stay near-exact through
  12 layers of GPT-2 under discrete token swaps. Nothing here suggests they would. Testing one is the only remaining
  step on this route.
- The model-side route to certifying contexts the host has not run is therefore **not viable with standard interval
  machinery**.

## Not claimed

- No relational method was tried.
- One nuisance (sentence-initial single tokens), GPT-2 small, 100 classes.
- The IBP here is a straightforward implementation. Tighter interval rules (for example a variance-aware LayerNorm)
  could shrink the blow-up, but cannot beat the tightest box, which already fails for k ≥ 8.
