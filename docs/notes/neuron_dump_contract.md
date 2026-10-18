# Neuron dump contract (emitter spec)

This is the dump a later `fieldrun` emitter must produce before anyone runs the non-vacuous forge-tax certificate. It is not a result, and it is not a rerun of the free-frame LP. `experiments/forge_tax_certificate.py` stays the block-level seam. The 0/40 Qwen finding is untouched. The same text is the §6 contract in the sister spec `../pic/spec/forge_tax_certificate.md`.

One record per held-out position `x`, plus one shared unembedding:

- **Per-neuron gate** `g_k(x) ∈ ℝ`. The input-dependent coefficient on neuron `k`. For a SwiGLU block this is the product that actually multiplies the output column, not a top-k label of the positive pole.
- **Output column** `a_k ∈ ℝ^d`. The neuron's column of `W_out`. Sources are rank-1: the write is `g_k(x) · a_k`. Blocks are not rank-1 and cannot be substituted for these columns.
- **SwiGLU sign regime.** The sign of the gate pre-activation and the sign of the value pre-activation, one of `{+,+} {+,−} {−,+} {−,−}`. The double-negative regime `−×−` writes a positive output and carries the negative pole's meaning. Dropping the regime and keeping only `g_k(x) > 0` scores those positions as the wrong proposition.
- **`U` fixed.** The model's unembedding (and bias). It is data, not a variable.
- **`A` free.** The write columns `{a_k}` are the variables of one LP shared across every position (`r_x = A g(x)`). Freeing a separate `A` per position collapses back to the vacuous free-frame question.
- **Margin split on `2δ`.** Each position is tagged by whether its measured decode margin clears `2δ`, with `δ` the per-token perturbation bound from `PIC_SPEC.md` §5.5. Positions with margin `≤ 2δ` and positions with margin `> 2δ` are certified as separate sets. A pooled rate is not the contract.

The free-frame LP remains a sound relaxation of this program. An emitter that omits the sign regime, frees `U`, or pools the two margin bands has not implemented this contract.
