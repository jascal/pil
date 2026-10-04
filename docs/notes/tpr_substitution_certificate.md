# Certified last-layer substitution (T5(a)): coverage of the TPR fits

**Theorem (`proved`).** i-orca `examples/pic_core/PIC_Binding.thy`, `substitution_certified_max`, kernel-checked
(Isabelle2025-2, `quick_and_dirty = false`).

Setting:
- a finite token set `V`;
- decode logits `L(v) = ⟨r, U_v⟩ + b_v` with argmax `t`;
- any substituted decode input `r̂`.

If `L(t) − L(v) > 2·max_{v∈V} |⟨r − r̂, U_v⟩|` for every `v ≠ t`, then `t` is still the strict argmax of
`⟨r̂, U_v⟩ + b_v`. The theorem is a statement about the decode alone. It says nothing about whether any model is
TPR-shaped.

**Checker.** `experiments/substitution_certificate.py` (tests: `tests/test_substitution_certificate.py`).
- Per context, float64 computes `m = L(t) − max_{v≠t} L(v)` and `δ = max_v |⟨r − r̂, U_v⟩|` over **all 50,257**
  GPT-2 tokens.
- A Soufflé query issues `certified(c)` iff `⌊m·2³⁰⌋ > 2⌈δ·2³⁰⌉ + ⌈τ·2³⁰⌉`, with `τ = 10⁻⁶` logits.
  Margins are rounded down and δ up, so the query can only be more conservative than the real-number condition,
  provided |float64 error| ≤ τ.
- A Python twin is tested for parity. Every certified context is also checked to keep its argmax; this is the
  theorem's conclusion, and a violation would mean a checker bug.

**Run.** `experiments/tpr_projection_margin.py evaluate --post-hoc-kl --certify`. Site: GPT-2 small decode input on
the list-copy task of [`tpr_projection_margin_prereg.md`](tpr_projection_margin_prereg.md); 4,000 test contexts.
- The dump was regenerated on CPU (seed 0), so arm metrics differ from the GPU run in the third decimal.
- Certificate verdicts are Soufflé query results. Coverage is `empirical` (one model, one task).

## Coverage

| arm | agrees with host | median `m` | median `δ` | **certified** |
|---|---:|---:|---:|---:|
| `real` (r̂ = r; sanity) | 1.000 | 3.89 | 0.00 | 4000/4000 |
| `oracle` d_F = 8 (MSE TPR) | 0.056 | 3.89 | 10.44 | **0/4000** |
| `oracle` d_F = 32 (MSE TPR) | 0.832 | 3.89 | 7.36 | **0/4000** |
| post-hoc KL-trained TPR, d_F = 8 | 0.986 | 3.89 | 54.70 | **0/4000** |
| `cleanup` (unbind → snap → rebind) | 0.082 | 3.89 | 12.14 | 0/4000 |
| `proj-tpr` (projection onto span W, rank 240) | 0.996 | 3.89 | 1.94 | 1955/4000 (0.489) |
| `proj-pca` (rank-matched PCA projection) | 0.998 | 3.89 | 0.81 | 3767/4000 (0.942) |

No certified context changed its argmax: the soundness check passed on every arm.

## Reading

1. **No TPR fit at this site is certifiably faithful.** That includes the KL-trained fit, which agrees with the host
   on 98.6% of decisions.
   - The certificate needs the substitution error to be small on **every** token's readout direction.
   - The KL fit matches the host on the few tokens that matter for the softmax and is far off on the rest:
     median δ = 54.7 logits against a median margin of 3.89.
   - So decision agreement is not certifiable faithfulness. The 98.6% is an `empirical` agreement rate; the
     certificate, which would be `proved` per context, covers 0%.
2. **Projections are partly certifiable.** Both projections keep `r` itself in the part of the space they retain.
   - The rank-matched PCA projection certifies 94%; projection onto the TPR's span certifies 49%.
   - That matches the outcome note: the TPR subspace is not special beyond its rank, and is worse than PCA.
3. **Limits of this certificate.** T5(a) uses one uniform δ over the whole vocabulary.
   - A pairwise certificate (`L(t) − L(v) > ⟨r − r̂, U_t − U_v⟩` per rival) is exact and could cover more.
   - Its i-orca statement is a one-line consequence of the same algebra, but it is **not** proved here. Its coverage
     is `open`.
   - Pre-norm substitution remains `open` (PROPOSAL T5 scope).

## Pairwise and hybrid certificates (i-orca #28)

**Theorems (`proved`)** in i-orca `PIC_Binding.thy`, all 15 surface theorems checked by `i-orca check --session PIC_Core`:
- `substitution_pairwise_iff`: the substituted decode keeps `t` **iff** `L(t) − L(v) > ⟨r − r̂, U_t − U_v⟩` for every
  rival `v`.
- `uniform_implies_pairwise`: the uniform condition implies the pairwise one.
- `substitution_certified_hybrid`: pairwise on the top-K rivals of the real logits, plus the tail bound
  `L(t) − L(v) > |⟨r − r̂, U_t⟩| + ‖r − r̂‖·u_max` outside them.

**Checker.** `certificate_facts` / `certify_all_souffle` issue all three verdicts from one Soufflé program, with the
same conservative fixed point and a Python twin. Every run also re-checks the proved relations on the data:
- uniform ⊆ pairwise;
- hybrid ⊆ pairwise;
- every certified context keeps its argmax.

**Run.** Dump regenerated on GPU (seed 0), 4,000 test contexts. Arm metrics differ slightly from the CPU-dump run
above, because fits are GPU-nondeterministic.

| arm | agrees with host | uniform | hybrid (K = 32) | **pairwise** |
|---|---:|---:|---:|---:|
| `real` | 1.000 | 1.000 | 1.000 | 1.000 |
| MSE TPR d_F = 8 | 0.055 | 0.000 | 0.000 | 0.055 |
| MSE TPR d_F = 32 | 0.844 | 0.000 (1/4000) | 0.000 | 0.844 |
| KL-trained TPR d_F = 8 | 0.986 | 0.000 | 0.000 | **0.986** |
| `cleanup` | 0.080 | 0.000 | 0.000 | 0.080 |
| projection onto span W | 0.996 | 0.476 | 0.000 | 0.996 |
| rank-matched PCA projection | 0.998 | 0.942 | 0.141 | 0.998 |

**Reading.**
- **Pairwise coverage equals decision agreement on every arm, to the context.** This is what the iff predicts: the
  exact certificate certifies exactly the contexts where the substitution keeps the decision. The 98.6% agreement
  of the KL fit therefore *is* a certified 98.6%, per context, on this finite set. The uniform form's 0% was the
  looseness of a single vocabulary-wide δ, not unfaithfulness.
- **The hybrid tail bound is too loose to help.** `‖r − r̂‖·u_max` with 50,257 tokens overwhelms the gaps outside
  the top 32. Its value is checkability from the top-K logits alone, and here it buys almost nothing.
- **What "certified" means now.** Pairwise certifies the *observed* contexts exactly. It does not extend to unseen
  contexts. A guarantee for unseen contexts needs a bound that holds over a domain, which is the uniform form or
  T5(b), and that remains `open`.
