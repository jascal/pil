# TPR projection as margin widening — pre-registration

**Status: written 2026-10-03, before any dump was generated or read.** Motivated by McCoy, Soulos, Linzen & Smolensky
(2026, arXiv:2608.29530). Their sentence-unpacking decoder, trained on *real* LLM encodings, scored *higher* when fed
the fitted TPR approximations: 0.96 vs 0.71 for GPT-OSS on complex sentences. They read this as "limitivism": the
network realises the symbolic structure noisily, and the TPR realises it exactly.

That is pil's thesis in another form: give up some host fidelity, gain retrievability. The theory is in
`../../../PIC_TPR_CONJECTURES.md`. T6(a): **linear** projection onto a subspace containing the readout differences
cannot change any margin, so a gain must come from out-of-span components or from **snapping** to a code point. T5:
last-layer substitution is certified by the margin theorem.

## 1. Setting

**Model and task.** GPT-2 small, with a copy task that needs the bound structure:

> `Here is a list of words: w1, w2, w3, w4, w5. Again: w1, …, wp,` with p ∈ {1,2,3,4}; gold next token ` w_{p+1}`.

The wᵢ are distinct single-token nouns.

**Site.** The decode input `u = ln_f(h_L)` at the final position, so logits are `L(v) = ⟨u, U_v⟩` (tied unembedding,
no bias). Substituting here is exactly the T5 setting: no nonlinearity sits between `u` and the decision.

**Roles.** These are task-specific pair roles, as in the paper's GPT-OSS analysis:
- list filler `wᵢ` → role `(list slot i, p)`: 20 roles;
- query filler `wⱼ` → role `(query slot j, p)`: 10 roles.

**TPR.** `û(σ) = W(Σ f ⊗ r) + b`, fit by MSE on train `u`.
- The main fit uses d_F = 8 and d_R = 30, so `W` is 768×240: a real subspace, not the whole space.
- d_F = 32 is reported as a secondary fit.

## 2. Arms (all evaluated on held-out test contexts)

| arm | vector fed to the decode |
|---|---|
| `real` | `u` |
| `oracle` | `û(σ)` from the gold structure (DISCOVER's own substitution) |
| `cleanup` | `û(σ̂)`, where `σ̂` comes from linear readouts trained on train `u` (unbind → snap → rebind); deployable |
| `proj-tpr` | `b + P_W(u − b)`, orthogonal projection onto `span(W)` (rank 240) |
| `proj-pca` | rank-matched control: projection onto the top-240 PCA subspace of train `u` |

## 3. Metrics

- **Gold accuracy:** argmax = gold token.
- **Host fidelity:** decision agreement with `real`, and `KL(real ‖ arm)`.
- **Decode-side margins:** `margin_to_worst(L, gold)` (`pil.geometry`). Mean margin, and **retrievable fraction**
  `Pr[margin ≥ γ]` at γ ∈ {0, 1, 2} nats. **γ = 1 is primary.**
- **T6(a) diagnostic (not a prediction):** the share of `‖U_t − U_v‖²` inside `span(W)`, averaged over gold/rival
  pairs.

## 4. Decision rules (fixed now)

- **Q1 — limitivism as margin.** `oracle` raises the γ=1 retrievable fraction by ≥ 0.10 over `real`, and does not
  lower gold accuracy. *Fail:* the TPR's exact realisation is not more decodable than the host's noisy one at this site.
- **Q2 — snapping is deployable.** `cleanup` has gold accuracy ≥ `real` *and* γ=1 retrievable fraction ≥ `real` + 0.05.
  *Fail:* unbinding errors cost more than snapping gains, and the gain is oracle-only.
- **Q3 — projection is not the lever.** `proj-tpr`'s γ=1 retrievable fraction is within 0.05 of `proj-pca`'s, and
  neither beats `oracle`. *Fail (interesting):* the TPR subspace is special beyond its rank, so linear projection
  onto it widens margins. That would bear on T6(a), via out-of-span components.
- **Cost accounting.** Every arm reports its host-fidelity price (KL and decision agreement). A widening is reported
  only together with its price.

## 5. Scope

One model, one templated task, supervised roles, last-layer site. Results are `empirical`. Nothing here is a
certificate. T5 would make `oracle`'s preserved decisions certifiable per context, but it is not kernel-checked yet.
