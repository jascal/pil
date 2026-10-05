# Certifying a whole finite domain by exhaustion — pre-registration

**Status: written 2026-10-05, committed and pushed before any code for this study existed.**

**Context.**
- jascal/pil#141: per-context certificates (`nearest_iff_halfspace` for clean-up, plus the pairwise iff for host
  agreement) certify about two-thirds of held-out SVO `d_F=32` test contexts. Each needs the observed residual.
- Unseen contexts: the SVO template domain is **finite**. With S ≠ O over 40 occupations, and 16 verbs, there are
  40 × 39 × 16 = **24,960** contexts.
  - Earlier studies sampled 12,000 of them (60% train, 10% validation, 30% test). About 13,000 were never even
    sampled.
  - Enumerating all of D gives a certificate **over D**: the explicit set of contexts on which substitute and host
    provably agree, context by context. This is the same kind of claim as rosetta's `equiv.dl` over a finite test
    domain. It still costs one GPT-2 forward pass per context.

**The questions.**
1. What share of the **whole domain D** is certified per context?
2. Do the certificates hold up on **never-sampled** contexts as well as on the held-out test split?

## 1. Model, domain, substitutes

- GPT-2 small; decode input; the SVO template only (LIST's domain, about 3.6 × 10¹⁰ contexts, is not enumerable).
- **D = all 24,960 SVO contexts.** It is produced by `certified_substitutes.build_rows` unchanged, with
  `N_CONTEXTS ≥ |D|`, which returns every combination in a seeded shuffled order.
- **Partition of D** (stimulus seed **71** fixes the order; split seed **72**):
  - the first 12,000 in that order are the **sampled pool**, split 60 / 10 / 30 into train / validation / test;
  - the remaining 12,960 are **never sampled**.
- **Substitutes:** TPR at SVO `d_F=8` and `d_F=32`; objectives `mse`, `cert`, `t6`; fit seeds **{0, 1, 2}**. All are
  trained on the train split only (#136/#139 code, unchanged). That is 2 settings × 3 objectives × 3 seeds =
  **18 fits**.
- `F_s` = fillers seen in role `s` in train. A context whose true filler is outside `F_s` cannot be cleaned exactly,
  so it is uncertified; such contexts are counted.

## 2. The certificate per context (exactly #141's)

- `clean(c)`: every half-space slack `‖d_a‖²/2 − ⟨M_s n, d_a⟩ > τ`.
- `agree(c)`: `min_{v≠t} ⟨u, U_t − U_v⟩ > τ`, with `t` the code point's decision.
- `full(c) = clean(c) ∧ agree(c)`.
- These are Soufflé verdicts over fixed-point facts (scale 2³⁰, τ = 10⁻⁶), with a Python twin. They are computed by
  `halfspace_verdict.py`'s functions, reused unchanged.

**Aborts (as #141):**
- the iff checks (`clean` = measured exactness, `agree` = measured agreement; ties within 10⁻⁶ counted);
- the neighbourhood checks (8 random plus 2 worst-case perturbations at `0.999·ρ_loc` around every certified
  context).

**The domain certificate.** For each fit, the certified set `D_cert ⊆ D` is saved as a bitmask over D's canonical
order: (S, V, O) indices, which do not depend on the shuffle.

## 3. Metrics (seed mean and range per cell)

- **`F_cov(D)`**: certified share of all of D.
- **`F_cov(train)`, `F_cov(test)`, `F_cov(never)`**: certified share of each part.
- `E` and `A` per part; the uncertified count due to an out-of-`F_s` filler.
- **Domain-wide equivalence:** whether any fit certifies **all** of D, and the smallest uncertified count over fits.
- The `ρ_loc / ‖n‖` median over certified contexts in all of D.

## 4. Decision rules (fixed now)

- **H1 (most of the domain is certified).** In at least one cell, seed-mean **`F_cov(D)` ≥ 0.5**.
- **H2 (never-sampled contexts certify like test contexts).** In every cell with seed-mean `F_cov(test)` ≥ 0.05,
  **`|F_cov(never) − F_cov(test)|` ≤ 0.05** (seed means). If no cell qualifies, H2 is **untestable**.
- **Domain-wide equivalence** (descriptive): reported as yes/no per fit. No hypothesis is attached.

**No expected outcome is stated.** #141's test-split coverage is prior, published context.

## 5. Not claimed

- The certificate is **proved over D**, a finite template domain, and only on `D_cert`. Nothing is claimed for
  sentences outside the template.
- Exhaustion runs GPT-2 on every context, so it saves no host compute. It certifies contexts the substitute never
  saw, not contexts the host never saw.
- GPT-2 small, the SVO template, the TPR family only.

## 6. Protocol and artifacts

- Script `experiments/domain_exhaustion.py`, frozen before the real run.
- `--smoke`: an enumeration-preserving smoke domain (the first 600 contexts of a seed-999 shuffle; 30 steps). Smoke
  numbers are not results.
- The run is wrapped in `systemd-inhibit --what=sleep:idle`.
- Any change after smoke goes in a dated addendum, before the real run, disclosing what the smoke run showed.
- Outcome: `docs/notes/domain_exhaustion_outcome.md`, the summary JSON, and the `D_cert` bitmasks
  (`docs/notes/domain_exhaustion_certified.npz`). None is edited after the first real run.

---

## Addendum A (2026-10-05, after a smoke run, before any real run): disclosure only, no design change

- **What was run:** the `--smoke` pipeline (the first 600 contexts of a seed-999 shuffle; pool 300; 30 steps), for
  all 18 fits, under `systemd-inhibit`. Seeds 71/72 are untouched.
- **Result:** it completed and wrote the summary and the `D_cert` bitmasks. The iff checks held (0 ties).
  - `F_cov` = 0 everywhere, since 30-step fits are undertrained.
  - 15 of 600 smoke contexts were out of vocabulary, because the smoke train split is only 180 contexts.
- **The design is unchanged. No expected outcome is stated.**

## Erratum (2026-10-05, after the real run): documentation only, no design change

§2 says the `D_cert` bitmask is over D's "canonical order: (S, V, O) indices". The frozen script (`a0c525c`,
`canonical_index`) actually indexes an **(S, O, V)** grid: `S·40·16 + O·16 + V`, with V fastest, packed with
`np.packbits` (default big-endian bit order). The committed artifact follows the script. The run is unaffected: every
per-part coverage figure in the summary is reproduced exactly by decoding the bitmasks this way. Decode with
`experiments/domain_certified.py`; `tests/test_domain_certified.py` pins the axis contract. Found in review (Grok,
#142).
