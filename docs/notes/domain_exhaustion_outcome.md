# Certifying a whole finite domain by exhaustion — outcome

**Pre-registration:** [`domain_exhaustion_prereg.md`](domain_exhaustion_prereg.md), with addendum A.
- Provenance, in order:
  - pre-registration `eeed093`, pushed before any code;
  - addendum A `b914142` (smoke disclosure, no design change);
  - script and tests frozen at `a0c525c` (`experiments/domain_exhaustion.py`, sha256 `35b674b5…`). It reuses
    `halfspace_verdict.py` (#141), `cleanup_radius.py` (#140) and `cleanup_certificate.py` (#139) unchanged, and the
    run used all of them unchanged.
- Seeds: stimulus 71 (domain order), split 72, fit {0, 1, 2}.
- **D = all 24,960 SVO contexts**: 7,200 train, 1,200 validation, 3,600 test, and 12,960 never sampled. 0 contexts
  were out of vocabulary.
- The dump and the run were held under one `systemd-inhibit --what=sleep:idle` lock and were not interrupted
  (3,820 s).
- Artifacts:
  - [`domain_exhaustion_summary.json`](domain_exhaustion_summary.json);
  - [`domain_exhaustion_certified.npz`](domain_exhaustion_certified.npz): the `D_cert` bitmask per fit over the
    canonical 40 × 40 × 16 (S, O, V) grid, plus the domain mask.
- Tags: each context in a fit's `D_cert` is **proved** for that fit (kernel-checked per-context certificate). Every
  count below is `empirical`.

## Verdict

**H1 (most of the domain is certified): PASS.** Seed-mean `F_cov(D)` is **0.666** (SVO `d_F=32`, `mse`) and
**0.692** (`t6`). The best single fit (`t6`, seed 2) certifies **18,145 of 24,960** contexts (0.727).

**H2 (never-sampled contexts certify like test contexts): PASS** in all 3 eligible cells:

| cell | `F_cov(test)` | `F_cov(never)` | gap |
|---|---:|---:|---:|
| SVO `d_F=32` `mse` | 0.665 | 0.659 | 0.006 |
| SVO `d_F=32` `cert` | 0.242 | 0.247 | 0.005 |
| SVO `d_F=32` `t6` | 0.687 | 0.685 | 0.002 |

**Domain-wide equivalence: no.** No fit certifies all of D. The fewest uncertified contexts is **6,815** (`t6`, seed
2).

**Soundness:** the iff checks held on all 18 fits, with 1 tie (`cert` seed 1, a slack within 10⁻⁶ of 0; counted, as
pre-registered). All **1,219,330** sampled neighbourhood perturbations held. Soufflé and Python agreed on every
context. No fit was flagged.

## Per cell (seed mean [min, max]; D = 24,960)

| setting | objective | `F_cov(D)` | `F_cov(train)` | `F_cov(test)` | `F_cov(never)` | `E(D)` | `A(D)` | median `ρ_loc/‖n‖` | fewest uncertified |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SVO `d_F=8` | mse | 0.0029 | 0.0028 | 0.0035 | 0.0029 | 0.024 | 0.047 | 0.0105 | 24,748 |
| SVO `d_F=8` | cert | 0.0046 | 0.0044 | 0.0047 | 0.0046 | 0.007 | 0.661 | 0.0045 | 24,699 |
| SVO `d_F=8` | t6 | 0.0122 | 0.0129 | 0.0132 | 0.0113 | 0.032 | 0.170 | 0.0083 | 24,082 |
| SVO `d_F=32` | mse | **0.6658** [0.6353, 0.7054] | 0.6814 | 0.6647 | 0.6586 | 0.853 | 0.767 | 0.0275 | 7,352 |
| SVO `d_F=32` | cert | 0.2503 [0.1426, 0.3057] | 0.2604 | 0.2422 | 0.2471 | 0.280 | 0.874 | 0.0117 | 17,329 |
| SVO `d_F=32` | t6 | **0.6925** [0.6596, 0.7270] | 0.7077 | 0.6873 | 0.6849 | 0.812 | 0.848 | 0.0281 | 6,815 |

## Post-hoc (not pre-registered; `empirical`)

- **The union over all 18 fits certifies 22,829 of 24,960 contexts (0.915).** That is, for 91.5% of the domain some
  fitted substitute provably agrees with GPT-2 on that context. This is a statement about some substitute per
  context, not about one substitute. 2,131 contexts are certified by no fit.
- **The intersection of the six SVO `d_F=32` `mse` and `t6` fits** is 8,555 contexts (0.343): contexts every one of
  those fits certifies.

## Reading (interpretation)

1. **Most of a whole finite domain is certified, context by context.** About two-thirds of all 24,960 SVO contexts
   carry a kernel-backed certificate that the substitute's cleaned decision equals GPT-2's. This is the first
   domain-level statement in this line: "proved on `D_cert`", with `D_cert` saved explicitly.
2. **Certificates generalise to contexts that were never sampled.** On the 12,960 never-sampled contexts, coverage is
   within 0.006 of the test split in every eligible cell. Training contexts certify only slightly more (by
   0.01–0.02). The substitute's certifiability is not a property of the contexts it saw.
3. **The whole domain is not certified,** and nothing here suggests a single TPR fit will reach it. Even the union of
   18 fits leaves 2,131 contexts uncertified.
4. **The `d_F=8` substitutes certify almost nothing,** as in #139–#141. The capacity at `d_F=32` is what makes exact
   clean-up common.

## Not claimed

- Exhaustion runs GPT-2 on every context, so it saves no host compute. It certifies contexts the substitute never
  saw, not contexts the host never saw.
- The certificate holds on `D_cert` only, within one finite template. Nothing is claimed for other sentences.
- GPT-2 small, the SVO template, the TPR family only. LIST's domain (about 3.6 × 10¹⁰ contexts) cannot be enumerated.
