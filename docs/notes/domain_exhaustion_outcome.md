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
  - [`domain_exhaustion_certified.npz`](domain_exhaustion_certified.npz): the `D_cert` bitmask per fit, plus the
    domain mask. The grid is 40 × 40 × 16 **(S, O, V)**, index `S·640 + O·16 + V` (V fastest; **not** (S, V, O)),
    packed with `np.packbits` (big-endian). Decode with `experiments/domain_certified.py`.
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
2. **Certificates generalise to filler combinations that were never sampled.** On the 12,960 never-sampled
   contexts, coverage is within 0.006 of the test split in every eligible cell. Training contexts certify only
   slightly more (by 0.01–0.02).
   - Every role filler did appear in training (0 out of vocabulary). So this is generalisation to unseen
     **combinations** (triples), not to unseen fillers.
   - H2 is judged on seed means, as registered. The `cert` cell's seed range is wide (`F_cov(D)` 0.14–0.31), and the
     means hide that.
3. **The whole domain is not certified,** and nothing here suggests a single TPR fit will reach it. Even the union of
   18 fits leaves 2,131 contexts uncertified. As a stability number, the 8,555 contexts (0.343) certified by **all**
   six `d_F=32` `mse` and `t6` fits are more informative than the union.
4. **The `d_F=8` substitutes certify almost nothing,** as in #139–#141. The capacity at `d_F=32` is what makes exact
   clean-up common.
5. **The binding constraint is the conjunction.** At `d_F=32` `mse`, E(D) = 0.853 and A(D) = 0.767 but
   `F_cov(D)` = 0.666. At `d_F=8` `cert`, agreement is 0.66 but coverage is 0.005. Clean-up and agreement must both
   hold.

## Not claimed

- Exhaustion runs GPT-2 on every context, so it saves no host compute. It certifies contexts the substitute never
  saw, not contexts the host never saw.
- The certificate holds on `D_cert` only, within one finite template. Nothing is claimed for other sentences.
- GPT-2 small, the SVO template, the TPR family only. LIST's domain (about 3.6 × 10¹⁰ contexts) cannot be enumerated.

## Post-review changes (2026-10-05; review by Grok on #142)

No result changed. The changes are:
- **Bitmask contract documented:** the (S, O, V) axis order and the big-endian packing (Artifacts, above). The
  pre-registration's "(S, V, O)" and the `canonical_index` docstring were wrong; the script's formula and the
  artifact were right. The pre-registration has a dated post-run erratum, and the docstring is fixed (a docstring-only
  diff). `experiments/domain_certified.py` decodes the masks, and `tests/test_domain_certified.py` pins the contract
  (it fails on an axis swap) and checks the artifact against the summary.
- **Reading 2** now says "unseen combinations, not unseen fillers", and notes the `cert` cell's seed range.
- **Reading 3** adds the intersection as the stability number; **reading 5** (the conjunction) is new.
- **Validation coverage**, post-hoc, computed from the committed bitmasks, so that the four parts sum to D:

  | cell | `F_cov(val)` (seed mean [min, max], 1,200 contexts) |
  |---|---:|
  | SVO `d_F=8` mse | 0.0025 [0.0000, 0.0075] |
  | SVO `d_F=8` cert | 0.0058 [0.0000, 0.0133] |
  | SVO `d_F=8` t6 | 0.0144 [0.0000, 0.0417] |
  | SVO `d_F=32` mse | 0.6542 [0.6133, 0.6983] |
  | SVO `d_F=32` cert | 0.2494 [0.1358, 0.3075] |
  | SVO `d_F=32` t6 | 0.6975 [0.6783, 0.7325] |

- **The table check** ("checked programmatically against the JSON") is now a test in the repo:
  `tests/test_domain_exhaustion_outcome.py`.
- **Known, not changed:** the script mutates module globals of `certified_substitutes` (`N_CONTEXTS`, `STEPS`). That
  is fine run as a script, but unsafe if imported in-process alongside other users. Changing it would alter the
  frozen code.
