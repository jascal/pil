# Stable certified training: gradient clipping and what a stable student certifies — outcome

**Pre-registration:** [`stable_student_prereg.md`](stable_student_prereg.md), with addendum A.
- Provenance, in order:
  - pre-registration `a61560f`, pushed before any code, disclosing an exploratory probe; **the thresholds were set
    after the probe, which predicted H1–H3 pass and H4 fail**;
  - script and tests frozen at `b3eaa26` (`experiments/stable_student.py`, sha256 `d4b16c67…`);
  - addendum A `413b950` (smoke; no design change). The run used the frozen script unchanged.
- A first launch failed at import, before any training, because it was run from a copy outside the repo. It was
  relaunched in place with the same frozen file (sha checked). No numbers were produced or seen.
- Seeds: stimulus 151, split 152, fit {20–24}. 4,000 σ-classes × 23 words; class split 2,400 / 400 / 1,200; 104
  outputs. Of the test classes, GPT-2 is **seen-constant on 68.8%** and all-constant on 67.8%.
- The run was held under `systemd-inhibit` and was not interrupted (2,378 s).
- Files: summary [`stable_student_summary.json`](stable_student_summary.json); log
  [`stable_student_run.txt`](stable_student_run.txt).
- Tags: each certificate is **proved** for that student, as a sound interval bound in float32 (sound up to rounding).
  Every count is `empirical`.

## Verdict

**As the disclosed probe predicted:**
- **H1 (0 of 5 `ibp_clip` fits collapsed): PASS** (0 of 5).
- **H2 (`ibp_clip` faithfulness ≥ plain − 0.02): PASS** (0.842 vs 0.839).
- **H3 (`ibp_clip` `cert16` ≥ 0.90): PASS** (0.9998).
- **H4 (`ibp_clip` `cert23` ≥ 0.50): FAIL** (0.000 on every seed).

**Soundness:** no abort fired; the k = 1 box was exact.

**Reported, no hypothesis attached:**
- The unclipped `ibp` arm collapsed on **4 of 5** fits (the probe found 6 of 10 plus 1 partial; #146 found 1 of 3).
- The collapse cut-off is the plain mean minus 0.10, which is 0.739.

## Per arm and seed (test classes)

| arm | seed | faithfulness (seen / held-out words) | `cert16` | `cert23` | `CF23` | `FC23` | `FC16` | collapsed |
|---|---:|---|---:|---:|---:|---:|---:|---|
| plain | 20 | 0.839 / 0.851 | 0.000 | 0.000 | 0.000 | — | — | — |
| plain | 21 | 0.838 / 0.851 | 0.000 | 0.000 | 0.000 | — | — | — |
| plain | 22 | 0.845 / 0.857 | 0.000 | 0.000 | 0.000 | — | — | — |
| plain | 23 | 0.833 / 0.836 | 0.000 | 0.000 | 0.000 | — | — | — |
| plain | 24 | 0.842 / 0.852 | 0.000 | 0.000 | 0.000 | — | — | — |
| ibp | 20 | 0.791 / 0.802 | 0.999 | 0.913 | 0.583 | 0.362 | 0.313 | no |
| ibp | 21 | 0.038 / 0.039 | 0.000 | 0.000 | 0.000 | — | — | yes |
| ibp | 22 | 0.038 / 0.039 | 0.000 | 0.000 | 0.000 | — | — | yes |
| ibp | 23 | 0.688 / 0.694 | 0.830 | 0.821 | 0.496 | 0.396 | 0.282 | yes |
| ibp | 24 | 0.395 / 0.395 | 0.172 | 0.000 | 0.000 | — | 0.252 | yes |
| ibp_clip | 20 | 0.846 / 0.855 | 1.000 | 0.000 | 0.000 | — | 0.312 | no |
| ibp_clip | 21 | 0.835 / 0.845 | 1.000 | 0.000 | 0.000 | — | 0.312 | no |
| ibp_clip | 22 | 0.844 / 0.852 | 1.000 | 0.000 | 0.000 | — | 0.312 | no |
| ibp_clip | 23 | 0.835 / 0.841 | 0.999 | 0.000 | 0.000 | — | 0.312 | no |
| ibp_clip | 24 | 0.852 / 0.863 | 1.000 | 0.000 | 0.000 | — | 0.312 | no |

`FC16` is the share of `cert16` classes where GPT-2's decision is not constant over the 16 seen words. Against it, the
share of test classes that are not seen-constant is **0.312** (1 − 0.688).

## Reading (interpretation)

1. **The collapse is an optimisation instability, and gradient clipping removes it.**
   - Unclipped, 4 of 5 fits collapsed (with 7 of 10 in the probe, and 1 of 3 in #146).
   - Clipped, 0 of 5 collapsed, with faithfulness equal to the plain student's. #146's seed-mean failure was largely
     this instability.
2. **A stable student certifies exactly the trained word set, and nothing beyond it.**
   - `cert16` is about 1.000 on every clipped seed, and `cert23` is 0.000 on every clipped seed.
   - The held-out words' box is not certified on even one class.
   - In #146 and the unclipped arm, the held-out transfer (`cert23` 0.47–0.91 on healthy seeds) came with unstable
     dynamics. The probe shows it directly: one seed got 0.998 unclipped and 0.000 clipped.
   - *Why* it does not transfer was not measured. A plausible mechanism is a projection that pulls the 16 trained
     embeddings together but not the others. That is `open`.
3. **The stable student is certifiably blind to the seen words on every class, including the ones where GPT-2 is not.**
   - `FC16` is 0.3125 on four seeds and 0.312 on the fifth, the same as the non-seen-constant share. The student is
     certified on essentially every class, so about 31% of its seen-word certificates assert an invariance GPT-2
     lacks.
   - That costs no faithfulness (0.842 vs plain 0.839). The plain student does not track the word dependence either.
   - This is consistent with the dependence not being learnable at this student size and data. It was not measured
     directly; plain per-class constancy was not recorded.
4. **The consequence for the certified-student direction.** Stability is solved for this setup. What remains is the
   one #146 identified:
   - the certificate asserts invariance wherever training made it cheap, not where it is true of GPT-2;
   - and it does not reach unseen nuisance words.
   - Both are the conditional-certificate and product-spec questions. Neither is a stability question.

## Not claimed

- One stabiliser at one setting (clipping at 1.0), on one template, one nuisance position, and one student size.
- Nothing is claimed about other recipes reaching held-out words. That is **open**, not "impossible".
- The certificates prove the student's invariance (float32). Agreement with GPT-2 is `empirical`.
