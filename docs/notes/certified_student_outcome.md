# Certifiable by construction: a certified-trained student with the right invariances — outcome

**Pre-registration:** [`certified_student_prereg.md`](certified_student_prereg.md), with addendum A.
- Provenance, in order:
  - pre-registration `7b4dbaf`, pushed before any code, disclosing an exploratory probe; **the thresholds were set
    after the probe**;
  - addendum A `cb52556` (smoke, plus a full-length smoke diagnostic, disclosed; no design change);
  - script and tests frozen at `22862e9` (`experiments/certified_student.py`, sha256 `4995f25a…`), and the run used it
    unchanged.
- Seeds: stimulus 141, split 142, fit {0, 1, 2}. 4,000 σ-classes × 23 nuisance words; class split 2,400 / 400 /
  1,200; 107 outputs. GPT-2's decision is constant across all 23 words in **69.9%** of test classes.
- The run was held under `systemd-inhibit` and was not interrupted (907 s).
- Summary: [`certified_student_summary.json`](certified_student_summary.json).
- Tags: each certificate is **proved** for that student (a sound interval bound). Every count is `empirical`.

## Verdict

**H1 (`CF23 ≥ 0.50`): FAIL** (0.328). **H2 (`FC23 ≤ 0.10`): FAIL** (0.513). **H3 (held-out-word faithfulness among
certified classes ≥ 0.90): FAIL** (0.532).

- **Soundness:** every certified class's concrete predictions matched its certified decision; the k = 1 box was exact.
  No abort fired.

## Per arm and seed (test classes)

| arm | seed | faithfulness (seen / held-out words) | `cert16` | `cert23` | `CF23` | `FC23` | held-out faithfulness among `cert23` |
|---|---:|---|---:|---:|---:|---:|---:|
| plain | 0 | 0.839 / 0.852 | 0.000 | 0.000 | 0.000 | — | — |
| plain | 1 | 0.840 / 0.846 | 0.000 | 0.000 | 0.000 | — | — |
| plain | 2 | 0.842 / 0.853 | 0.000 | 0.000 | 0.000 | — | — |
| ibp | 0 | 0.024 / 0.025 | 1.000 | 1.000 | 0.018 | 0.982 | 0.020 |
| ibp | 1 | 0.820 / 0.830 | 0.988 | 0.469 | 0.359 | 0.234 | 0.833 |
| ibp | 2 | 0.815 / 0.821 | 0.995 | 0.895 | 0.607 | 0.322 | 0.743 |

`CF23` is certified over all 23 words and faithful to GPT-2 on all 23. `FC23` is the share of certified classes where
GPT-2's decision differs from the certified one on some word.

## Reading (interpretation)

1. **Certified training is unstable here.** `ibp` seed 0 collapsed to an almost constant output. It is trivially
   certifiable (certified on 100% of classes) and almost never faithful (0.024). The full-length smoke diagnostic in
   addendum A did not collapse, so this is seed-dependent, not systematic.
2. **Even when it trains, the student learns invariance everywhere, not where GPT-2 has it.**
   - The non-collapsed seeds are certified over the 16 seen words on **98.8–99.5%** of test classes. Yet GPT-2's
     decision is constant across all 23 words in only 69.9% of them.
   - Those seeds were trained with the clean loss on every variant, including classes where GPT-2's decision depends
     on the seen words. The interval pressure still overrode that signal.
   - The result is a false-certificate rate of **23–32%** of certified classes.
3. **What the certificates are worth.**
   - They are sound statements about the student: across all 23 words, including 7 it never saw, its decision provably
     does not change.
   - As claims about GPT-2 they are wrong in about a quarter to a third of the cases where they are issued.
   - Agreement with GPT-2 costs only 2–3 points against the plain student (0.82 vs 0.84) when training does not
     collapse.
4. **The probe's failure mode survives the redesign.** Training on all classes did not teach the student *where*
   to be invariant.

## What this suggests (interpretation, not tested here)

- **If the nuisance really is meant to be irrelevant** (a product spec that time adverbs must not change the decision),
  the certified invariance is the *desired* property, and GPT-2's sensitivity is noise, not ground truth. Then the
  right metric is faithfulness on GPT-2-invariant classes plus the certified rate. By that reading the non-collapsed
  seeds do well: certified everywhere, about 0.82 agreement.
- **If the student must match GPT-2's sensitivity,** it needs a way to certify **conditionally**: learn which classes
  are invariant (a gate or abstention head) and certify only those. Or it needs an architecture that separates the
  nuisance pathway, so interval pressure cannot leak into the decision.
- **Stability** needs the standard certified-training remedies (a longer ramp, CROWN-IBP-style mixing, a smaller
  weight). Any of these is a new pre-registration.

## Not claimed

- One nuisance position, one template, GPT-2 small as the teacher, a 2-layer student, one IBP weight and schedule.
