# Stable certified training: does gradient clipping remove the collapse, and what does a stable student certify? — pre-registration

**Status: written 2026-10-06, committed and pushed before any code for this study existed.** Follows #146
(`certified_student_*`), whose `ibp` arm collapsed on one of three seeds.

**Question.** #146's pre-registered verdict was the seed mean, collapse included. This study asks whether the collapse is
an optimisation instability that a standard fix removes, and what the student certifies once training is stable. It
does **not** address false certificates (a separate study, the conditional certificate).

**Disclosure: an exploratory probe was run before this pre-registration** (scratchpad; the #146 dump, stimulus 141;
fit seeds 10–19, disjoint from this study's). The training loop was #146's, with the RNG draw order changed slightly.
- **Original schedule** (#146: box ramp over the first half, weight 0.5, no clipping), 10 seeds:
  - 6 collapsed (test faithfulness 0.02–0.36), 1 partly collapsed (0.64), 3 healthy (0.80–0.82);
  - every seed shows loss spikes during the ramp, and the collapses are sudden (accuracy 0.84 → 0.02 within 250 steps);
  - `cert23` on the three healthy seeds: 0.998, 0.939, 0.000.
- **Gradient-norm clipping at 1.0**, schedule otherwise unchanged, 6 seeds (five of them collapsed unclipped):
  - **0 collapsed**: faithfulness 0.838–0.847 (the plain student's is about 0.84), with no loss spikes;
  - **`cert16` 0.998–1.000, but `cert23` 0.000 on all 6.** The stable student certifies exactly the trained word set
    and none of the held-out words.

**The thresholds below were set after seeing the probe, and the probe predicts H1–H3 pass and H4 fails.** This is a
confirmation on fresh data, not a test of an unseen effect.

## 1. Task, data, student: as #146 except the seeds

- **Same template, 23 words, held-out set** (Apparently, Eventually, Finally, Luckily, Naturally, Once, Today), student,
  frozen GPT-2 embeddings, IBP bound, certificates and metrics as `certified_student_prereg.md` §1–3.
- **Fresh seeds:** stimulus **151**, split **152**, 4,000 σ, a 60 / 10 / 30 split.
- **Fit seeds {20, 21, 22, 23, 24}** for every arm.

## 2. Arms (fixed now)

- **`plain`**: clean loss only (#146's plain arm).
- **`ibp`**: #146's IBP arm unchanged (ramp over the first 3,000 of 6,000 steps, weight 0.5). This is the baseline whose
  collapse rate is measured.
- **`ibp_clip`**: as `ibp`, plus `clip_grad_norm_(parameters, 1.0)` before every optimiser step.
- That is 3 arms × 5 seeds = **15 fits**.
- **One change to the bound's code (promised in #146's review):** the attention-softmax interval endpoints are sorted
  (`min`/`max`). This cannot change a result unless float rounding had crossed them. A test checks that the new bound
  equals #146's on random inputs.

## 3. Collapse rule and the new metric (fixed now)

- **Collapse:** an `ibp` or `ibp_clip` fit is *collapsed* if its test seen-word faithfulness is below the `plain` arm's
  seed mean minus **0.10**.
- **`FC16` (false certificates over the seen words):** among the `cert16` classes, the share where GPT-2's decision is
  not constant over the 16 seen words. Reported for every arm, with the test seen-constant rate alongside it.

## 4. Decision rules (thresholds set after the disclosed probe)

- **H1 (stability):** `ibp_clip` has **0 of 5** collapsed fits.
- **H2 (no faithfulness cost):** `ibp_clip` seed-mean seen-word faithfulness **≥ `plain` seed mean − 0.02**.
- **H3 (certification retained):** `ibp_clip` seed-mean **`cert16` ≥ 0.90**.
- **H4 (transfer to held-out words):** `ibp_clip` seed-mean **`cert23` ≥ 0.50**. The probe predicts this **fails**.
- **Reported, no hypothesis attached:**
  - the `ibp` arm's collapse count (the probe found 6 of 10, plus 1 partial);
  - `CF23` and `FC23` for every arm;
  - `FC16` against the seen-constant rate.

## 5. Not claimed

- One fix (clipping at 1.0) is tested. Other stabilisers (a longer ramp, a smaller weight, CROWN-IBP) are not.
- H1 passing would show this fix stabilises this setup, not that IBP training is stable in general.
- H4 failing would show the stable recipe does not transfer to held-out words; whether some other recipe can is
  **open**.
- False certificates are measured, not addressed. Certificates prove the student's invariance (float32, sound up to
  rounding); faithfulness to GPT-2 is `empirical`.

## 6. Protocol and artifacts

- Script `experiments/stable_student.py` and tests, frozen (commit and sha256) before the real run.
- **`--smoke`:** stimulus 999, 200 σ, 300 steps. Smoke numbers are not results.
- The run is wrapped in `systemd-inhibit --what=sleep:idle`.
- Any change after smoke goes in a dated addendum, before the real run.
- **Outcome:** `docs/notes/stable_student_outcome.md`, the summary JSON, and a table test. The outcome is edited only
  through a labelled post-review section.
