# Conditional certificates: claim invariance only where a word-blind gate predicts GPT-2 is invariant — pre-registration

**Status: written 2026-10-06, committed and pushed before any code for this study existed.** Follows #146 and #147. The
stable (gradient-clipped) student certifies its decision over the 16 trained nuisance words on essentially every
class. So about 31% of its certificates assert an invariance GPT-2 lacks (`FC16` 0.312), and none reach the 7 held-out
words.

**The idea.** Keep the stable student, but **issue** a certificate only when a separate gate predicts that GPT-2 is
invariant on this sentence.
- The gate never sees the nuisance position: its input is the sequence with position 0 dropped. So its decision is
  invariant to the nuisance word **by construction**. The issued claim ("issued, and the student's decision is constant
  over the 16 words") is therefore sound whenever the IBP certificate is.
- Elsewhere the system answers without a certificate (it abstains from *certifying*, not from answering).
- What is measured is whether the issued certificates are **right about GPT-2**, and how many are issued.

**Disclosure: an exploratory probe was run before this pre-registration** (scratchpad; the #146 dump, stimulus 141). It
asked whether a word-blind gate can predict, on test classes, that GPT-2 is seen-constant (base rate 0.708).
- A small transformer gate (2 seeds) reached **AUC 0.783–0.788**. At coverage 0.25 / 0.50 / 0.75, the share of gated
  classes that are seen-constant was 0.933–0.957 / 0.890–0.893 / 0.797–0.812.
- Logistic regression on one-hot subject, verb and object did about as well (AUC 0.782; 0.927 / 0.875 / 0.810).
- Predicting all-23 constancy gave the same picture (AUC 0.784–0.791).
- The probe did not combine the gate with a certified student, and did not measure faithfulness among gated classes.

**The thresholds below were set after the probe.** It suggests H1 is likely to pass and H2 is borderline. It says
nothing about H3.

## 1. Data

- **Same template, 23 words and held-out set as #146** (the held-out words are Apparently, Eventually, Finally,
  Luckily, Naturally, Once, Today).
- **Fresh seeds:** stimulus **161**, split **162**, 4,000 σ, a 60 / 10 / 30 split. **The 10% validation split is used**
  (to set the gate's threshold).
- **Fit seeds {30, 31, 32, 33, 34}.**

## 2. System (fixed now)

- **Student:** #147's `ibp_clip` recipe, unchanged:
  - #146's architecture, IBP loss on seen-constant train classes, ramp, and weight 0.5;
  - gradient-norm clipping at 1.0;
  - the sorted softmax bound.
- **Gate:** the same architecture (frozen GPT-2 embeddings, a trained projection, 2 layers, `d = 128`, no LayerNorm)
  with a single logit.
  - Its input is the sequence with **position 0 removed**, and its target is GPT-2's seen-constancy on train classes.
  - Training: BCE, Adam lr 10⁻³, 3,000 steps, batch 256, with the gate seed equal to the fit seed. No early stopping.
- **Threshold τ (set on the validation split, never on test):** the lowest gate score τ such that, among validation
  classes with gate ≥ τ and `cert16`, **at least 0.95** are seen-constant, with at least **20** such classes. If no τ
  qualifies, nothing is issued for that seed.
- **Issued certificate:** gate ≥ τ **and** `cert16` (the sound IBP certificate over the 16 seen words).

## 3. Metrics (test classes)

- **coverage:** the share of test classes with an issued certificate.
- **`FCi`:** among issued classes, the share where GPT-2's decision is **not** constant over the 16 seen words
  (false certificates).
- **faithfulness among issued:** the share of issued classes where the certified decision equals GPT-2's on all 16
  seen words.
- **Reported, no hypothesis attached:**
  - the ungated stable student's `FC16`;
  - gate AUC for seen-constancy;
  - among issued classes, the share where GPT-2 is also constant over the 7 held-out words, and the share where the
    student's certified decision equals GPT-2's on them (extrapolation; **not certified**).
- **Soundness checks (an abort, as before):** certified classes must have concrete predictions equal to the certified
  decision over S, and the k = 1 box must be exact. In addition, the gate's output must be identical across all 23
  variants of a class (it is blind to position 0).

## 4. Decision rules (thresholds set after the disclosed probe)

- **H1 (honest certificates):** seed-mean **`FCi` ≤ 0.10**.
- **H2 (useful coverage):** seed-mean **coverage ≥ 0.20**.
- **H3 (issued certificates are faithful):** seed-mean **faithfulness among issued ≥ 0.90**.
- A seed with nothing issued counts as coverage 0, and is excluded from the `FCi` and H3 means. If no seed issues
  anything, H1 and H3 are `untestable`.

## 5. Not claimed

- The issued certificates are about the **student** (proved, float32, sound up to rounding) and the gate (invariant by
  construction). That they are right about GPT-2 is `empirical`, and holds only to the measured `FCi`.
- **Held-out words are not certified.** Their numbers are extrapolation.
- One template, one nuisance position, GPT-2 small, one gate design.
- The gate learns which sentences are nuisance-sensitive from labelled train classes. It is not a theory of why.

## 6. Protocol and artifacts

- Script `experiments/conditional_certificate.py` and tests, frozen (commit and sha256) before the real run.
- **`--smoke`:** stimulus 999, 200 σ, 300 steps. Smoke numbers are not results.
- The run is wrapped in `systemd-inhibit --what=sleep:idle`, **launched in the repo** (not a copy).
- Any change after smoke goes in a dated addendum, before the real run.
- **Outcome:** `docs/notes/conditional_certificate_outcome.md`, the summary JSON, and a table test. The outcome is
  edited only through a labelled post-review section.

---

## Addendum A (2026-10-06, after the smoke runs, before any real run): two implementation details, no design change

- **The gate's invariance check now compares inputs, not outputs.** As first written, it compared the gate's float
  outputs across the 23 variants. A unit test showed that identical inputs at different batch positions can differ in
  the last bits. The check now asserts that the gate's **integer input** (the sequence with position 0 removed) is
  identical across variants, and evaluates the gate once per class. This is the structural guarantee §2 relies on.
- **Smoke mode only:** the minimum issued count for τ is 5, not 20. The smoke run's validation split has 20 classes,
  so the issuing path could never run. With 5 it ran on 3 of 5 seeds. **The real run uses 20**, as pre-registered.
- **The smoke run** (stimulus 999, 200 σ, 300 steps) completed, and the soundness checks held. The numbers are poor at
  300 steps (student faithfulness 0.08–0.18), as in #146's and #147's smoke runs. They are not results.
- **Script frozen at `2002d8b`**, sha256 of `experiments/conditional_certificate.py` `52de3eee4c12ec5f…`. Check it with
  `git show 2002d8b:experiments/conditional_certificate.py | sha256sum`.
- Thresholds in §4 unchanged; no expected outcome beyond the disclosed probe.
