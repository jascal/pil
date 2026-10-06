# Certifiable by construction: a certified-trained student with the right invariances — pre-registration

**Status: written 2026-10-06, committed and pushed before any code for this study existed.**

**Framing.** The student is the **product**. Its decisions carry proved properties, here invariance to a declared
nuisance token. Agreement with GPT-2 is a distillation-quality metric, **not** a certificate. A certificate proves
something about the student. Whether that property is *right* (GPT-2 really is invariant there) is measured.

**Disclosure: an exploratory probe was run before this pre-registration** (scratchpad only; stimulus seed 121; one
seed; seeds disjoint from this study's). Same task, a student trained only on classes where GPT-2's decision is
constant:
- interval-bound (IBP) training raised the certified share of held-out classes from 0.00 to 0.93, with agreement
  0.930 vs 0.947;
- but on classes where GPT-2's decision **depends** on the nuisance word, the IBP student was certifiably constant on
  0.996 of them, at 0.51 agreement. It certified a wrong invariance.

This study's design (training on all classes, a false-certificate metric) responds to that. **The thresholds below
were set after seeing the probe.**

## 1. Task and data

- **Template:** `"{W}, the {S} {V} the {O}. The {O} was {V} by the"`. W is a sentence-initial nuisance word; S, V, O
  are the 40 occupations and 16 verbs of #136, with S ≠ O. All 23 words are single tokens, so variants are
  position-aligned (as in #144).
- **Nuisance words:** 23. **Held out from all training (fixed by seed 133):** Apparently, Eventually, Finally,
  Luckily, Naturally, Once, Today. The other **16 are seen** in training.
- **Labels:** GPT-2 small's next-token argmax for every (σ, W).
- **Classes:** a class is one σ with its 23 variants. **Stimulus seed 141** draws 4,000 σ. **Split seed 142** splits the
  classes 60 / 10 / 30.
  - A class is *seen-constant* if GPT-2's decision is the same across the 16 seen words, and *all-constant* across all
    23.

## 2. Student and training (fixed now)

- **Student:** a 2-layer causal transformer, `d = 128`, 4 heads, ReLU MLP (4d), **no LayerNorm**, learned positions,
  and a linear head over the output set (GPT-2's decisions on train variants).
- **Input embeddings:** GPT-2's **frozen** token embeddings (768-d) followed by a trained linear projection to `d`.
  So a held-out word still has a meaningful embedding.
- **Training uses train classes and seen words only:**
  - **clean loss:** cross-entropy on every variant (all train classes, all 16 seen words);
  - **IBP loss** (IBP students only): worst-case cross-entropy over the interval box spanned by the 16 seen words'
    projected embeddings at position 0, on **seen-constant** train classes only. The box is ramped from 0 to full over
    the first half of training. The weight is 0.5: `loss = (1 − 0.5α)·clean + 0.5α·IBP`.
- Adam, lr 10⁻³, 6,000 steps, batch 256; fit seeds **{0, 1, 2}**.
- **Arms:** `plain` (clean loss only) and `ibp`. That is 2 arms × 3 seeds = **6 fits**.

## 3. Certificates and metrics (test classes)

- **The certificate (sound IBP):** class `c` is certified over a word set S if, on the box spanned by S's projected
  embeddings, the lower bound of the student's predicted logit exceeds every other logit's upper bound. Reported for
  `S = seen-16` and `S = all-23`.
- **Faithfulness:** the student's prediction equals GPT-2's decision.
- **Metrics:**
  - per-variant faithfulness on seen and on held-out words;
  - `cert16`, `cert23`;
  - **`CF23` (certified and faithful):** certified over all 23, and the certified decision equals GPT-2's decision on
    all 23 variants;
  - **`FC23` (false-certificate rate):** among `cert23` classes, the share where GPT-2's decision differs on some
    variant from the certified one;
  - **held-out-word faithfulness** among `cert23` classes: GPT-2 agrees with the certified decision on the 7 held-out
    words.

**Soundness checks (any failure aborts the run):**
- every class certified over S must have the student's concrete prediction equal to the certified decision for every
  word in S;
- with |S| = 1 the box is exact (zero width).

## 4. Decision rules (thresholds set after the disclosed probe)

- **H1 (certifiable and faithful):** for the `ibp` arm, seed-mean **`CF23` ≥ 0.50** of test classes.
- **H2 (honest certificates):** for the `ibp` arm, seed-mean **`FC23` ≤ 0.10**.
- **H3 (unseen nuisance words):** for the `ibp` arm, among `cert23` classes, seed-mean held-out-word faithfulness
  **≥ 0.90**.
- The `plain` arm is reported as the baseline; no hypothesis is attached to it.
- No expected outcome beyond the disclosed probe is stated.

## 5. Not claimed

- Certificates prove the **student's** invariance only. Faithfulness to GPT-2 is `empirical`.
- One nuisance position, one template, GPT-2 small as the teacher, a 2-layer student.

## 6. Protocol and artifacts

- Script `experiments/certified_student.py` and tests, frozen before the real run.
- `--smoke`: stimulus 999, 200 σ, 300 steps. Smoke numbers are not results.
- The run is wrapped in `systemd-inhibit --what=sleep:idle`.
- Any change after smoke goes in a dated addendum, before the real run.
- Outcome: `docs/notes/certified_student_outcome.md`, the summary JSON, and a table test. The outcome is edited only
  through a labelled post-review section.

---

## Addendum A (2026-10-06, after smoke runs, before any real run): disclosure only, no design change

- **The smoke run** (stimulus 999, 200 σ, 300 steps, all 6 fits) completed, and the soundness checks held. The `ibp`
  arm **collapsed** (faithfulness 0.03–0.06): with 300 steps the box reaches full size within 150 steps.
- **To check whether this was a design flaw** (the shared frozen-embedding projection must blur the nuisance words
  while keeping the fillers apart), one full-length fit (6,000 steps) per arm was run on the **smoke data**: seed 0,
  120 train classes, 60 test.
  - **It did not collapse:** `plain` gave faith 0.629 and `cert23` 0.000; `ibp` gave faith 0.657, `cert23` 0.850,
    `CF23` 0.533, `FC23` 0.373, and held-out-word faithfulness among certified classes 0.647.
  - The collapse was a 300-step artifact, so **the design is unchanged.**
- **Disclosure:** these are one fit each on a tiny non-study dataset; seeds 141/142 are untouched. **No expected
  outcome is stated,** and the thresholds in §4 are unchanged.
