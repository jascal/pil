# How much could a model-side bound certify? The nuisance-class hull ceiling — pre-registration

**Status: written 2026-10-05, committed and pushed before any code for this study existed.**

**Context.**
- jascal/pil#141 / #142: per-context certificates (`nearest_iff_halfspace` for clean-up, plus the pairwise iff for
  host agreement) certify about two-thirds of SVO contexts. Every one needs the observed residual `u`.
- To certify a context **the host has not been run on**, a model-side bound must cover a *set* of contexts. Regions
  certified for different structures `σ ≠ σ′` are disjoint. So a set-level bound can only help when many contexts
  share one `σ`, that is, when contexts contain **nuisance** tokens that `σ` does not encode.
- **Key fact:**
  - The clean-up condition (every half-space slack > 0) and the agreement condition (`⟨u, U_t − U_v⟩ > 0` for every
    `v`) are **strict linear inequalities** in `u`.
  - They hold on the convex hull of a class's residuals **iff** they hold at every residual.
  - So the share of classes in which *every* variant is certified is the exact **ceiling** for any sound set-level
    bound over that class, whatever the propagation method.
- **Disclosure: exploratory measurements were made before this pre-registration.** They used scratchpad data only,
  with stimulus seed 81 and seeds disjoint from this study's.
  - A first nuisance choice (adjectives) failed. GPT-2's decision changed with the adjective in 100% of classes,
    because the adjectives are copied into the prediction.
  - Sentence-initial time phrases (below) gave hull ceilings of about 0.19 over all held-out classes, and about 0.28
    over classes whose host decision is constant, for `d_F=32` `t6` (2 seeds; `mse` 0.00–0.14).
  - **The thresholds in §4 were set after seeing those numbers.**

**The question.** For held-out σ-classes of nuisance variants, how often is the whole class certified (the hull
ceiling)? That is the most any model-side bound could certify without running the host on each variant.

## 1. Model, template, data

- GPT-2 small; decode input at the final position; logits `U u` (tied, no bias).
- **Template:** `"{P} the {S} {V} the {O}. The {O} was {V} by the"`, with
  `P ∈ {Yesterday, | On Monday, | Last week, | This morning, | In the evening, | Once again, | Later that day, | At noon,}`.
  S, V and O are the 40 occupations and 16 verbs of #136, with S ≠ O.
- **σ = (S, V, O)**; P is nuisance (not encoded in σ). A **σ-class** is the 8 variants of one σ.
- **Fresh seeds:** stimulus **91** draws 1,200 σ (from all 24,960), giving 9,600 contexts. Split seed **92** splits
  the **classes** 60 / 10 / 30 into 720 train, 120 validation and 360 test classes. Fit seeds are **{0, 1, 2}**.

## 2. Substitutes

- TPR `d_F=32` (the setting that certifies; `d_F=8` certified ≤ 0.04 in #139–#142). Objectives `mse`, `cert`, `t6`
  (#136/#139 code, unchanged). That is 9 fits.
- Each fit is trained on every variant of the train classes. The substitute sees `σ` only, so it maps all 8 variants
  of a class to one code point `x(σ)`.

## 3. Certificates and the ceiling

- **Per context:** #141's certificate (`halfspace_verdict.py`, reused unchanged). These are Soufflé verdicts over
  fixed-point facts with a Python twin, and the same iff and neighbourhood aborts.
- **Class (hull) certificate:** `hull(c)` holds iff `full` holds for all 8 variants of class `c`. By linearity, every
  convex combination of the class's residuals is then certified too. **A kernel lemma stating this** (strict linear
  inequalities are preserved under convex combination) **will be added to i-orca and merged before the real run.**
- **Hull sanity check (aborts the run):** for every hull-certified class, 16 random convex combinations of its 8
  residuals must have exact clean-up and the code point's decision.
- **Decision-constant classes:** classes in which the host's argmax is the same for all 8 variants. Only these can be
  hull-certified, since `full` requires host = code decision for every variant.

## 4. Metrics and decision rules (thresholds set after the disclosed exploration)

- **Primary:** `H_all`, the share of test classes that are hull-certified (seed mean per objective).
- **Secondary:**
  - `H_const`, the hull-certified share among decision-constant test classes;
  - the share of test classes that are decision-constant;
  - per-context `F_cov`;
  - the median within-class residual radius;
  - median `ρ_loc` among certified contexts.
- **H1 (non-trivial ceiling):** `H_all ≥ 0.10` for at least one objective.
- **H2 (ceiling on decision-stable classes):** `H_const ≥ 0.20` for at least one objective.
- **No expected outcome beyond the disclosed exploration is stated.**

## 5. Not claimed

- **No bound is computed here.** The ceiling is what a perfectly tight sound bound would certify. Whether interval or
  linear-relaxation propagation through GPT-2 gets close is study (2), not this one.
- The nuisance is one specific set of 8 time phrases. Other nuisances (for example adjectives) can change the host's
  decision.
- GPT-2 small, one template, the TPR family only.

## 6. Protocol and artifacts

- The i-orca convexity lemma is merged before the real run.
- Script `experiments/nuisance_hull.py`, frozen before the real run.
- `--smoke`: stimulus 999, 80 classes, 30 steps. Smoke numbers are not results.
- The run is wrapped in `systemd-inhibit --what=sleep:idle`.
- Any change after smoke goes in a dated addendum, before the real run.
- Outcome: `docs/notes/nuisance_hull_outcome.md` plus the summary JSON. Neither is edited after the first real run
  except through a labelled post-review section.

---

## Addendum A (2026-10-05, after a smoke run, before any real run): disclosure only, no design change

- **What was run:** the `--smoke` pipeline (stimulus 999, 80 classes, 640 contexts, 30 steps) for all 9 fits, under
  `systemd-inhibit`. Seeds 91/92 are untouched.
- **Result:** it completed, with 0 ties.
  - The host decision was constant in 71% of smoke test classes.
  - `F_cov` = 0 and `H_all` = 0, since 30-step fits are undertrained. So the hull check ran on 0 classes on smoke data;
    the unit tests exercise it, including the abort.
- **The design is unchanged.**
