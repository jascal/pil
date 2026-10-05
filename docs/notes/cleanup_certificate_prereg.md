# The T6(b) clean-up certificate on GPT-2's decode input — pre-registration

**Status: written 2026-10-04, committed and pushed before any code for this study existed.**

**Context.**
- jascal/pil#135 / #136 / #138: the uniform substitution certificate covers ≤ 0.010 of test contexts for every
  trained substitute of GPT-2 small's decode input. #138 found that the hull ceiling does not explain this: GPT-2's
  own margins are a median 5–8% of the ceiling.
- jascal/i-orca#30 (`pic_core/PIC_Cleanup.thy`, kernel-checked) proves the T6(b) clean-up certificate. Code points
  are `x(σ) = W·T(σ) + b₀`, and the host residual is `u = x(σ) + n`.
  - Clean-up decodes `P(u − b₀)` with a linear left inverse `P` of `W` (`‖P y‖ ≤ K‖y‖`). It unbinds each role with a
    readout `w_s` (`⟨r_s, w_s⟩ = 1`), snaps to the nearest filler, and rebinds.
  - **Clean-up radius** `ρ(σ) = min_{s∈D(σ)} (γ_s/2 − κ_s(σ)) / (K·‖w_s‖)`: within it, clean-up returns `x(σ)`
    exactly (`cleanup_certified(1)`).
  - **Agreement radius** `β(σ) = min_{v≠t} m_v(σ) / ‖U_t − U_v‖`, with `t` and `m_v` the **code point's** own
    decision and margins: within it, the host decides `t` (`cleanup_certified(2)`).
  - The certificate: `‖u − x(σ)‖ < min(ρ(σ), β(σ))`.
- The certificate uses the **code point's** margin, not the host's. That is the one lever the uniform certificate
  lacks.

**The question.** Does the T6(b) certificate cover a meaningful share of contexts where the uniform one covers
almost none? If it does not, which radius binds: `ρ` (clean-up geometry) or `β` (code margins)?

## 1. Model, site, tasks, data

- GPT-2 small, frozen. The site is the decode input `u = ln_f(h_L)` at the final position; logits are `U u` (tied,
  no bias). The host decision is GPT-2's own argmax.
- **Tasks:** LIST and SVO, as in #136 (`certified_substitutes.py`: `build_rows`, the `TPR` family, `train`).
- **Fresh seeds:** stimulus **41**, split **42**, fit seeds **{0, 1, 2}**. 12,000 contexts per task, split
  60 / 10 / 30. Test contexts with an unseen (filler, role) pair are dropped and counted.

## 2. Substitutes and cells

Only the **TPR** family is used, because clean-up is defined on its structure. T6(b) needs a left inverse of `W`, so
the tensor dimension `d_F · n_role` must be ≤ 768:

| setting | `d_F` | tensor dim | in study |
|---|---:|---:|---|
| LIST | 8 | 240 | yes |
| LIST | 32 | 960 | **no**: `W` has no left inverse |
| SVO | 8 | 24 | yes |
| SVO | 32 | 96 | yes |

**Objectives** (Adam, lr 3e-3, 3,000 steps, batch 512):
- **`mse`** and **`cert`**: exactly as in #136.
- **`t6`** (new): `0.1·‖x − u‖² + relu(‖u − x‖ − β(x) + 0.1)`, averaged over the batch.
  - `β(x) = min_{v≠t} ⟨x, U_t − U_v⟩ / ‖U_t − U_v‖` over the **full** vocabulary, with `t` the host decision.
  - The hinge pushes each code point to sit deeper inside the host's decision cell than its distance to `u`.

So there are 3 settings × 3 objectives = **9 cells**, each with 3 fit seeds.

## 3. The certificate, computed per test context

For a fitted TPR (`ef`, `er`, `W`, `b₀`):
- **`P` = `W⁺`** (Moore–Penrose), and `K = 1 / σ_min(W)` (float64 SVD).
  - The cell is flagged **no-left-inverse** if `σ_min(W) / σ_max(W) < 10⁻⁸`, or if `‖P W − I‖_max > 10⁻⁸`.
- **Role readouts: the dual basis of `er`,** `w_s = (er⁻ᵀ)_s`, so `⟨r_t, w_s⟩ = δ_ts` and every crosstalk
  `κ_s = 0`. The cell is flagged **no-dual-readout** if `cond(er) > 10⁸`.
- **`F_s`** = the fillers seen in role `s` in train. **`γ_s`** = the minimum distance between distinct embeddings in
  `F_s`.
- `ρ(σ) = min_{s∈D(σ)} γ_s / (2·K·‖w_s‖)`; `β(σ)` from the code point's own logits over all 50,257 tokens;
  `‖n‖ = ‖u − x(σ)‖`.
- **Verdict:** `certified` iff `‖n‖ < min(ρ, β)`. This is a **Soufflé** query over fixed-point facts (scale 2³⁰),
  with `‖n‖` rounded up, `ρ` and `β` rounded down, and slack 10⁻⁶. A Python twin is used for parity tests.

**Soundness checks (any failure aborts the run; nothing is reported as a result).** Clean-up is actually run on every
test context: unbind `P(u − b₀)` with `w_s`, take the nearest filler in `F_s`, and rebind. Every certified context
must satisfy both of the following:
- (i) the recovered structure equals `σ` on every bound role;
- (ii) the host's argmax equals the code point's argmax.

The theorem guarantees both.

## 4. Metrics (test split, seed mean and range per cell)

- **Primary: T6 coverage**, the fraction of test contexts certified.
- **Secondary:**
  - **uniform coverage** of the plain substitute `x(σ)` (the #136 checker), as the comparison;
  - **exact clean-up rate**, the fraction of contexts where clean-up recovers `σ`, whether or not certified;
  - **cleaned agreement**, the fraction where the code point's decision equals the host's;
  - **the bottleneck split** among uncertified contexts: `R` = the share with `‖n‖ ≥ ρ`, `B` = the share with
    `‖n‖ ≥ β`;
  - medians of `‖n‖`, `ρ`, `β`.

## 5. Decision rules (fixed now)

- **H1: the clean-up certificate covers what the uniform one cannot.** In a setting, take the best objective by
  seed-mean T6 coverage. H1 passes in that setting if that coverage is **≥ 0.05** and **≥ 10×** the same cell's
  uniform coverage. **Overall: yes** in ≥ 2 of the 3 settings; **partial** in 1; **no** in 0.
- **H2: the T6 objective matters.** `t6` beats both `mse` and `cert` on seed-mean T6 coverage by **≥ 0.03**, in ≥ 2
  of 3 settings.
- **Bottleneck label, per cell** (descriptive; from seed-mean `R` and `B`):
  - **ρ-limited** if `R ≥ 0.8` and `B < 0.8`;
  - **β-limited** if `B ≥ 0.8` and `R < 0.8`;
  - **both** if both are ≥ 0.8;
  - **mixed** otherwise.
- **Flagged cells.** A cell flagged no-left-inverse or no-dual-readout in any seed counts as T6 coverage 0 for that
  seed. It is reported, never re-fitted.

**No expected outcome is stated.** (#138's addendum B withdrew a prediction made from smoke data, and this study
makes none.)

## 6. Not claimed

- The certificate is per context: it needs `u`. It certifies a whole **ball** around `x(σ)`, so it is robust to
  perturbations of `u` within that ball, but nothing extends to contexts whose `u` is not checked.
- A clean-up certificate says the host and the cleaned code point agree. It does not say GPT-2 *computes* a TPR.
- GPT-2 small, two templated tasks, the TPR family only. LIST at `d_F = 32` is out of scope (no left inverse).

## 7. Protocol and artifacts

- Script `experiments/cleanup_certificate.py`, committed (frozen) before the real run.
- `--smoke`: stimulus seed 999, 600 contexts, 30 training steps. Smoke numbers are not results.
- **Any change after a smoke run goes in a dated addendum to this file, before the real run, and discloses what the
  smoke run showed.**
- Outcome: `docs/notes/cleanup_certificate_outcome.md`, plus the run summary JSON next to it. Neither file is edited
  after the first real run.
