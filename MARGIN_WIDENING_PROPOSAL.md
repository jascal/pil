# Margin-widening loop: training PIL frames for certified quantizability

**Train the decode geometry so the downstream certified-quant bit-width drops — by targeting the
*scale-free normalized margin* (the actual certificate quantity), tuned by a margin-vs-fit weight, and
paying a *provable* capacity price.** Closes the measure→loss→learn loop with fieldrun: fieldrun measures
where the margin certificate binds, PIL learns frames that widen it, fieldrun re-certifies more dropped
bits.

*Status: proposal + synthetic PoC (`experiments/margin_widening_loop.py`) + the **decisive real-model
sweep** (`experiments/margin_widening_realdata.py`, both with `…RESULTS.txt`). **The §8 sweep is done and
it revised the conclusion:** the normalized term is refuted (real `‖r‖` is norm-pinned), the raw-margin
knee survives and ships, and the deliverable is the certified-bit headroom map. Certificate: i-orca
`PIC_Quant` (`frame_quant_logit_bound`, `quant_decode_preserved`) + capacity ceiling `DecodeCapacity.thy`
(`geometry.log10_packing_bound`). Downstream: fieldrun
[`CERTIFIED_QUANT_PROPOSAL.md`](../fieldrun/CERTIFIED_QUANT_PROPOSAL.md) §10 / v1.5. Read §8/§9 first — they
supersede the original normalized-margin framing below. Nothing in PIL's shipped behaviour changes.*

---

## Current status & go/no-go

> **Recommended immediate action (updated after the §8 real-model sweep).** Ship the **raw-margin knee**
> — a margin auxiliary at a small target, λ≈0.5–1.0 — as an opt-in regularizer (it reproduces as a free
> lunch on real residuals across architectures, §8). **Do NOT build the normalized (`widen_t`) term:** the
> decisive sweep (§8) **refuted** its premise — the model's final norm pins the read-out `‖r‖ ≈ const` on
> every real architecture, so `γ̃ ∝ margin` and `widen_t ≡ raw`. The remaining deliverable is the **certified-
> bit headroom map** (size × arch × domain × language, §8) — which, like τ*, shows *no universal law*.

**What this is, today:** a proposal *and* a PoC that **measured the margin regimes on synthetic data
before recommending any PIL training change** — the same measure-before-build discipline as fieldrun's
Step 0. The PoC answers three things: (1) is the *normalized* margin a better target than the raw margin
PIL already hinges on? (2) what does the opposite regime (narrowing margins) do? (3) is there a
margin-vs-fit hyperparameter with a usable sweet spot?

| question | PoC verdict (synthetic, 8× over-complete clustered substrate) |
|---|---|
| naive normalized hinge `relu(γ − margin/‖r‖)` | **harmful** — its `1/‖r‖` gradient down-weights the binding (high-‖r‖) positions; −1.38 bits vs raw |
| principled `widen_t` (per-position target `γ·‖r‖`, detached) | **≈ raw** on homogeneous ‖r‖ (−0.04 bits) — *no synthetic free lunch; the win needs ‖r‖-heterogeneity* |
| `narrow` (squeeze margins down) | **quant-brittle + slow** — binding margin→0, top1 0.99→0.87, slower descent |
| margin-vs-fit knob (λ × small target γ) | **free-lunch knee** — at γ≈0.15, raising λ lifts certified margin ~13× and top1, *lowers* nll & speeds descent; cost only at large γ |
| does the widen win hold across sizes? | **no — it is scale-dependent and flips sign**: +1.10 bits at dim=16, ≈0 mid, −0.5…−0.7 bits at high over-completeness/large dim (no universal law) |
| does capacity bind? | **yes, as over-completeness grows**: γ-code coverage `code/V` falls 0.96→0.49 by 32× — the frame can't separate all V tokens, and that is where the widen advantage goes negative |

**Go/no-go — RESOLVED by §8.** The decisive size × architecture sweep on real `--source-dump` data is
**done** (§8), and it splits the proposal cleanly:
- **Ship:** the **raw-margin knee** — a free lunch that reproduces on real residuals across architectures.
- **Drop:** the **normalized (`widen_t`) term** — *refuted*. On all 9 real cells the final norm pins
  `‖r‖` (cv 0.003–0.11 ≪ 0.35), so normalization is a no-op (matched-target Δbits = −0.08).
- **Keep as the deliverable:** the **certified-bit headroom map** (§8) — size/arch/domain/language-
  dependent, no universal law (Pythia's quantizability improves with scale, Qwen's is flat — the τ* shape).
- **Interpretability:** margin-widening is a **select-side** knob; it does *not* shift retrieve/compose
  (that lives in the sources, reachable only by whole-model training). See §8.

---

## 0. The loop

Downstream (fieldrun v1.5, kernel-checked in i-orca `PIC_Quant`) a quantized read-out preserves the decode
at position `x` iff the **margin certificate** holds:

> `2·δ(x) < margin(x)`,  with read-out distortion `δ(x) ≈ c·2⁻ᵇ/√d · ‖r(x)‖` (TurboQuant; measured tight in v1.5).

So the bits are **margin-gated**: the smallest-margin ("forge-tax") positions cap how far you can
quantize. fieldrun *measures* that cap; it cannot move it. **PIL can** — it learns the frame `{U_v}` that
sets the margins. The loop:

```
fieldrun --source-dump  ─▶  margins, residuals r, certified bits b*(x)   (measure where the cert binds)
        │                                                                        ▲
        ▼                                                                        │
PIL: learn U to widen the *certified* margin  ─────────────────────────▶  re-certify: more bits dropped?
```

## 1. The certificate → the right training target

PIC logit `L(v) = Σⱼ⟨dⱼ,U_v⟩ + b_v`; margin `= L_t − max_{v≠t} L_v`. Invert the certificate for the
**minimum certified bits** at `x`:

> `b*(x) = log₂( 2c·‖r(x)‖ / (√d · margin(x)) )  =  const(c,d) − log₂( γ̃(x) )`,
> where **`γ̃(x) = margin(x) / ‖r(x)‖`** is the **scale-free normalized margin**.

Two consequences, both load-bearing:

1. **Lower `b*` ⇔ larger `γ̃`.** The thing to maximize for quantizability is `γ̃`, *not* raw margin.
2. **`γ̃` is the right target precisely because it is scale-free.** PIL fixes `‖U_v‖=1` (`normalize_U`)
   but `‖r‖` is free; raw margin can be inflated by growing `‖r‖` — which grows `δ` in lockstep, **no
   certified-bit gain**. `γ̃` is invariant to that rescaling (both `margin` and `‖r‖` scale together), so
   it is exactly the part of the geometry that controls the certificate. (And `c` cancels in any
   *comparison*: bits saved `= log₂(γ̃_A/γ̃_B)` is `c`-free — the PoC's reporting unit.)

PIL today hinges on **raw** margin (`learner.py`: `relu(margin_target − m_worst)`, absolute
`margin_target`). The proposal: target `γ̃`.

## 2. The objective — and a gradient pitfall the PoC caught

The obvious form — divide the live margin by `‖r‖` — is **wrong**:

```
naive (widen):  relu(γ − margin/‖r‖)        # gradient ∝ 1/‖r‖  →  DOWN-weights the high-‖r‖ binding positions
```

`r = Σⱼ dⱼ` is data (independent of `U`), so `1/‖r‖` is a per-position gradient scale that *de-emphasises
exactly the diffuse, high-‖r‖, forge-tax positions the certificate binds on*. The PoC confirms it back-
fires (−1.38 bits, top1 0.95). The correct form treats `‖r‖` as a per-position **target shift**, detached
from the gradient:

```
principled (widen_t):  relu( γ·‖r‖.detach() − margin )   # unscaled ∂margin/∂U; bar rises where ‖r‖ is large
```

This is a `‖r‖`-weighted margin target: it demands *more* absolute margin where the residual is diffuse —
the certbit-binding positions — without distorting the gradient. On homogeneous `‖r‖` it reduces to the
raw hinge (which is why the PoC finds `widen_t ≈ raw` there); on heterogeneous `‖r‖` it diverges in favour
of the binding positions. (Even tighter: hinge directly on `b*(x)`; equivalent to first order.)

## 3. The margin-vs-fit knob (the phase diagram)

The trade is governed by **two** hyperparameters already in `PILConfig`: `margin_weight` (λ, the weight of
the margin term against the NLL/fit goal) and the **target** γ (a *small but meaningful* normalized
margin). The PoC sweeps λ × γ for `widen_t` and reports the fit goal (final `nll`), the learning rate
(`t90` = steps to 90 % of final top-1), top-1, and the certified payoff (binding `nm_p10`):

```
 m_weight  target     nll     t90    top1  nm_p10        (synthetic; seeds=2, 600 steps)
     0.00    0.15   1.900   187     0.916   0.011        ← pure NLL: quant-brittle (binding margin ≈ 0)
     0.25    0.15   1.898   175     0.981   0.088
     0.50    0.15   1.897   150     0.997   0.122        ← free-lunch knee: nll at floor, top1↑, faster, cert↑
     1.00    0.15   1.896   150     1.000   0.143
     2.00    0.30   2.103   175     0.998   0.200        ← tradeoff regime: cert↑ but nll +11 % (fit cost)
     2.00    0.50   2.142   175     0.993   0.153        ← over-separation: fit cost without extra cert
```

**Reading it:** at a *small* target (γ≈0.15) the margin term is an **aligned auxiliary regularizer** —
raising λ lifts the binding certified margin ~13× (0.011→0.143), raises top-1 (0.916→1.000), keeps `nll`
at its floor, and *speeds* convergence (t90 187→150). Separating classes and fitting them are not in
conflict here. The genuine **fit cost appears only at large γ** (0.30–0.50: `nll` climbs 11–13 %), where
the objective demands *over*-separation the likelihood does not want. So "a small but meaningful margin" is
not a heuristic — it is the measured knee, and λ,γ are the dials. (Raw shows the same shape: `nll` 1.895 at
target 1.0, rising to 1.96 by target 4.0.)

**The knee vs. the baselines** (the shippable free lunch, at a glance):

| config | nll ↓ (fit) | top1 ↑ | t90 ↓ (speed) | binding nm_p10 ↑ (cert) | |
|---|---|---|---|---|---|
| pure NLL (λ=0, no margin term) | 1.900 | 0.916 | 187 | 0.011 | quant-brittle |
| raw hinge (target 2.0, λ=0.5) | 1.949 | 0.993 | 175 | 0.126 | PIL today |
| **knee: `widen_t` γ=0.15, λ=1.0** | **1.896** | **1.000** | **150** | **0.143** | **free lunch** |

vs no-margin training the knee is a strict win on *every* axis (≈13× certified margin, +8 pts top1,
−20 % steps, lower nll); vs the raw hinge it matches on this homogeneous-‖r‖ substrate (the §2/§6 gate) at
lower nll and faster descent.

## 4. The provable price — capacity & dynamics

Margin-widening is **not** unconditionally free; it has a kernel-proved ceiling. The decode-side
packing/separation bound (i-orca `DecodeCapacity.thy`, `geometry.log10_packing_bound`):

> `N_γ̃ ≤ (1 + 2ρ/γ̃)^d` — the number of `γ̃`-separated propositions in `ℝ^d`.

Widening `γ̃` *reduces the number of distinguishable tokens* at fixed `d` — eventually you must trade
vocabulary/capacity or add dimension. **In the PoC this ceiling is dormant** (`log₁₀ N_γ̃ ≈ 14.4 ≫
V=192`), so the synthetic regime never pays the capacity tax — an honest negative: the capacity price
would only show near the packing limit (large `V/d`, real models). Two dynamics findings:

- **`widen_t` tracks `raw` step-for-step** (identical descent curve, final top1 1.00) — the principled
  widening costs nothing in convergence.
- **`narrow` is slower *and* caps lower** (top1 0.58→0.90, never reaches 1.0; binding margin → 0): squeezing
  margins fights the fit and yields a quant-fragile frame. The regime axis is directional.

## 4a. Size & architecture dependence — there is no universal law

Sweeping `(dim × over-completeness)` shows the widen advantage is **not scale-invariant** — it flips sign,
and the capacity price (§4) activates as the frame crowds:

```
   size      V  ceil_log10  raw nm10 | wt nm10 | dtop1  dbits  code/V      (knee gamma=0.15 vs raw t=2.0)
   16x8    128        9.6      0.02       0.04  +0.026 +1.10   0.89   ← small dim: widen_t WINS (+1.1 bits)
   24x4     96       14.4      0.10       0.11  +0.010 +0.12   0.96
   24x8    192       14.4      0.11       0.13  +0.002 +0.15   0.91   ← neutral crossover
  24x16    384       14.4      0.17       0.15  +0.002 −0.25   0.73   ← capacity starts to bind (code/V↓)
  24x32    768       14.4      0.22       0.16  +0.002 −0.52   0.49   ← 51% of tokens can't reach gamma=0.2
   48x8    384       28.9      0.32       0.20  +0.000 −0.68   0.73   ← large dim: raw wins
```

**Reading it:** at **small dim / tight packing** (16×8) the principled normalized target buys a full extra
certified bit over raw (+1.10); around 8× it is neutral; at **high over-completeness or large dim** raw
wins (−0.25 … −0.68) because the frame physically cannot widen every token's margin — `code/V` (the
fraction of vocab γ-decodable) collapses 0.96→0.49, the **measured capacity price**. top1 stays ≈1
throughout, so the price is paid in certified margin, not accuracy.

This is the **same shape as the program's other scaling verdicts** — the spec §8 τ* result (refuted a
universal ≈exp(H) law; capacity-associated but **architecture-dependent**: Pythia grows ~0.5·nb, Qwen
flat). The certified-margin advantage is very likely **architecture-dependent too**: it is a property of
the residual-stream geometry (how diffuse `r` is, how crowded `{U_v}` is), which differs across families.
**Synthetic sweeps cannot vary architecture** — only the geometry knobs (dim, V/dim, coherence). So the
real model-size × architecture sweep (§6) is not optional polish; it is where the operating regime — win,
neutral, or lose — is actually decided for a given target model.

## 5. PoC results (full output: `experiments/margin_widening_loop.RESULTS.txt`)

`experiments/margin_widening_loop.py` — three regimes (`raw` / `widen` / `widen_t` / `narrow`) on a fixed
8× over-complete synonym-clustered planted frame, plus the λ×γ weight sweep. Headline verdicts:

```
at top1 ≥ 0.946 (95% of best raw) — best binding nm_p10 per regime:
  raw     : nm_p10=0.139  top1=0.996  t90=167  code=173
  widen   : nm_p10=0.053  top1=0.952  t90=175  code=153   bits vs raw = −1.38  (naive /‖r‖ is harmful)
  widen_t : nm_p10=0.135  top1=0.995  t90=175  code=173   bits vs raw = −0.04  (≈ raw on homogeneous ‖r‖)
  narrow  : (could not clear the top1 floor — fidelity collapses)
```

## 6. The closed loop with fieldrun — a size × architecture sweep (the decisive test)

The synthetic substrate has near-homogeneous `‖r‖` and shows the win is scale-contingent (§4a), so the
decisive test must run on **real models across sizes and architectures**, using the existing seam:

1. **Measure** — `fieldrun --source-dump` → `pil.fieldrun_io.load_source_dump` gives `D` (block vectors),
   `r = Σ_b D` (hence `‖r(x)‖`), `cands`, `margin`. Compute `γ̃(x)` and the `b*(x)` distribution; locate
   the forge-tax tail (the bits-binding positions).
2. **Train** — fit a PIL frame on these real sources with `widen_t` at the §3 knee (small γ, moderate λ).
3. **Re-certify** — feed the new frame's logits back through fieldrun's certified-quant allocator
   (`experiments/certified_quant/step1_5_embed.py`): does it certify *more* dropped bits at equal held-out
   decode-flip rate?

**The sweep (not a single point).** §4a proves the answer is regime-dependent, so run the loop across a
size × architecture grid and report `Δbits` (certified) and held-out flip rate per cell:

| axis | points (start) | available locally |
|---|---|---|
| **size** | 0.5B → 7B (→ larger) within a family | Qwen2.5-0.5B (have); 7B via HF `convert` |
| **architecture** | Qwen vs Pythia/GPT-NeoX vs Llama (rope vs neox, tied vs untied embed, MLP shape) | one bundle present; others via `convert` |

The prior from the spec's τ* result (Pythia grows vs Qwen flat — no universal law) is that the certified-
margin behaviour will **differ by family**, not scale uniformly. Output: a per-(size,arch) verdict —
*where* margin-widening cashes out in real certified bits, and where it does not.

## 7. Honesty / tags

- **`[proved]`** the certificate `2δ<margin ⇒ decode preserved` (`PIC_Quant`) and the capacity ceiling
  `N_γ̃ ≤ (1+2ρ/γ̃)^d` (`DecodeCapacity.thy`). These are kernel facts, not claims of this doc.
- **`[empirical]`** every PoC number here is synthetic (planted clustered frames), argmax-side, seeds≤3.
  The λ×γ knee is robust across both runs; the regime ranking is clear. **Not** validated on real models.
- **`[open]`** the central hypothesis — that `widen_t` raises *real* certified bits at equal fidelity —
  is **untested** (needs §6). The PoC deliberately shows it is *null on homogeneous data*, which is the
  honest reason to gate the build on the fieldrun measurement rather than ship a normalized term on faith.
- **Caveats:** capacity never binds in the PoC (so its "code" column tracks realized margin, not a
  capacity price); `c` is a calibration constant (only `Δbits`/Pareto are `c`-free); PIL accepts reduced
  host fidelity by design (`README`), so "fit cost" is measured in `nll`, not host-match.

## 8. v1.5 decisive sweep — real-model results (`experiments/margin_widening_realdata.py` + `…RESULTS.txt`)

The §6 size × architecture sweep is **done**, on real `fieldrun --source-dump` residuals (the decode logit
is `⟨r_x,U_v⟩`, so the per-block `d̃_b` collapse to `r_x` and this is a frame readout on the *real*
residuals). Two architecture families (rope: Qwen, Llama; neox: Pythia), the Pythia size ladder
(70m→1b), Qwen 0.5B/7B, plus domain (code) and language (Spanish) axes. Four verdicts:

**1. The gate is CLOSED everywhere → the normalized term is REFUTED.** `‖r‖` cv ∈ [0.003, 0.11] on all 9
cells (≪ the 0.35 needed to matter). The source-dump residuals are **final-norm-folded**, and every
architecture's final RMSNorm/LayerNorm pins `‖r‖ ≈ const` — so `γ̃ = margin/‖r‖ ∝ margin` and `widen_t`
collapses to `raw`. The matched-target control confirms it: Llama Δbits = **−0.08** (the unmatched "+1.84"
was purely a target-scale artifact). **The synthetic ‖r‖-heterogeneity the proposal banked on does not
exist on real models.**

**2. The raw knee SURVIVES — reproduces across architectures.** A small-target raw-margin auxiliary lifts
the binding certified margin and helps (never hurts) fit, λ-robust (0.25→2), on Qwen, Llama, and Pythia
residuals alike. (`N≈70` → train-side; the larger-corpus held-out test is the only loose end.)

**3. Headroom map (the deliverable) — no universal law, the τ* shape.** Certified-bit proxy `bstar_p90`
(c-absorbed; cross-cell deltas only):

| axis | cells | finding |
|---|---|---|
| size × arch | Pythia 70m→1b: **13.0→9.4**; Qwen 0.5B→7B: **10.2→10.5** | quantizability improves with scale for Pythia, **flat** for Qwen — exactly the spec §8 τ* Pythia-grows/Qwen-flat split |
| domain | code 7.7 vs prose 10.2 | structured/predictable text is **more** certifiable |
| language | ES 10.6 vs EN 10.2 | Spanish marginally **less** certifiable for Qwen |

**4. Interpretability — margin-widening is select-side only.** Widening sharpens **select** (decode margin,
e.g. Qwen 0.05→0.15) but leaves **retrieve/compose** (the target's block-PR) ≈ unchanged (Qwen 8.8→8.9,
Pythia 18.0→18.1): the frame can't redistribute a fixed `d̃_b`. Retrieve/compose is **architecture-dependent
and frame-invariant** (Pythia composes diffusely PR≈18; Qwen/Llama retrieve PR≈9). So frame-only widening
sharpens *decisions, not attributions*; shifting compose→retrieve needs **whole-model** training.

**Whole-model (principled, no run needed):** every arch has a final norm immediately before the unembed,
which pins the read-out `‖r‖` *regardless of which weights train* — so the normalized distinction stays
moot frame-only OR whole-model; only modifying/removing the final norm could open the gate. Whole-model's
extra value is reshaping retrieve/compose (the `d̃_b`), at host-fidelity cost.

## 9. Phasing (updated)

- **v1 (PR #1):** theory + synthetic PoC (regimes, λ×γ phase diagram, dynamics, size sweep). Shipped.
- **v1.5 (this PR):** the **decisive real-model sweep** (§8). Verdict: **normalized term refuted**, **raw
  knee survives**, **headroom map** built, **interpretability = select-side**. **No change to PIL's loss.**
- **v2:** land the **raw-margin knee** (small target, λ≈0.5–1.0) as an opt-in `PILConfig` auxiliary — the
  one surviving recommendation. Ablate vs `frame_reg`/support-size/host-fidelity and the prune-stats term.
  *(The `widen_t`/normalized objective is dropped — §8.)* Loose end: a longer-corpus held-out knee test.
- **v3:** the only routes left for a *bigger* win — **whole-model** training (move `d̃_b`: reshape
  retrieve/compose) and/or **relaxing the final norm** (the only way to make `‖r‖` heterogeneous and revive
  a normalized margin); the indefinite/Krein frame (`γ̃` signature-dependent); unify with prune+quant.
