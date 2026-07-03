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
- **Keep as the deliverable:** the **certified-bit headroom map** (§8) — domain/language-dependent and
  **size-saturating** (the same-data Pythia ladder §8a shows headroom plateaus past ~400M, R²=0.27 — no
  scaling law; Qwen's apparent flatness is just that plateau).
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
residuals alike. *Effect size (binding `nm_p10`, λ=0→knee):* Llama 0.121→0.148 (**+0.29 bits**), Pythia-410m
0.113→0.145 (**+0.37 bits**), Qwen −0.265→0.12 (the margin term also *rescues* fit, top1 0.57→1.0). So
≈0.3 certified bits where fit is already clean, more where it rescues it. (`N≈70` → train-side; the
larger-corpus held-out test is the only loose end.)

**3. Headroom map (the deliverable) — no universal law; it SATURATES.** Certified-bit proxy `bstar_p90`
(c-absorbed; cross-cell deltas only):

| axis | cells | finding |
|---|---|---|
| size × arch | full same-data Pythia ladder 14m→2.8b: **9.9, 12.5, 11.7, 9.0, 9.4, 9.8, 9.2**; Qwen 0.5B→7B: 10.2→10.5 | **NOT a scaling law** (fit R²=0.27): high/noisy <400M, **plateaus** past ~400M (≥400M mean 9.3, spread 0.76). Qwen's "flatness" is consistent — both points sit in the ≥400M plateau. See §8a. |
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

### Synthetic → real: what changed

| claim | synthetic PoC (§3–4a) | real-model sweep (§8) |
|---|---|---|
| `widen_t` vs `raw` | ≈equal on homogeneous ‖r‖; ±1 bit only under engineered heterogeneity | **≡ raw** (‖r‖ norm-pinned, cv≤0.11; matched Δbits −0.08) → **refuted** |
| raw margin knee | free lunch (↑cert margin, ↑top1, faster, ~0 fit cost) | **reproduces** across arch (~+0.3 bits) → **ship** |
| scaling | widen advantage flips sign with dim/over-completeness | **headroom saturates** past ~400M (same-data Pythia ladder, R²=0.27) — no scaling law; Qwen-flat is just the plateau (§8a) |
| capacity price | real but dormant (ceiling ≫ V) | not exercised (real `code/V` not near 1) |

### 8a. Same-data Pythia ladder — the arch-vs-data confound, RESOLVED (`experiments/pythia_ladder_scaling.py`)

*What changed: §8's headroom claim is corrected here — the size trend is **saturating, not scaling**, and
the earlier Pythia-vs-Qwen split was a size-range artifact.*

The PR-#2 open question — *is Pythia's headroom trend a size effect or a data-ladder artifact?* — is now
settled. The Pythia suite trains every size on the **same Pile data in the same order**, so the full ladder
(14m→2.8b) on **one fixed eval text** (N=131) holds training data constant by construction. Result
(`…RESULTS.txt` has the ASCII shape):

```
size    14m   70m   160m  410m   1b   1.4b  2.8b
bstar_p90 9.9  12.5  11.7   9.0   9.4   9.8   9.2     fit −0.86 bits/decade, R²=0.27 (WEAK)
                          └────── plateau: ≥400M mean 9.34, spread 0.76 ──────┘
```

**Verdict: NEITHER arch nor data — it's a *saturating capacity* effect.** With data fixed there is **no
clean scaling law** (R²=0.27); headroom is high/noisy below ~400M and **plateaus past ~400M**. A flat
"constant past 400M" model (mean 9.34, residual spread 0.76) describes the tail far better than the
log-linear fit — i.e. saturation, not a power law. So:
- The PR-#2 reading "Pythia improves with scale 13→9.4" was an **over-read of a narrow size range** (70m→1b
  = the rising+knee region) on short text. *(Corrected here.)*
- **Qwen's "flatness" is just the plateau** — both Qwen points (0.5B, 7B) sit in the ≥400M saturated
  regime, exactly like Pythia ≥410m. No architecture difference is needed to explain it.
- Data is **not** the driver (same Pile across the ladder → same saturation). Consistent with τ*: no
  universal law. The gate stays CLOSED at every size (cv ≤ 0.035) — the §8 refutation holds across the ladder.

### Open questions (honest — hypotheses, not claims)

- **Why does Pythia compose (block-PR≈18) while Qwen/Llama retrieve (≈9)?** Likely the parallel
  attn+MLP residual structure spreads contribution across more blocks; untestable from logits alone —
  would need per-block ablations (fieldrun `--block-ablate`) to confirm causally.
- **What sets the ~400M saturation knee, and does it move with eval distribution?** Measured on one fixed
  text; a second eval distribution would test whether the plateau location is text-invariant. *Speculation
  (not a claim):* `bstar = −log₂(margin/‖r‖)`; with `‖r‖` norm-pinned, the plateau is really a *margin*
  plateau — past ~400M the model's next-token confidence on ordinary text stops sharpening (it already
  "knows" the easy continuations), so the certified read-out margin saturates. The high/noisy <400M end is
  plausibly genuine instability of small-model frame geometry (few-shot-confident on a handful of tokens,
  diffuse elsewhere), not just statistical — but N=131 can't separate the two; needs more positions / seeds.

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

## 10. T-traj discharge experiment 1: do real P3 steps satisfy the theorem's premises? (2026-07-03)

T-traj (kernel-proved, i-orca `examples/pic_learn/PIC_Learn.thy`, merged main `735973f`) is
unconditional; this section reports whether **real optimizer steps** — AdamW(wd=1e-4) + row
renormalization, the actual P3 — live inside its premises, and with how much slack.
Instrumentation: `pil/certify.py` (`TrajectoryCertificate`, off by default; effective updates
measured post-normalization→post-normalization; decisions frozen at entry; ρ-constancy asserted;
silent contexts counted, not dropped). Sweep: `experiments/t_traj_discharge.py` — {easy, hard,
mwl_raw, real(pythia-70m `--source-dump`, 70 positions, recon 1.00)} × lr {5e-3, 3e-4} × 2 seeds
× 2 entry points (step 0 / 25%); tracked banks half train, half held-out. Raw numbers:
`results/t_traj_discharge.{txt,json}`. All rows **empirical**.

**Harness soundness (the theorem's self-checks).** Across all 20 cells × 600 steps × 2 entries:
**zero** flips-while-premise-held and **zero** transfer violations (`m' < m − 2δ`). The forbidden
events never occur on real AdamW+renorm steps — the measurement matches the mathematics.

**Findings (descriptive):**

1. **Entry point dominates.** At entry@0 the premise almost never holds (init margins 0.13–0.26
   vs per-step `2δ` up to ~0.6 at lr 5e-3) and essentially every tracked decision flips. At
   entry@25% — the picard-relevant "cell visited mid-loop" case — hold-rates are 0.99 (easy),
   0.72–0.83 (hard/mwl, lr 5e-3), but only **0.24–0.47 on real residuals**.
2. **The premise is margin-limited, not step-limited.** The hot lr (5e-3) has *higher*
   post-warmup hold-rates than 3e-4 (hard: 0.72–0.83 vs 0.37–0.60, and at 3e-4 the rate
   *declines* over time): margins grow faster than δ under the hot lr, while the slow lr
   lingers in the small-margin churn zone. Naively cooling the optimizer does not rescue the
   certificate.
3. **The a-priori trajectory budget is vacuous everywhere.** `2Σδ` crosses `m₀` within 1–22
   steps in every cell (vacuous fraction 1.00); telescoped slack ≈ the whole budget. The usable
   form on real runs is the **per-step check** (`step_decode_preserved`), re-armed each step —
   exactly the plan's runtime-corollary framing; the trajectory form is for *clipped* updates.
4. **Bias drift is real but small.** β contributes 2–4.6% of δ on synthetics and ≈0 on real —
   Correction 1's term belongs in the theorem, and ρε dominates it 20–50× in practice.
5. **Per-context ρᵢ buys nothing.** Tracking the tighter per-context premise
   (`m_i > 2(‖r_i‖ε + β)`) moves hold-rates by ≤0.01 everywhere including real. The
   conservatism lives in `ε = max_v ‖ΔU_v‖` and in the margins themselves, not in ρ spread.
6. **Held-out contexts flip more than train contexts** (never-flipped 0.54–0.64 vs 0.69–0.82 on
   hard at 5e-3) — the corpus-mirage direction, visible even in the certificate.
7. **hard ≡ mwl_raw at shipped defaults**: the learner's `total_loss` (margin_weight 0.5,
   target 2.0, frame_reg 0.05) *is* the mwl raw objective; the two regimes produce identical
   trajectories. One fewer distinct regime than intended; kept for the record.

**Reading (per the decision rule; no headline).** At these lr/dim settings the per-step premise
holds at high rates only post-warmup on synthetics; on real residuals it fails at rates
0.53–0.76, dominated by **small margins relative to ε** (not β, not ρ). A certified P3 therefore
cannot rely on the a-priori budget and cannot be rescued by lr cooling alone; the candidate fix
is a **trust-region P3** — clip per-row `‖ΔU_v‖` (and `|Δb_v|`) so `2(ρε_t + β_t)` stays under
the current tracked minimum margin, re-arming the per-step certificate each iteration —
**untested; achievability open**.
