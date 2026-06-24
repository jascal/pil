# Margin-widening loop: training PIL frames for certified quantizability

**Train the decode geometry so the downstream certified-quant bit-width drops — by targeting the
*scale-free normalized margin* (the actual certificate quantity), tuned by a margin-vs-fit weight, and
paying a *provable* capacity price.** Closes the measure→loss→learn loop with fieldrun: fieldrun measures
where the margin certificate binds, PIL learns frames that widen it, fieldrun re-certifies more dropped
bits.

*Status: proposal + a runnable, measured PoC (`experiments/margin_widening_loop.py`,
`…RESULTS.txt`). Certificate: i-orca `PIC_Quant` (`frame_quant_logit_bound`, `quant_decode_preserved`)
and the capacity ceiling `DecodeCapacity.thy` (`geometry.log10_packing_bound`). Downstream consumer:
fieldrun [`CERTIFIED_QUANT_PROPOSAL.md`](../fieldrun/CERTIFIED_QUANT_PROPOSAL.md) §10 / v1.5, which showed
the read-out (embed) is certifiable f16→int8 and the bits are margin-gated. Nothing in PIL's shipped
behaviour changes; this adds an optional objective + an experiment.*

---

## Current status & go/no-go

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

**Go/no-go.** The actionable finding is the **λ×γ knee** (§3) — a small-but-meaningful margin is a
free-lunch auxiliary regularizer usable today. The *normalized-vs-raw* win (§2) is **regime-dependent**:
§4a shows it is +1 bit at small dim, neutral mid, negative when the frame is crowded — and the prior (the
spec's τ* result) is that it is **architecture-dependent**, with no universal law. So the decisive test is
a **size × architecture sweep on real fieldrun `--source-dump` data** (§6), not a single run. Build order:
sweep real models → land `widen_t` only for the (size, arch) cells where it beats raw on certified bits at
equal held-out fidelity; everywhere else ship just the tuned λ×γ knob.

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

## 8. Phasing

- **v1 (this PR):** the theory (certificate→`γ̃` target, the gradient pitfall, the capacity price) + the
  measured PoC (regimes, the λ×γ phase diagram, dynamics, **the size sweep showing scale-dependence**).
  **No change to PIL's shipped loss.**
- **v1.5 (next):** run §6 as a **size × architecture sweep** on real `fieldrun --source-dump` — the
  decisive test, since §4a/τ* say the verdict is regime-specific. Per cell: `Δbits` certified at equal
  held-out flip rate.
- **v2:** land `widen_t` and/or the small-γ knee as an opt-in `PILConfig` objective, **gated on the
  (size, arch) cells where v1.5 showed a win**; joint with the capacity ceiling (allocate dimension where
  `γ̃` widening would otherwise overflow the packing bound — the `code/V`→1 regime of §4a).
- **v3:** the indefinite/Krein frame variant — `γ̃` becomes signature-dependent (`[r,U_t−U_w]/‖r‖_majorant`);
  unify with the prune+quant single certificate (i-orca `PIC_Prune`/`PIC_Quant`).
