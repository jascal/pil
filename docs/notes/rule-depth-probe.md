# Structure axis: composition depth buys nothing on pythia-70m's decode residual

**Status:** empirical, one small model + a thin data budget (caveats below carry weight). Reproduce:
`python experiments/rule_depth_probe.py` (needs tropic's `scratchpad/m2_train.pt` pythia-70m windows).

## Why

A sister track (tropic F11–F14) measured the *rate* axis of pythia's decode: how much of the next-token
decision is computed vs retrieved, and found it **data-limited, not a hard complexity floor** (the R(D)
knee moves with data). That answers *how much* is computed, not *what the computation is*. This probes
the **structure** axis with the tool the program already has: the PIC rule learner induces an explicit
weighted-Datalog program and can **deepen** — stack strata, i.e. bounded composition. If stacking depth
lifts coverage/accuracy, the computed residual is *symbolic-in-a-richer-class* (composition captures it);
if depth is flat, a flat lookup + a remainder that bounded symbolic composition can't compact.

## Result

pythia-70m, 8-token context → argmax, ~2.6k in-candidate windows (top-512 targets), max_strata capped 1..4:

| max_strata | strata used | rules | coverage@0.25 | hard-acc |
|-----------:|------------:|------:|--------------:|---------:|
| 1 | 1 | 993 | 0.871 | 0.178 |
| 2 | 2 | 1191 | 0.869 | 0.163 |
| 3 | 3 | 1369 | 0.853 | 0.184 |
| 4 | 4 | 1598 | 0.867 | 0.182 |

**Composition depth buys nothing.** The learner *did* deepen (1→4 strata) and grow (993→1598 rules), but
coverage and accuracy are flat. The depth-1 flat program captures everything the learner can; stacking
bounded Datalog composition recovers none of the residual it misses. (Note the gap between
coverage@0.25 = 0.87 and hard-acc = 0.18: `certified_fraction` measures margin/confidence, not
correctness — the program fires confidently but is mostly wrong; hard-acc is the meaningful readout, and
it too is flat and low.)

## Reading — and the caveats matter here

The flat depth curve is consistent with the residual being **not captured by bounded symbolic
composition** at this model/budget. But three caveats keep this from being a universal claim, and each
points at the fair follow-up:

1. **Data budget.** ~2.6k windows is ~8× less than the wheelhouse run; F14 showed *everything* about this
   decode is data-limited. Depth may only pay off with more data.
2. **Model scale.** pythia-70m is tiny. fieldrun's tree-vs-list diagnostic showed genuine recursion
   *emerges with scale* (absent at 0.5B, present at 1.5B) — a 70M model may have little compositional
   structure for depth to capture, flat here for that reason rather than "no structure exists."
3. **Bounded composition ≠ recursion.** Stratified Datalog is bounded depth, not a recursion scheme. The
   fieldrun tree-catamorphism synth is the recursion tool; a flat result here does not rule out recursion
   helping on a model where it emerges.

So: **on pythia-70m's decode, at this budget, bounded symbolic composition adds nothing over a flat
lookup** — the first structure-axis data point. The fair test of "is the computed fraction
symbolic-in-a-richer-class or genuinely distributed" is a *bigger model + more data + a recursion-scheme
learner* (where fieldrun already located the emergence). This probe is the cheap negative that motivates
paying for that.

## pythia-1b re-run (3 seeds) — a weak, seed-sensitive hint, not a robust effect

Ran the same probe on pythia-1b's decode (its own argmax on the same windows, ~2.9k in-candidate),
three seeds. hard-acc by depth:

| depth | seed 0 | seed 1 | seed 2 | 70m (seed 0) |
|------:|-------:|-------:|-------:|-------------:|
| 1 | 0.160 | 0.170 | 0.160 | 0.178 |
| 2 | 0.166 | 0.146 | 0.161 | 0.163 |
| 3 | 0.182 | 0.157 | 0.173 | 0.184 |
| 4 | **0.196** | 0.173 | 0.176 | 0.182 |
| Δ(1→4) | +0.036 | +0.003 | +0.016 | +0.004 |

The mean depth-1→4 gain at 1b is **+0.018** vs 70m's **+0.004** — so depth *does* seem to pay a bit more
at scale, in the direction fieldrun's "recursion emerges with scale" predicts. **But it is not robust:**
seed 1 is essentially flat and non-monotonic, and seed 0 (the clean monotone rise) is the optimistic
tail of the spread. Honest verdict: a **weak, seed-sensitive hint** of scale-dependent composition, not a
confirmed effect. This is consistent with the rate axis (tropic F14) showing pythia's decode is still
**data-limited** even at 1b — ~2.9k windows is too few to resolve a small structural signal. The clean
structure test still needs *more data + a model where the computed circuits are actually exercised* (the
regime where fieldrun's tree-vs-list gap was decisive), not natural-text decode at a CPU budget.
