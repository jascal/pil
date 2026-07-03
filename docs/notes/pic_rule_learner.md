# The PIC rule learner: learning weighted-Datalog programs by semiring backprop

**Status:** design note (v0.1), companion to `pic/spec/PIC_LP.md` and `pic/spec/PIC_SPEC.md`
§4.3/§6.5. This is the *learning* arrow that the analysis stack (`fieldrun` → `rosetta`) runs in
reverse: instead of decompiling a trained transformer into PIC-LP rules, **train the PIC-LP rules
directly from data**, starting from random rules, with gradient credit assignment on the incidences
and discrete structural edits on the rule set.

The one-line claim: **a PIC program is a differentiable object at `T > 0` and a Datalog program at
`T = 0`, and Maslov dequantization connects them — so we can train in the log-semiring and ship in
the Boolean/tropical semiring, with the decode unchanged by §3.1 (decode temperature-invariance).**

---

## 1. What is being learned (the hypothesis class)

A **layered PIC rule program** over a token vocabulary `V` (the *same* vocabulary as the LLMs
modeled in `rosetta` — pythia/GPT-NeoX, Qwen2, llama; loaded from the package's
`bundle.tokenizer.json` or the HF snapshot's `tokenizer.json`):

- **EDB (the facts).** A window of `W` context tokens at relative offsets `o ∈ {0, …, W−1}` left of
  the decode position: atoms `tok(p − o, c)`. This is exactly rosetta's `tok(inst, pos, id)`
  grounding.
- **Rules (stratum ℓ ≥ 1).** Rule `k` has
  - a **body**: a conjunction of literals. A stratum-1 literal is a *token-class predicate* at an
    offset — `φ_{k,o}(c) = σ((⟨E_c, w_{k,o}⟩ + β_{k,o}) / τ)` with `E` a token-embedding table
    (fixed-random, learned, or loaded from the host LLM). A stratum-`ℓ>1` literal is the (possibly
    negated) firing of a rule from an earlier stratum. Every literal carries a **slot relevance**
    `ρ ∈ [0,1]` mixing it with `true` (`ρ = 0` ⇒ the literal is absent — the wildcard).
  - a **gate**: `g_k(x) = ⊗_lit  (1 − ρ + ρ · m_lit(x))` — the `⊗`-monomial of its body, evaluated
    in the training semiring (product = probabilistic AND; hard AND at export).
  - a **head**: a write direction `a_k ∈ H = ℝ^d`. The rule's incidence on proposition `v` is PIC's
    native `⟨a_k, U_v⟩` with `U` a shared learned frame — the `pic_encoder` of `PIC_SPEC.md` §4.3,
    now trained end-to-end. Intermediate-stratum rules have no head; their *firing* is the derived
    atom later strata consume.
- **Decode.** `L(v) = Σ_k g_k(x) · ⟨a_k, U_v⟩ + b_v`, read out by `⊕_T`: softmax at `T = 1`
  (training), argmax at `T = 0` (deployment). By decode temperature-invariance the argmax is the
  same at every `T` — the trained program and the exported Datalog program decode identically
  wherever the gates are crisp.

**Datalog reading.** A hardened stratum-1 rule with exact-token slots is literally rosetta's ngram
clause `ctxlogit(I,Tk,S) :- tok(I,P−o₁,c₁), …, gram(…)`; a hardened class predicate becomes a fact
set `cls_k(c)` plus the literal `tok(I,P−o,C), cls_k(C)`; a stratum-2 rule is an IDB clause over
derived predicates `fired_j(I)` with stratified negation for negative literals. Export is §6.

## 2. Why this can work (and what "sparser than backprop" means)

Deep-NN backprop updates **every** weight on **every** example. Here:

- the gate is near-binary after annealing, so the gradient of a non-fired rule's head is ≈ 0 and
  the gradient of its body is ≈ 0 except within the annealing band — credit assignment touches the
  **fired coalition**, not the network;
- the head is **additive** (a semiring `⊗`-monomial per rule, `⊕` across rules only at the decode),
  so *exact* per-rule ablations are O(1): `Δ L = −g_k(x) ⟨a_k, U_·⟩`. Utility, pruning, and the
  margin certificate (§5.5: a rule may drift by `δ` without flipping any decision whose margin
  `> 2δ`) are all computable in closed form, which no dense NN offers;
- structure is grown where the loss is, not pre-allocated: rule **birth is functional-gradient
  boosting** (§4.3) — a new rule's zero-initialized head receives, on its first gradient step,
  exactly the residual error restricted to the examples its body fires on.

## 3. Determining initial random states (design decision i)

Pure-uniform random bodies fail structurally: under a Zipfian vocabulary of 50k types, a uniformly
random exact-token conjunction of arity `A` fires with probability ≈ `Π_o p(c_o)` ≈ never, so the
program is born dead (zero gradient everywhere — the discrete analogue of dead ReLUs). The fix is
the same move NN init theory makes (calibrate variance so signals neither die nor explode), applied
to *firing measure*:

**Random-in-data-measure init.** To create a random rule: draw a training position `x` at random,
draw an arity `A ~ Geometric` clipped to `[1, A_max]`, draw `A` distinct offsets (biased toward
recency, matching the measured n-gram recency structure), and set each slot's predicate to the
*observed* token at that offset — sharp in the slot (`ρ = 1`), broad in the class (predicate
initialized so `φ(c_obs) ≈ 1` with soft mass on the embedding neighborhood, so gradients can widen
it into a class). The **head starts at zero**: a newborn rule provably does not perturb the decode
(additivity), and its first update is the boosting residual. Firing calibration is then automatic:
every rule fires on at least its seed example, and expected firing rate equals the empirical
measure of its body pattern.

This *is* still "starting from random rules" — random in the only measure under which random rules
are alive. The uniform-measure baseline is kept as an ablation arm (expected: dead program).

## 4. The learning loop (design decisions ii and iii)

The loop is pil's Generate → Gate → Refine, now over program structure:

### 4.1 Refine — semiring backprop on incidences (ii)

Loss (labels as in `learner.py`): `NLL(L/τ_dec, t)` (decode-side) + `λ_m · ReLU(γ★ − m_worst)`
(decode-side margin hinge) + structural regularizers (frame-side / program-side):
`λ_ρ · Σ ρ` (body sparsity — atom removal pressure), `λ_a · Σ‖a_k‖₁`-style head sparsity, and an
optional firing-rate target. All parameters — slot predicates `w, β`, relevances `ρ`, heads `a_k`,
frame `U`, biases — update by autograd **through the semiring ops**; `⊗` is the product t-norm
(smooth), `⊕_T` is logsumexp. **Annealing:** predicate temperature `τ` and a gate hardness
schedule move the program from smooth (early, gradients flow broadly) to crisp (late, semantics
approach Boolean); a final **hardening** step rounds `ρ` to {0,1} and classes to their level sets
and re-fits only heads/biases (a convex problem — the gates are fixed Booleans) so the exported
program's incidences are exact for the hard semantics, not an approximation of the soft ones.

### 4.2 Gate — feedback signals (the sufficient statistics for structure)

Collected per rule per epoch, each cheap because of additivity:

| signal | definition | drives |
|---|---|---|
| firing rate `f_k` | `E[g_k]` | death (dead rule), body edits |
| utility `u_k` | exact ablation: mean margin/NLL delta from zeroing rule `k` | death (useless/harmful rule) |
| ambiguity `η²_k` | between-target / total variance of the targets on `{x : g_k(x) ≈ 1}` (pil `scoring.py`) | split / specialization |
| slot marginal `∂Loss/∂ρ` sign & magnitude | is each literal helping where the rule fires | atom add/remove |
| residual mass | per-example loss on the worst quantile | birth targeting |

### 4.3 Structural edits (iii)

Discrete moves between SGD phases (the outer loop):

- **Birth (boosting).** Take the current worst-loss examples; seed new rules from their contexts
  (§3 init, head = 0). Because heads are zero, births are decode-neutral at creation and get
  trained into the residual. This is functional gradient boosting over rule space; the *targeting*
  is the lever pil already validated (generation recovers what no frame objective can).
- **Death (pruning).** Remove rules with `u_k ≈ 0` (harmless) or `u_k < 0` (harmful) after a
  protection age; exactness of the ablation makes this safe (no retrain-to-measure loop).
- **Specialize (split).** A rule with high `η²_k` (fires on conflicting targets) is cloned; each
  clone gets one extra discriminating literal chosen by information gain over the fired set (the
  ID3 move, but the clone's heads then retrain by SGD). This is the discrete counterpart of what
  the margin hinge cannot fix by weights alone.
- **Generalize (widen/drop).** `ρ`-decay already deletes unhelpful literals continuously; a merge
  pass unions the class predicates of rule pairs with near-identical bodies (Hamming 1) and
  near-parallel heads (`cos(a_i, a_j) ≈ 1`).
- **Deepen.** When the stratum-1 residual plateaus but ambiguity stays high (the XOR signature:
  no conjunction of input literals separates, but a function of *rule firings* does), allocate a
  new stratum whose bodies range over existing rule firings. Parity/composition live here.

## 5. Efficient execution against data

Training eval is one batched einsum chain per stratum:
`match[B,K,W] = σ((E[x][B,W,e] · w[K,W,e] + β)/τ)` → gate `g[B,K] = Π_W (1−ρ+ρ·match)` →
`L[B,V] = (g · G) A Uᵀ + b`, all O(B·K·W·e + B·K·d + B·V·d). Exact-token slots use a gather fast
path (no embedding matmul). Hard (deployment) eval is integer hashing of ngram keys — rosetta's
own execution model — plus Boolean stratum propagation; implemented in pure Python/numpy with
semantics *identical to Soufflé* and cross-checked against `souffle` on the exported program.

## 6. Export and verification

`pil.datalog_export` renders a hardened program in rosetta's house style: `tok(inst,pos,id)` EDB;
`cls_<k>(c)` fact sets for widened classes; one clause per rule
(`fired_k(I) :- tok(I,P−o,c), …` / `ctxlogit(I,Tk,S) :- fired_k(I)` with the top-K incidences
`S = ⟨a_k, U_v⟩` materialized per candidate); `logit = sum`, `decide = max` — the two aggregates
that are PIC's `⊗` and `⊕`. Verification gate: `soft-argmax@T→0 == hard-eval == souffle` on a
held-out slice, reported as an exact-match fraction (target 1.00 after hardening; any mismatch is
a bug, not noise, by §3.1).

## 7. Relation to prior art (honesty section)

Differentiable ILP (∂ILP), Neural Logic Machines, and neuro-symbolic rule learners share the
"soft logic + gradient" move. What is PIC-native here: (a) the head is not a scalar rule weight
but an **incidence in a learnable frame** `⟨a_k, U_v⟩` — the geometry (Gram, Welch, margins,
capacity theorems §5.1–5.3) applies verbatim, and vocabulary-scale heads are O(d) per rule, not
O(V); (b) the semiring family with **decode temperature-invariance** gives an exact, certified
train-soft/ship-hard bridge (train at `T=1`, export at `T=0`, same argmax) instead of a lossy
distillation; (c) the margin certificate (§5.5) bounds clause-weight drift, so pruning/quantizing
exported incidences is *certifiable*; (d) structure search is boosting-with-exact-ablations, not
template enumeration (∂ILP's blowup). Claims tags: the semiring/margin/invariance statements are
**proved** (i-orca, cited in PIC_SPEC); everything about learnability/benchmarks below is
**empirical**; "sparser credit assignment ⇒ better sample efficiency" is **open** until measured.

## 7.5 What building it taught (findings, tagged)

- **AND-rules alone fail parity** *(empirical, decisive)*: parity-8 plateaus ≈ 0.57 under
  conjunctive bodies — no proper-subset literal has correlational signal, so purity/boosting
  are blind. The fix is not deeper AND-strata but the **PIC-T3 threshold gate**
  `Σ_o ρ_o·lit_o ≥ θ` (softened `σ(β(z−θ))`, annealed β): parity-8 exact in one stratum,
  parity-16 ≈ 0.99 on strictly-unseen patterns, beating a matched MLP with ~21× fewer
  parameters. Inspection shows the learner *rediscovers the textbook depth-2 threshold
  circuit*: full-window count rules whose head signs alternate with the ones-count parity.
  The T3 gate is PIC-native (the ⊗ = + coalition bracket behind a margin turnstile ⊢_γ) and
  stays pure Datalog (a Soufflé `count` aggregate).
- **Random init must be random-in-data-measure** *(empirical)*: uniform-random bodies are
  born dead under a Zipfian vocabulary (§3's prediction, confirmed); data-seeded rules with
  zero heads train immediately (boosting residual is their first gradient).
- **Fixed-offset ground rules are the wrong basis for relational structure** *(empirical)*:
  the induction task (A B … A → B, position-varying) stalls near the MLP's mediocre score
  when bodies are (offset, token) literals only. The remedy is the **equality literal**
  `x[o₁] == x[o₂]` — the smallest instance of PIC-LP's position-indexed join (`PIC_LP.md`
  §7.5) — exported as the pure-Datalog join `tok(I,o₁,C), tok(I,o₂,C)`.
- **Exactness of the bridge is real, not aspirational** *(verified)*: after `round_structure`
  + head refit, soft argmax == Boolean-gate argmax == Soufflé `decide` at 1.00 agreement on
  every program tested (the §3.1 temperature-invariance bridge, executed).
- **Rules need a retrieval backbone for open-vocabulary LM** *(empirical)*: on wikitext-2 a
  ~1000-rule program lands *under* the bigram floor — anchored clauses cannot match a lookup
  table's capacity. The fix is rosetta's own architecture (ngram tables + idiom rules): a
  **lookup source family** per offset (`d = A_o[x[o]]`, identity gates — the *retrieved*
  fraction), with AND/T3/eq rules learning the *composed* residual on top; exported as
  `lkp{o}(C,V,W)` fact tables + one clause, rosetta-style.
- **Algebraic generalization needs background knowledge** *(empirical)*: modadd (p=97, 50%
  of pairs) transfers 0% held-out for both ground-rule programs and the matched MLP (the
  grokking regime). The ILP move — a **schema library** (modular arithmetic, copy/pointer)
  whose *selection* is learned from data hit-rates and whose weight is SGD-trained — closes
  it, exporting as an arithmetic Datalog clause over `num(token, value)` facts. The copy
  schema is exactly rosetta's `ind_ctxlogit`; the channel is the "richer proposers" roadmap
  item made concrete.

- **Tabular needs ordinal literals** *(empirical)*: on classic tabular datasets, equality
  bins fragment monotone structure (wine stuck at 0.833); the per-literal `x[o] ≤ anchor`
  mode (valid because bin tokens are ordered within a feature; exported as the Datalog
  constraint `tok(I,o,C), C <= c`) recovers the tree-style split (wine 0.944). Verdict vs
  the incumbents: GBT still wins on raw floats (2.4–7.5 pts across breast-cancer / wine /
  digits / adult); PIL beats LogReg + MLP on adult and is alone in emitting a certified
  Datalog program with §5.5 per-decision robustness and abstention.

## 8. Benchmarks (the "standard problems" bar)

1. **Parity** (len 8–12, tokens "0"/"1" in the host tokenizer): non-linearly-separable; requires
   the deepening move (stratum ≥ 2). The classic NN-hardness probe.
2. **Modular addition** `a + b (mod p)` as token sequences — the grokking benchmark.
3. **Sequence copy / induction**: `A B … A → B` — the recursive PIC-LP clause, learnable via
   equality atoms.
4. **Real-text LM**: next-token on the rosetta pythia-160m corpus slice (same tokenizer, same
   `tok` grounding), scored vs (a) an MLP of matched parameter count and (b) the frequency/ngram
   floor; comparability with the host LLM via the package's `logit_cache.json`.

Metrics per task: top-1 accuracy (hard program), rule count / literal count (program size),
firing sparsity, avg margin, and the soft↔hard agreement fraction.
