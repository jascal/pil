# PIL theory: the two-algebra structure, the capacity limit, and questions for Grok

*Companion to `pil_learning_dynamics.md` (§5b/§5c empirics) and `fieldrun/PIC_PROPOSAL.md`
(the inference-time calculus). Status tags: **[PROVED]** = kernel-checked in i-orca;
**[MEASURED]** = empirical, this repo or fieldrun; **[OPEN]** = conjecture for this exchange.*

## 0. One-paragraph thesis

PIL is a **bilevel, two-algebra** learning problem. The transformer/PIC core factors as
**linear accumulation inside, tropical (max-plus) decision outside**. Optimizing the *frame*
`U` (the outer/decode geometry) is **decoupled** from the achievable margin once the
bottleneck is upstream **[MEASURED §5b]**; the lever is **generation** — adding rules/sources,
which lifts the effective dimension and so buys decision capacity **[MEASURED §5c]**. The
"structure is the hard limit" intuition is, we believe, a **tropical capacity statement**: the
number of tokens decodable with margin `≥ γ` is bounded by a packing number in `d` dimensions,
the decision-side sibling of the Welch bound and — under the data distribution — the same
object as the measured `τ⋆ = min(e^{H}, d)`. The open theory is to make that bound precise and
to characterize generation as dimension-lift / tropical-rank growth.

## 1. Objects and the two-algebra factorization

Hilbert space `H = ℝ^d`. Sources (circuits / partial evidence) `d_j ∈ H`; proposition frame
`U = {U_v}_{v∈V} ⊂ H` (the unembedding rows), `|V| = V`. Per context, a gate produces
`r = Σ_j a_j d_j` (the residual stream). Logits, decode, margin:

$$L_v = \langle r, U_v\rangle + b_v,\qquad \hat v = \arg\max_v L_v,\qquad \Delta(r) = L_{(1)} - L_{(2)}.$$

- **Inner algebra — linear.** `r = Σ_j a_j d_j` (additive) and `c_j^v = ⟨d_j,U_v⟩` (bilinear).
  Accumulation and attribution are entirely linear; this is the PIC "additive core" (T5 is an
  exact product-of-experts / log-linear model) **[PROVED, fieldrun T5 `RecoveredProbability`]**.
- **Outer algebra — tropical.** `decode(r) = ⊕_v L_v` in max-plus (`⊕ = max`, monomials `L_v`
  affine in `r`). The decision region of token `v` is the **Laguerre power cell**
  `C_v = {r : L_v ≥ L_w ∀w}` with weights `‖U_v‖²`, and the normalized margin is the **facet
  distance** to the nearest tropical hypersurface **[MEASURED, fieldrun FINDINGS §5b]**.

So the model is a tropical polynomial whose monomials are produced by a linear map. This is the
ReLU-network-as-tropical-rational picture **[PROVED, i-orca `examples/tropical`: `troprat_*`,
`OneHiddenLayerIsTropicalRational`]**.

## 2. What is already proved (i-orca kernel, zero `sorry`)

- **Head/tail decode certificate** (`HeadTail.thy`). Partition `V = H ⊔ T`. `head_certifies_decode`:
  if `decode(T) ≤ decode(H)` then `decode(H∪T) = decode(H)` exactly (and `head_argmax_in_head`);
  `tail_is_residue`: otherwise the decode lives in `T`. The compact head reproduces the decode
  **iff** it tropically dominates the tail; the tail is the explicit residue. This is the formal
  face of the measured "no compact-faithful unembed" boundary (~65% head / ~35% open-class tail).
- **Margin certificate** (`provable_opt/decode_margin_certified`, PO-T3): a `δ`-bounded logit
  perturbation cannot flip a decode with margin `> 2δ`; the `2δ` guard is proved tight.
- **Two-temperature soundness** (Maslov sandwich): the same program returns the exact softmax at
  `T=1` (log-semiring) and the exact greedy decode at `T=0` (tropical) — one program, two semirings.
- **Newton-polytope propagation**: tropical product = pointwise sum, monomial support =
  **Minkowski sum** (`NewtonSupportIsMinkowskiSum → tprod_slope_sumset`); submultiplicative
  monomial count. (Relevant to rule *composition*, §4.)
- **Welch bound** (`superposition` corpus): mean-squared coherence `≥ (V−d)/(d(V−1))` for `V>d`.

## 3. What we measured (PIL, held-out unless noted)

**(§5b) Frame optimization is decoupled from the decode when the bottleneck is upstream.**
Over-complete (`V=192`, `d=24`, planted synonym clusters, within-cluster `|cos|=0.82`). Sweeping
the frame-potential weight `frame_reg` moves the frame geometry (`fp/welch` 1.35→1.09) with **every
decode metric flat** (retr/top1/margin/nll), and `|Δretr| ≤ 0.01` across the whole synonymy×noise
plane while baseline retr swings 0.02→0.80 with the *real* levers. Reading: a readout-frame
objective cannot manufacture signal absent from the sources; the margin is **source-SNR-bound**.

**(§5c) Generation is the lever; min-margin targeting barely helps.** Synonyms collide on `r` but
hard clusters are **XOR-coded** (provably not linearly separable). A frame-only linear readout is
pinned at **chance** (hard-cluster acc 0.454, margin ≈ 0); generated ReLU rules (emitted as extra
sources) lift it to **0.93**. The `signal=False` control collapses all arms to chance → generation
recovers signal **only when it exists**. But **min-margin-targeted** rule placement barely beats
**random** rules (0.934 vs 0.911 at budget 32; behind at low budget), because **SGD already
allocates rules to the at-risk facets** — targeting buys initialization, not allocation.

**(prior, fieldrun) The recoverable-decode law `τ⋆ = min(e^{H(\text{output})}, d)`** — the effective
rank of the *output distribution*, capped by `d`; the forge tax = the open-class Zipf tail beyond it.

## 4. The central claims we want stress-tested

**C1 — The hard limit is a tropical capacity bound. [NOW PROVED — `i-orca/examples/tropical/DecodeCapacity.thy`]**
The right certified form is a *separation lemma*, sharper and cleaner than the cell-count conjecture.
Call token `v` **γ-decodable** if some residual `r` with `‖r‖ ≤ 1` decodes to `v` with margin `≥ γ`:

$$\textbf{(separation)}\quad v,w\ \text{each γ-decodable},\ v\neq w \ \Longrightarrow\ \|U_v - U_w\|\ \ge\ \gamma,$$

**independent of the biases `b`** — they cancel when the two witness inequalities are added
(`⟨r_v,U_v-U_w⟩ + ⟨r_w,U_w-U_v⟩ = ⟨r_v-r_w,\,U_v-U_w⟩ ≥ 2γ`, then Cauchy–Schwarz with `‖r_v-r_w‖≤2`).
So the γ-decodable set is a **γ-separated code** in `ℝ^d`, hence by sphere packing

$$N_\gamma \ \le\ \Big(1 + \tfrac{2\rho}{\gamma}\Big)^{d},\qquad \rho = \max_v\|U_v\|. \tag{packing corollary}$$

This is the **decision-side sibling of the Welch bound** (Welch bounds the *coherence* of `V` vectors;
this bounds the *count* of γ-separated ones — both are packing in `ℝ^d`), and it explains §5b directly:
frame tuning rotates the code but cannot pack more than `N_γ` cleanly-separated decodes into `d`
dimensions. `margin_pair_separation` / `decode_capacity_separated` / `head_capacity` are kernel-checked
(zero `sorry`); the explicit packing constant is the standard covering-number corollary. The `τ⋆` link
is now precise (see §4.1).

**C2 — Generation = effective-dimension lift = tropical-rank growth.** Adding `K` rules
`φ_k(x)` makes the readout linear in the augmented vector `[r, φ_1, …, φ_K] ∈ ℝ^{d+K}`, raising the
capacity bound. So **the minimal number of rules to reach margin `γ` on a target decode is a
"computational/tropical rank" of that decode**, and we conjecture it equals the *composed fraction
/ forge tax*. §5c is consistent (acc rises monotonically with budget `M`; the XOR needs `≈2` rules
per hard cluster). This unifies forge tax, `τ⋆`, and capacity under one number.

**C3 — Tropical algebra is the language of the limit, not the optimizer.** The decode is tropical
and the at-risk facets / dead rules / capacity ceiling are tropical objects — but **gradient
descent already performs the facet allocation** that explicit tropical targeting would encode
(§5c null). So tropical's contribution is *diagnosis, pruning, and the capacity ceiling*, while the
*generation move itself is linear* (find the discriminating direction). Tropical should only beat
SGD where the tropical-margin landscape has plateaus/saddles gradients can't cross.

### 4.1 The precise HeadTail ↔ capacity ↔ τ⋆ bridge (the connection Grok asked to pin down)

- **HeadTail → capacity (PROVED).** `HeadTail.head_certifies_decode`: the head `H` reproduces the decode
  when it tropically dominates the tail. Any token the head decodes with margin `≥ γ` is, by definition,
  γ-decodable, so the certifiable head `⊆ gdecodable U b γ`. By `head_capacity` its frames are a
  γ-separated code, hence `|H| ≤ (1+2ρ/γ)^d`. **The certifiable head is capacity-bounded** — there is only
  room for `N_γ` confident decodes, and HeadTail's "tail = residue" is exactly the overflow beyond capacity
  (the forge tax). This is a *proved* bridge between the two i-orca theorems.

- **capacity ↔ τ⋆ (the honest, careful statement; bridge still OPEN).** The packing bound is
  `(1+2ρ/γ)^d` — exponential in the *dimension*. The measured `τ⋆ = min(e^{H}, d)` is a *rank* (effective
  dimension), `≤ d`. They are **not the same quantity**: capacity is a cell *count*, τ⋆ is a *dimension*.
  The precise relation we claim is that **τ⋆ is the effective dimension that enters the capacity exponent** —
  the decode lives in a τ⋆-dimensional effective subspace, so the operative bound is `(1+2ρ/γ)^{τ⋆}`, and the
  forge tax appears when the distribution demands more confident-decode mass (`≈ e^{H}` distinct outputs) than
  `(1+2ρ/γ)^{τ⋆}` margin-γ cells provide. Equating capacity *with* τ⋆ would be an overclaim; "τ⋆ is the
  exponent's effective dimension" is the defensible form, and making it a theorem is **Q1**.

This also sharpens **C2**: generation raises the effective dimension (`[r,φ_1,…,φ_K] ∈ ℝ^{d+K}`), so it raises
the capacity *exponent* — the only lever that can, since §5b shows frame tuning cannot. The minimal rule count
to reach margin γ is then a lower bound tied to how much exponent (effective dimension) the target decode needs.

## 5. Questions for Grok

**Q1 (capacity theorem — the priority).** Make C1 precise. For `V` affine monomials
`L_v = ⟨·,U_v⟩+b_v` in `ℝ^d`, what is the tight bound on the number of power cells containing a
`γ`-ball (equivalently, tokens decodable with margin `≥ γ`)? Is it the sphere covering number
`(O(1)/γ)^{d-1}`, and does conditioning on a data distribution `D` with output entropy `H` give
the refinement `τ⋆ = min(e^{H}, d)`? Is there a clean derivation via tropical/Laguerre Voronoi
combinatorics or via the dual (the upper envelope / regular subdivision of the lifted points
`(U_v, b_v)`)?

**Q2 (generation as tropical rank).** Formalize C2. Is the minimal number of ReLU rules (hidden
units) needed to realize a target decode with margin `γ` a well-defined "tropical rank" of that
piecewise-linear map, and is there a *lower* bound (you provably need `≥ ρ` rules)? Does it
coincide with the number of vertices/cells the target's Newton polytope must add — i.e. growth by
Minkowski sum (`NewtonSupportIsMinkowskiSum`)? Does this rank equal the forge tax / composed
fraction, or merely bound it?

**Q3 (the two-algebra optimization).** Is there a principled formulation of the bilevel problem —
**outer** = tropical max-margin (place/keep monomials on the upper envelope), **inner** = linear
realizability (generate sources that route data into the cells)? Tropical convexity and tropical
SVM (Gärtner–Jaggi; Yoshida et al.) give max-margin in the tropical projective torus a convex
structure — does PIL's objective inherit it, and is there a minimax/dual we can exploit (or a
hardness obstruction)?

**Q4 (when does targeting beat SGD?).** Characterize the regime where explicit min-margin-facet
targeting provably beats gradient allocation of rules. Concretely: does the tropical-margin
objective develop plateaus/saddles (vanishing gradient on a not-yet-covered facet because no
existing rule fires there) that gradient descent cannot escape but a facet-seeded rule can? This is
the optimization-landscape question behind the §5c null.

**Q5 (temperature homotopy).** Is annealing `T:1→0` (Maslov dequantization, log-semiring →
tropical) a *convergent* and *principled* schedule? Do minimizers of the smooth `T=1` (softmax/NLL)
objective connect continuously to the `T=0` margin optima, or are there phase transitions /
symmetry-breaking events where cells appear or merge?

**Q6 (conservation / reallocation).** Prove or refute: *at capacity, generation is reallocation,
not addition* — is there a conservation law on the tropical upper envelope by which lifting one
monomial onto the envelope must demote another at fixed `d`? Is the right object a tropical optimal
transport / assignment problem over a bounded number of `γ`-cells?

**Q7 (sanity / which known results bind).** Two checks where you might catch an error: (a) is
"linear accumulation inside, tropical decision outside" the correct and *complete* factorization,
or is there latent tropical structure in the accumulation (e.g. via the LayerNorm/attention
softmax) we are wrongly treating as linear? (b) Which existing results — ReLU linear-region counts
(Montúfar; Serra), tropical rank/Barvinok, tropical PCA/SVM — *directly* bound `N_γ` or the rule
count, so we don't reinvent them?

## 6. What we'll do with the answers

A positive Q1/Q2 becomes a kernel-checked i-orca theorem (decision-side Welch sibling, tying
`HeadTail.thy` ↔ `τ⋆`), converting the measured hard limit to a certified one. Q3/Q4 redesign the
PIL proposer (and tell us whether tropical targeting is ever worth more than SGD). Q5/Q6 are the
training-schedule and capacity-allocation principles for the real (fieldrun-seeded) runs.
