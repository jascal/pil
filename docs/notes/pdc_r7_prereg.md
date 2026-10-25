# PDC R7 preregistration: learn native-source admission, 2026-10-25

Frozen before the first learner step. R5's epistemic stop remains active;
this arm cannot advertise theory soundness. R6's refused-slice improvement
and empty admitted stratum, and R8-H's zero answer coverage, are retained.
No reserved R5 template-3 context may enter the data, loss, guard or tuning.

## Fixed computation and decoder scope

Use the pinned native GPT-2 model and the exact R5/R6 artifacts. The
training/admission domain is all 75 R5 contexts. The protected bank is those
75 plus all 108 R6 hop-2 forwards, including their recorded duplicates.
No context is dropped for a small margin or a wrong theory answer.

All backbone parameters, token embeddings, measured residual writes and
input-dependent gates stay fixed. Learn an **untied output frame and bias**;
do not overwrite GPT-2's tied input embedding. The learnable vocabulary is
the fixed R8-H lexical probe union all protected native global winner IDs,
sorted/deduplicated before learning. All other output rows and their biases
remain fixed at the original frame. This keeps the expanded lexical scope;
do not replace it by an easier nine-answer-row problem.

Export a self-contained NPZ from fieldrun, consumed by pil without code
imports across repos. It contains the 75x25 measured, final-LN-folded native
source vectors, protected residuals r=actual native final hidden minus the
fixed LN beta vector, initial frame rows U and effective biases b=U beta,
full native winner IDs, bounded-addition labels, and the fixed outside terms.
Keep the numerical closure r-sum(native sources) explicit and frozen. It
belongs to the full residual/tail but is not an eligible native singleton.
No write, closure, gate, candidate set or teacher label is trained.

For each protected context, compute the maximum logit among all fixed
outside rows once. This is the exact argmax representative while those rows
and residuals stay fixed. Compute outside logsumexp separately for decode
cross-entropy. Never use the logsumexp as an argmax atom. Verify the complete
initial frame's strict global decisions match every recorded native winner.
If the gate fails, record an inapplicable run and its mismatches; do not
silently exclude contexts or pretend a candidate-set match is full-host agreement.

## Oracle and loss

On each of the 75 contexts, the target for admission is its recorded native
global winner, not the theory's possibly different number. Try the 25 native
singletons in capture order. Covered rivals are all learned-pool tokens.
Require m>max(0.5,2delta), with tail midrange kappa and delta half the tail
score range including target and all covered rivals. Store gamma as
(m+max(0.5,2delta))/2. This is the same strict centered-tail admission guard
as R4/R6. Refusal is frame/procedure relative, not universal impossibility.

Two deterministic full-batch arms, same initialization and safeguards:

1. **decode-only control:** ordinary cross-entropy to the true addition token.
2. **admission objective:** that same cross-entropy plus mean
   softplus(min_j(max(0.5,2delta_j)-m_j)), weight 1.0. The min is over the
   25 fixed native singleton choices. No FP regularizer is optimized.

Cross-entropy has all pool rows plus one *loss-only* outside-logsumexp term,
so it equals the full-vocabulary denominator with fixed outside rows.
R5's wrong native winners remain protected; decreasing loss must not be
misreported as correcting those decisions or restoring theory soundness.

## Fixed optimizer, enforcement and invariants

CPU float64, torch seed 20261025, four torch threads. AdamW, learning rate
0.001, betas (0.9,0.999), weight decay 0, exactly 200 proposals per arm.
No stochastic batches, learning-rate sweep or best-checkpoint selection.
The final (200th) accepted/restored state is the endpoint. Optimizer moments
retain full proposals when applied steps are clipped, as in existing PIL P3.

Reuse `CertifiedP3`, renorm=false, eta=0.9, m_floor=0, no exact-floor
exclusions, alpha_min=2^-20. Teacher row norms encode real margins and are
not projected to unit length. Enforce zero full-host decision flips on the
entire protected bank. Log actual max row drift epsilon, bias drift beta,
rho=max norm(r), current minimum margin and every accepted-step premise
2(rho*epsilon+beta)<=0.9*minimum_margin. This is PIC_Learn's proved
step_decode_preserved condition, with fixed outside rows having zero drift.
Check its consequence, not merely the bound's arithmetic.

Add a candidate guard preserving every previously admitted training
context's existence of a valid singleton certificate. A proposed loss of
that certified mass triggers further halving; an unresolved step restores
the exact previous frame. No accepted frame may report admission with a
failed m>gamma>2delta inequality. This is an explicit runtime invariant,
not a claim that unconstrained learning increases margins.

Measure FP with PIL's mean squared off-diagonal *cosine* convention, and
W=max(0,(M-d)/(d*(M-1))). Normalize rows for this geometric diagnostic only,
not for decoding. Every accepted endpoint must have nonzero finite rows
and FP>=W up to 1e-10 numerical tolerance; otherwise abort as a failed
invariant. Report the per-step checks and use an independent endpoint audit.

## Advance gates and negative outcome

The admission arm passes only if all invariants hold, its endpoint admission
rate rises by at least **five percentage points** over its own initialization,
and it exceeds the decode-only endpoint by at least **two percentage points**.
Full-host agreement must stay 100% on all protected contexts and R6 hop-2
theory accuracy must not fall in any arm. The guard constraints are part of
the measured protocol, so preserved bank accuracy is not a generalization claim.

A smaller gain, a tie with plain decoding, or an improvement in FP/W without
the required admission movement is a negative learning result at this fixed
budget. Do not extend the run or select another objective after seeing it.
Zero accepted motion is a measured plateau, not an impossibility proof.
Store both initial/final frames, full proposal/clip logs, frozen data hashes,
all oracle verdicts, outside-reduction checks and explicit invariant failures.
The 25 untouched contexts are reserved for a later frozen-policy audit.
