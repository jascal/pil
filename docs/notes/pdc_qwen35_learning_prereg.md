# Qwen3.5-4B guarded admission learning protocol, 2026-10-25

This is a new-model R7 arm, not an extension or retuning of the negative
GPT-2 R7 run or Qwen singleton diagnostic. No optimizer proposal or learned
Qwen frame has been evaluated. Fix this protocol before reading training
outcomes; the two-hop baseline is being captured under its own frozen R6
protocol, not selected for learning performance.

## Data and computation kept fixed

Consume a self-contained NPZ interface exported by fieldrun. Train on all
100 previously observed Qwen single-position cases and all65 native source
writes per case; they are development data now, including the formerly
heldout wordings. No fresh-confirmation claim is made. Preserve the actual
native final RMS scale and gates, source vectors and numerical closure.
The closure is part of full residual/tail scoring, not another singleton.

The protected bank is those100 plus ALL108 named paths from the separately
frozen Qwen two-hop comparison, with duplicate contexts retained as such.
Its physical native budget is54 unique second-hop contexts, not108. Native
full-host labels, whether correct or wrong, are frozen. No hard context
may be removed for a small margin, wrong theory answer or refusal. Require
initial float64 full-vocabulary decisions to match all208 native labels;
otherwise retain an inapplicable-data report and do not train.

Learn an UNTIED output frame and affine bias initialized to native U and
zero. Input embeddings and the full backbone do not move. Candidate pool
is the original2261 lexical probes UNION every actual native protected
winner ID, frozen before learning, sorted and deduplicated. All outside
unembedding rows/biases stay fixed. Do not shrink to the nine digit labels.
Outside MAX per protected context is exact for argmax while these rows and
residuals stay fixed. Outside LOGSUMEXP for training cases is separately
stored and used only in the CE denominator. No partition aggregate is an
argmax atom. No source write, gate, input embedding or theory label changes.

The exporter streams original unembedding rows, records the fixed outside
normalized feature Gram and full-vocabulary geometry. It pins input/model
hashes and checks all observed native labels before any learner step.
No new model inference is needed after the R6 captures. H1 labels and all
H2 contexts stay fixed, including the generated first-hop token, so exact
protected-output preservation also preserves finite-bank hop-2 theory rates.
This is not a theorem of generalization or transfer to replayed residuals.

## Objective and two fixed arms

Try every65 native singleton in capture order. Admission target is its
frozen native global winner; covered rivals are the entire fixed pool.
Require m>max(0.5,2delta), where2delta is the range of the actual omitted-tail
scores across target AND all covered rivals. Bias is counted once and
cancels from the tail. Any admitted source stores a witness gamma strictly
between max(0.5,2delta) and m. This keeps the original R4/R7 strong guard;
the zero pairwise Qwen diagnostic is not used to choose an easier rule.

- decode control: full-vocabulary CE to the100 arithmetic truths, with the
  exact fixed-outside logsumexp denominator.
- admission arm: that CE plus weight1.0 times mean
  softplus(min_source(max(0.5,2delta)-m)). No FP regularizer, source scaling,
  source mixtures, learned gates or extra loss selection.

Both use identical native initialization, float64 CPU, seed20261027, four
Torch threads, AdamW lr0.001, betas(0.9,0.999), weight_decay0, exactly200
full-batch proposals. Save final step200, not best checkpoint. Reuse PIL's
CertifiedP3 with renorm=false, eta0.9, margin floor0, no exact exclusions,
alpha_min2^-20; keep optimizer moments of full proposals after clipping.
Clip until all208 decisions remain frozen and
2(rho*max_row_drift+max_bias_drift)<=0.9*minimum_protected_margin. Outside
rows have zero drift. Record every numerical premise and its checked
argmax consequence. Existing PIC_Learn supplies the step preservation
lemma; this does not prove an optimizer improves admission.

Additionally preserve every previously admitted context's existence of a
valid singleton (do not require the same witness). Reject/backtrack or
exact-restore if this set shrinks. An empty old set is explicitly vacuous;
no redundant full-bank oracle computation is needed to check an empty set.
Track the per-context admission mask each step. No early stopping for a
negative or positive learning outcome; invariant failure stops as failed
engineering, not as a scientific gate pass.

## FP/W scope and computation

Do not renormalize decoder rows. For FP diagnostics only, use cosine/unit
directions, mean off-diagonal squared similarity. Report pool FP and its
Welch value; if pool size<=2560 its W=0 is vacuous and must not be advertised
as tight capacity evidence. Also evaluate FULL-VOCABULARY FP via the feature
Gram: G=sum_v(normalized U_v outer normalized U_v); off-diagonal sum is
||G||_F^2 minus the sum of squared row self-inner-products. Updated-pool
Gram plus frozen-outside Gram permits this without a quadratic vocabulary
matrix. Verify FP>=W within1e-10 at every accepted/restored endpoint.

Detect and record zero native rows explicitly. They remain decoder rivals
with unchanged scores. Normalize zero rows to zero as PIL does; full token
FP still uses all token-pair denominators and its numerical FP>=W check is
reported as such, not a unit-direction theorem if zero rows exist. Separately
report the active nonzero-direction FP/W with its valid unit assumptions.
No currently nonzero updated row may become zero. Save zero-row IDs/counts;
never silently drop them from decoding, the outside maximum or the loss.
Geometry movement alone is not the scientific success criterion.

## Frozen advance gates and limitations

A positive admission-learning result requires ALL invariants, >=5 percentage
points admission rise over initialization and >=2 points above decode-only
control at final step200. All208 protected full-host outputs must remain
unchanged; EVERY two-hop arm's externally graded full-host accuracy must
therefore remain its recorded baseline, with no drop. Certified admission
mass must not shrink at any accepted step. These are protected-bank
requirements, not unseen-data results. If admitted, actual next-hop replay
with that source is still a separate untested question.

A smaller gain, a tie with control, flat FP-only movement or zero motion is
a negative fixed-budget result. No extension, extra learning rate, rank
restriction, threshold, source mixture, checkpoint selection or alternate
objective follows this run. Save all data/initial/final frame hashes, step
logs, admission masks, full geometry, per-arm hop-2 rates and independent
endpoint oracle verification. Full intermediate parameter trajectories are
not stored; independent endpoint reconstruction and logged per-step budgets
must be distinguished. Fresh evaluation and actual admitted replay require
separate frozen follow-ups; neither can be claimed from this development bank.

## Verified two-hop baseline and endpoint chart gate, before learning

The independently verified two-hop baseline is now fixed: rederive and
refused-crystallize host accuracy12/27 each; generated/tool27/27 each.
Each rederive/crystallize readout commits25, with11 affirmed and14 false;
each appended arm commits27, all affirmed. Native/reference agreement54/54.
These are fixed development/protection outcomes, not an arm-selection basis.
All108 path labels, including30 wrong host labels, remain protected. Learning
cannot be advertised as repairing those wrong decisions while requiring
those same decisions to remain unchanged.

The NPZ also retains the108 path source programs, so the independently
graded PDC charts can be checked at both endpoints. Report full-host hop-2
accuracy AND both readouts' affirmed/hallucinated/abstained counts per arm.
In addition to the original admission-rise/control bars and full-host
preservation, a local advance must not reduce any arm/readout's affirmed
commitment count or increase its hallucinated commitment count relative to
this baseline. This is a final-checkpoint quality gate, not another loss,
source selection criterion or hyperparameter sweep. Failure is retained as
a negative local gate even if singleton admission moves. No post-hoc best
checkpoint is selected. The full R7/R6 replay claim still needs actual
prospective replay after any successful admission; a protected frozen bank
is not a substitute for that new intervention.

Report endpoint admitted-target-class and source-witness distributions.
A constant embedding or class-biased witness is not evidence of input-
dependent arithmetic computation merely because it passes on some bank
contexts. Do not remove such witnesses or redefine the eligible-source set
after seeing them; interpret the recorded result at its actual scope.

## Execution handoff, 2026-10-26, before the first optimizer proposal

The fixed data export and its independent NumPy audit passed: all208
native labels match; NPZ SHA256
771a77f88c9b5074165570ec75ea7787660ab59b58fbcb96e46f24fda80d8e99.
The pool is2261 rows; full vocabulary248320, width2560. Minimum native
protected margin is0.3948954771. All native rows are nonzero. Full-frame
FP0.0268514330 >= W0.0003865995; pool W0 is separately identified as vacuous.
These input facts do not change any optimizer, objective or stopping bar.

The endpoint chart-quality gate is evaluated separately for both learning
arms, against all four R6 memory arms and both readouts. Admission learning
advances only if its own chart-quality gate passes together with the fixed
rise/control/invariant requirements; the control's quality result is also
reported, never omitted from the source-admission comparison. PDC/theory
chart grading is performed by a separate fieldrun artifact consumer, not by
importing fieldrun code into PIL. A pending chart audit is not a passing gate.
