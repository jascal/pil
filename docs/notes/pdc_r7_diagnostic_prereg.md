# R7 saved-frame diagnostic protocol, frozen 2026-10-25

User-supplied revised Grok recommendation requests a four-arm diagnostic
on existing R7 development artifacts. This is an explicitly new offline
analysis, not a rerun, extension or rescue of R7 or R8. Original negative
results, gamma floor and advance gates remain unchanged. No fresh model
forward, optimizer proposal, prompt, theory label or consumed final-25
artifact may be loaded. Results cannot be confirmatory generalization.

## Frozen inputs and endpoints

Use only fieldrun's published r7_fixed_data.npz (SHA-256
ec7a855eaad8998c6eff6dac225e610b6dce641e283037399cfa93b0655ea195)
and PIL's saved decode_initial.npz, decode_final.npz and admission_final.npz.
Verify admission_initial equals decode_initial without scoring another
endpoint. Report initialization, decode endpoint and admission endpoint
separately; never pick the best endpoint. Frame and source arrays are fixed.
The 75 admission contexts are the R7 development bank. The 183 protected
residuals comprise that bank and 108 preexisting R6 forwards; they are used
only to check the unchanged full-host decoder using its exact fixed-outside
maximum. Do not load final heldout captures or choose bodies using theory.

## Four arms, exact interpretation

For each frame, the target remains the original native full-host winner.
Covered score candidates are all 1703 original pool tokens. For coalition B,
incidence is sum_{j in B} d_j U^T, partial score adds bias once, m is its
target gap against the strongest covered rival. Tail is r U^T minus that
coalition incidence, including the frozen numerical closure; 2delta is its
range across ALL covered tokens including target. The bias cancels in drift.

1. **original**: all 25 native singleton bodies; admit iff any satisfies
   m > max(0.5,2delta), exactly the original R7 oracle.
2. **half_floor**: identical 25 bodies, only absolute floor 0.5 -> 0.25;
   admit iff m > max(0.25,2delta). This does NOT divide an already realized
   gamma by two or relax the theorem's drift premise. In either arm store
   gamma=(m+max(floor,2delta))/2 so m>gamma>2delta is strict.
3. **highest_plus_bias**: one native singleton maximizing its target
   partial score (first native source on ties), plus the empty coalition
   whose score is bias. Same original guard/floor 0.5. This empty/bias body
   is a present coalition, NOT an absent body or zero finite score.
4. **random_plus_bias**: one uniformly random native singleton plus the
   same empty/bias body, same original guard. NumPy default_rng seed
   20261026 generates one source index per context before any score is
   computed; save the 75 indices. Reuse across all three frames. Its
   cardinalities and body count match arm 3 (one size-1 and one size-0 body).
   Sampling a permutation of all 25 singletons would merely reproduce arm 1;
   that is not used as a fictitious randomized control.

The phrase 'highest-scoring native source' is target-score ranking, not
post-hoc ranking by admission or arithmetic truth. Arms 3/4 cannot establish
absence of all size-2 or larger coalitions. Arm 3's chosen singleton already
appears in arm 1; only its bias body is newly eligible. Conditional on a
correct host and valid covered certificate, random admitted bodies are not
expected to have low theory accuracy. This diagnostic makes no truth claim.

## Metrics, diagnosis, computation and fixed interpretation

For each endpoint/arm report all 75 verdicts, admission numerator/rate,
covered-candidate agreement among admissions (undefined if none), per-body
margin, tail range, witness gamma, covered candidate scope, body evaluation
count and the 183-context unchanged full-host fraction. Report absolute
floor-only, strict drift-only and positive body-margin counts from the
SAME precomputed original singleton score tables. These are diagnostic
premise counts, not extra candidate arms or alternative service policies.
Report how many contexts have ANY singleton with a strict host-target
argmax even when the stronger certificate fails. This distinguishes no
matching singleton from failure of that particular sufficient condition.
It does not establish causal adequacy, necessity or a unique bottleneck.

Count both logical per-arm body evaluations and actual shared coalition
score computation. All 25 singleton plus one bias body are scored once
per context/frame (26*75*3), each with 1703 row incidences; arms reuse them.
Pool protected logits use 183*1703 row dots per frame. The fixed outside
maximum is loaded; no output rows, model layers or positions are newly
executed. Report actual script runtime descriptively, not as service savings.

Apply the supplied thresholds separately to EACH endpoint, never to a best
selected endpoint: positive if arm 2 or 3 has admission >=0.20 and unchanged
protected full-host fraction >=0.99; negative if ALL four arms have admission
<0.05; otherwise inconclusive. Exact boundary values belong to inconclusive
unless positive applies. Any failed input identity, strict inequality,
nonfinite value, unexpected original-arm result (must match saved R7), or
full-host invariant fails the audit instead of becoming a scientific result.
A positive is a local oracle sensitivity result, not a new real-model
coverage claim. A negative retires only these four procedures on these
fixed frames; it cannot show that no small coalition exists, diagnose the
protected optimizer constraint causally, or retire all chart methods.

Freeze this protocol, executable, random indices and input hashes in a run
manifest BEFORE evaluating frame scores. Independently reconstruct every
admitted witness and the original-arm endpoint verdicts from saved NPZs.
Save full diagnostics and stop after this one analysis. No further gamma
factors, body substitutions, size sweeps or new model runs are authorized
by this diagnostic. Publish the four rates even if all zero.
