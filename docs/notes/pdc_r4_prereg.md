# PDC R4 preregistration, 2026-10-24

Freeze before either domain's oracle evaluation. Read `pic`'s forge-tax note
as the affine free-frame LP it states: all measured writes fixed, one common
decoder (U,b), full-data faithfulness, and a tested singleton coalition.
Store solver statuses; numerical failure is unknown, never infeasibility.
Check primal inequalities or normalized Farkas multipliers independently.

## A necessary correction to the goal's premise

The historical 0/40 is **zero 1-irreducibility certificates**, not zero
admissions: `results/forge_tax_certificate.txt` and the note's section 7 say
all 40 positions were reducible. The original `/tmp/tau` dump no longer
exists. The goal's proposed negative admission regime is therefore not an
established negative domain. Also, the note's explicit three-context instance
is faithful-frame feasible but singleton-*infeasible*. It is a refusal control,
not an admission control. Do not invert either result to meet an expectation.

Preserve the requested experiments by separating three outcomes: free-frame
reducibility, source admission under the margin certificate, and an empty set
of irreducibility certificates at fixed N. A negative result may retire an
expected sign but is not an impossibility theorem.

## Positive control, N=64

For n=0..63, y=n mod 2; sign=+1 for y=0 and -1 otherwise. Source 0 is
`(sign*(1+(n mod 4)/4), 0)`, source 1 is `(0,(n mod 7)/7)`. Target is y;
candidates are {0,1}. The shared native frame is U0=(1,0), U1=(-1,0), b=0.
Test every singleton with coalition size k=1. Source 0 must be admissible
with this stored frame: coalition margin at least 2, centered tail delta=0.
The free-frame oracle must also find a faithful singleton realization.
If this control fails, stop R4 and fix the implementation before R6.

## Refusal control, N=3

Exactly the note's e1+e2/e1/e2 example, labels 0,1,1. Sources at context 0
are e1,e2; other contexts split their residual into one nonzero source and
one zero source. Faithful frame U0=(2,2), U1=(0,0), b0=-3,b1=0 has margin 1.
All singleton LPs for context 0 must be infeasible with checked Farkas duals.
This establishes an actual refusal arm separately from the Qwen sample.

## Held-out regime rerun, N=80

Qwen2.5-0.5B-Instruct, CPU stored f16 bundle, no routing or KV quantization.
Manifest SHA256 `2b2752136948bca5d497a150dca854dd0eaf377bf0e0821108561f1f60be8e9f`;
weights SHA256 `13068ab3acaa03552ec1f72b81fa963b2add3ea70fddb331467d2814d2082acf`.
Use the first 80 positions exported by `--source-dump --n 80 --kcand 24`
on `fieldrun/sweeps/corpora/english.json`, SHA256
`080942e729338f67782f20776a8e674c4c8ec81e296cf2030da384533e4c1f25`.
This is a new fixed sample in the same architectural regime, not a bitwise
replication of the missing original dump. Keep all 80 positions; no margin
filter. Native dump reconstruction must report 1.00 before LP use.

Use gamma=1 for free-frame faithfulness/coalition LPs, affine bias included.
For each position, try singleton sources in decreasing write norm, stopping
on a checked feasible frame. If none feasible, all infeasible cases require
checked duals; any unknown makes the position unknown. Store the faithful
frame witness in compressed form for each reducible position. Report the
irreducible count even if it is still zero at N=80.

For source admission, use the same stored faithful witness and compute the
actual coalition margin m and tail scores on the identical candidates,
including the target. Set kappa=(max tail+min tail)/2 and delta=(max tail-min
tail)/2. A source is admitted only when a strict gamma with m>gamma>2delta
exists; store gamma=(m+2delta)/2 and the checked inequalities. This is a
sufficient guard; free-frame reducibility alone is not admission. The
admitted write is the coalition vector, so substitution error is zero; the
tail bound separately certifies agreement with the full residual.

Kernel packaging is `i-orca/examples/pdc/PDC_Admission.thy`, intended lemmas
`admitted_source_preserves_decision`, `pic.crystallization_admitted_write`,
and `pic.crystallization_complement_certified`. Cite only after a strict build.
No learned gates or vectors are allowed in these arms. Solver tolerance
1e-7; independently report the numerical residuals without calling them exact
real-arithmetic certificates. The synthetic refusal duals must be exactly
checkable when rationalized, not merely accepted by a solver status.
