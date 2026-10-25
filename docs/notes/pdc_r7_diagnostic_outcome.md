# R7 saved-frame diagnostic: negative (2026-10-25)

The user-requested four-arm offline diagnostic completed once. **Every arm
admits 0/75 contexts at initialization and at both saved R7 endpoints.**
All three endpoints therefore fall in the frozen negative region. No
fresh model calls, optimizer steps, prompt generation, threshold sweep or
access to final template-3 artifacts occurred. Original R7/R8 claims and
the frozen original evidence archive remain unchanged.

| Saved frame | Original | Half absolute floor | Highest source + bias | Random source + bias | Protected full-host matches |
|---|---:|---:|---:|---:|---:|
| Initial | 0/75 | 0/75 | 0/75 | 0/75 | 183/183 |
| Decode endpoint | 0/75 | 0/75 | 0/75 | 0/75 | 183/183 |
| Admission endpoint | 0/75 | 0/75 | 0/75 | 0/75 | 183/183 |

Conditional covered-host agreement among admissions is **undefined**, not
100%, in every arm. The protected full-host fraction is separately 100%:
these diagnostics hold the frames fixed, so that preservation is a checked
invariant, not evidence that an intervention improved learning.

## What the same body tables identify

At EACH endpoint, 9/75 contexts have at least one native singleton whose
strict covered-candidate argmax matches the original host target, ignoring
the stronger admission guard. None of the 75 contexts has a singleton
passing the strict centered-tail drift condition m>2delta; none has a
singleton with m>0.5 either. Thus:

- Matching singleton decisions do exist in 9 contexts, without this
  sufficient certificate. Absence of a matching singleton is not a valid
  explanation for all 75 refusals.
- Halving the absolute floor to 0.25 does not rescue any certificate. Since
  no singleton satisfies m>2delta even with the floor removed, changing
  the absolute floor alone cannot explain or fix these zero admissions.
  This is a logical consequence of the frozen table, not another threshold run.
- In 66 contexts no native singleton strictly matches the host target over
  this candidate pool. Size-2 and larger coalitions were NOT examined.
- These observations neither establish that the sufficient drift bound is
  necessary nor distinguish causal effects of the protected optimizer
  constraint. No real-model theory-soundness or useful-service claim follows.

The operational decision is to retire these four fixed procedures on
these frames as a coverage remedy. It does not retire every oracle, body
family, learning budget or reasoning-chart approach. Stop after these
numbers; no body shopping or further ablation was performed.

## Arm definitions and limitations

The original and half-floor arms each try all 25 fixed native singletons.
Highest selects the singleton with largest target partial score, using
native order on ties; it was already eligible in the original arm. A
present bias-only empty coalition is added, not an absent body. The random
arm uses one uniformly sampled singleton plus that bias body, matching the
highest arm's cardinalities; seed 20261026 and all 75 indices were saved
before scores were computed. A random permutation of all 25 singletons
would have been identical to the original arm, so is not used as a
misleading random control.

The half-floor arm changes max(0.5,2delta) to max(0.25,2delta). It does NOT
halve an already chosen gamma below its theorem premise. Any admission
would need a witness gamma strictly between that maximum and the body
margin. Delta is recomputed from each actual coalition tail, including the
fixed closure and all 1703 covered tokens. Protected outputs additionally
use the fixed outside maximum, never the loss-only outside logsumexp.

## Artifacts, independent verification and accounting correction

`results/pdc_r7_diagnostic/manifest.json` pins the input NPZ, all saved
frames, original R7 verdicts, protocol, executable and random indices before
the scoring stage. `results.json` stores every arm verdict and metric;
`body_diagnostics.json` stores all 26 body margins/tail ranges per context
and endpoint. `verification.json` independently recomputes every body using
individual U@source projections and full-score-minus-partial drifts, checks
all four arm choices and all original R7 endpoint verdicts, verifies the
input hashes and reconstructs all 183 protected outputs. All 5850 body tables pass;
maximum numeric discrepancy is 6.83e-13. Its input resolver maps the
frozen original-machine paths to caller-provided artifact locations,
so the verifier can be used from another checkout without those paths.

Each endpoint materializes 26*75=1950 unique body tables, shared by all
arms. Logical arm evaluations are 1875,1875,150,150 respectively. The
bias-only body requires no source projection. The original executable's
`body_row_dot_calls` field is a table-annotation count (26*75*1703), not
instrumented native dispatch. The independent verifier explicitly corrects
this accounting: native-source projections are 25*75*1703 row dots per
endpoint; protected pool projections are 183*1703, plus a duplicate 75*1703
training projection for tails. No native layers/positions run and no
service compute saving is claimed. The frozen executable and original
raw result are preserved; this correction is not a quality-based rerun.

Inputs are in the original GitHub evidence release (PDC v0.10). The new
small protocol/scripts/results are added directly to PIL PR #155. This
follow-up is not included in the old immutable archive and does not alter
its SHA-256. It is a development-artifact diagnostic, not fresh confirmation.
