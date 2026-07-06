# Feedback sharpening: the RLHF analog, tested in three modes

Simulated attributable feedback (the element benchmark's own errors name their fired keys) applied
to the served package, with regression accounting on the 532 previously-correct answers:

| mode | result | lesson |
|---|---|---|
| naive key bans (22) | **+4 / −12** (net −8) | culprit keys are SHARED — banning a key that misfires for one element breaks others; attributable ≠ safely bannable |
| collateral-guarded bans (8) | +0 / −0 (inert) | nearly all culprits are load-bearing for correct answers — subtraction cannot sharpen this package |
| corrective patches at W=6 (677) | **+26 / −28** (churn) | the runtime's max key length (W=6) is too short to be prompt-specific — a patch for Gold's mass hijacks Silver's |

## The unifying lesson

**Key reach governs everything.** The same constraint that drove the element benchmark's k-ladder
(0.150 → 0.751), that made mass irrecoverable (digit drift beyond every key), and that made
window-judged admission blind to query-shaped rules — now bounds *repairability*: feedback can
only patch at key lengths the runtime scans, and those keys must be long enough to be
context-specific. The named engineering fix is structural: **tuple-keyed tiers** (immune to the
int64 packing overflow that forced SAFE_KGRAMS to clamp k for large vocabularies), giving
arbitrary key reach — after which prompt-specific corrective patching becomes the clean surgical
operation the attributable-feedback design promises. The positive claims survive in exact form:
feedback IS attributable (every wrong answer named its key), regressions ARE measurable
per-round, and the collateral guard works — the missing piece is reach, not the feedback loop.

## Postscript: tuple-keyed tiers landed — and the constraint moved

`TupleFrame` (2D-row store + compacted-id trie lookup) replaces int64 key packing: **exact at any
k and any vocabulary** (unit-proven: bit-identical to the packed tier at safe k on 300/300 fired
lookups; brute-force-exact at k=10 where packing overflows). Two skeletons fell out of the
packing era: pre-guard high-k runs had silently **wrapped** keys (functioning as an accidental
hash table), and `SAFE_KGRAMS` had silently clamped the elements package to k≤5 — the W=6
mystery solved.

But with true k=2..12 available, the benchmark moved only 0.751 → 0.753 (mass 0.017 → 0.025),
and the feedback pattern was unchanged — because **the judge admitted exactly one tier (k=6)**:
on window-shaped validation, k=6 already covers, so deeper reach shows zero marginal. The reach
constraint has migrated from the data structure into the admission objective. This is the third
independent sighting of the same lesson (bAbI movement rules, elements W, now tuple tiers):
**query-shaped judging is the single unlock** — the machinery for reach now exists end to end
and waits on a judge that scores where deployment queries actually live.