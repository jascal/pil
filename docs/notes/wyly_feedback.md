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
