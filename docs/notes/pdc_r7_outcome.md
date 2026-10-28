# R7: negative admission-learning result (2026-10-25)

The frozen 200-proposal experiment completed both arms. Decode-only and
admission-objective arms each start and end at **0/75 admitted contexts**.
The five-point improvement and two-point control advantage gates both fail.
This is a measured negative result at the preregistered budget, not an
impossibility theorem. No run extension, sweep or checkpoint selection occurred.

All 183 protected full-host winners remain unchanged in both arms, including
all 108 R6 hop-2 observations. Thus their existing theory accuracy is unchanged;
this finite protected-bank constraint is not a generalization result. The 25
reserved contexts were not loaded. Input embeddings, native sources, gates,
backbone and outside output rows remain fixed; only the untied output frame
and effective bias can move. All 200 steps per arm have nonzero accepted
motion, but clipping is severe. Neither arm has an admitted context whose
preservation could provide a nonvacuous admission-mass test.

The independent NumPy endpoint auditor reconstructs all admission inequalities
and full-vocabulary winners using the exact fixed-outside maximum. Largest
endpoint discrepancy is 5.41e-13. It separately checks FP >= W: endpoint FP is
0.14523654 (decode) / 0.14523685 (admission), W=0.0007153043. FP is diagnostic,
not a substitute for the failed admission gate. Logged per-step drift bounds
satisfy 2 delta <= 0.9 m_protect throughout; endpoint reconstruction is independent,
while intermediate budgets are checked from logs rather than a saved full
parameter trajectory. Kernel theorem `decode_fixed_outside_max` in i-orca
checks the outside-max reduction; PIC_Learn supplies the step preservation
lemma. A partition-function aggregate is used only in cross-entropy.

Evidence: `results/pdc_r7/manifest.json`, `results.json`, both `*_steps.json`,
initial/final NPZ frames and `verification.json`; input NPZ is published by
fieldrun's `scripts/pdc_learning_export.py` at
`experiments/pdc/r7_fixed_data.npz` with its companion JSON manifest.
Preregistration: `docs/notes/pdc_r7_prereg.md`.
Eleven focused P3/admission tests pass, including rejection/restore of the
new optional candidate guard. R5's epistemic stop remains in force. Useful
R8 coverage and the final held-out criterion remain unmet.
