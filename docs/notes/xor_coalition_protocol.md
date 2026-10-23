# XOR coalition generalization — prospective protocol

Written before running the new sweep. This is a local prospective protocol, not a
publicly timestamped pre-registration. Prior seed-0 results informed the design.

Question: do unrelated packing and material single-rule deletions survive changes
in seed, budget, and number of hard routing decisions? Does cumulative deletion
expose dependence even where no single rule clears the existing 0.05 drop cut?

Fixed design: existing `train_bank`, CPU, one thread, dim=32, 24 total clusters,
2048 examples, 1500 steps, batch=64, existing 75/25 split. Cross seeds 0–4,
M={8,16,32}, n_hard={8,16,24}: 45 trained banks. Seed 0 at n_hard=24,
M={8,16} is a reproduction, not independent confirmation. Other cells are new.
Task size here means number of hard clusters; total vocabulary stays fixed.

No retraining after deletion. Keep the existing definitions: material iff intact
hard-within accuracy minus ablated accuracy >0.05; packing iff a rule moves at least
two disjoint hard clusters by >1e-3 in mean within-cluster margin. Record all banks,
including poor fits. Report quality >0.65 separately; do not silently filter.

For each bank record single-rule drops, hard-row sign flips, moved hard clusters,
and per-rule top-1/top-3 shares of absolute mean margin change across hard clusters.
Concentration requires >0.5. Clean-input and code-weight diagnostics stay in the
original audit; this sweep's concentration is explicitly noisy hard-cluster only.

Rank rules once by single-deletion accuracy drop on the TRAINING split (ties by
rule index); freeze this order. Evaluate cumulative deletion of 0..M rules on the
untouched holdout, compared with 20 seeded random permutations of the same bank.
Report every curve, area under accuracy-vs-deleted-fraction, and smallest number
of deletions causing >0.05 accuracy drop (null if never reached). Random curves
are within-bank controls, not 20 independent trained replicates. Ranking is an
in-sample diagnostic, not a claim of generalization-optimal pruning.

Primary descriptive result: coalition bank counts per (n_hard,M), out of five
seeds; also fraction of rules packing, material, and flipping any hard row.
Robust across this grid means coalition in >=4/5 seeds in EVERY cell; otherwise
report the boundary instead of changing the threshold. Secondary: paired bank
AUC differences ranked minus random mean; report seed means/ranges, no p-value
from dependent permutation curves. This is a finite synthetic robustness study,
not a real-model or general superposition theorem.
