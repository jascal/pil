# CFQ role decoder — attempt 1 (honest negative)

**Status:** measured, negative (2026-07-11). `experiments/campaign_cfq_role_decode.py`
tests whether a question-conditioned per-slot role decoder beats the global
majority-role prior (from the [edge headroom diagnostic](cfq_edge_headroom.md)) in the
oracle-predicate regime. It does **not** — but for an *instrumentation* reason, so the
surface-adjacency hypothesis is untested, not refuted.

## Setup
Oracle-predicate regime (gold predicate multiset given; roles predicted from the question
only — **no gold-role leak**), role-typed multiset edge-F1, per split. Four methods:
`majority_joint`, `per_slot_marginal`, `global_feature_prior` (majority conditioned on
mention-count bucket), and the per-slot **local-surface decoder** (mention→`ENT` vs
indefinite "a/an X"→non-`ENT`, with a majority-joint fallback on any alignment ambiguity).

## Result (mcd1/2/3, overall role-typed edge-F1)

| method | mcd1 | mcd2 | mcd3 |
|---|---|---|---|
| majority_joint (no-leak) | 0.427 | 0.342 | 0.377 |
| per_slot_marginal | 0.397 | 0.322 | 0.345 |
| global_feature_prior | 0.433 | 0.345 | 0.342 |
| **decoder** | 0.427 | 0.342 | 0.377 |

Decoder `fallback_rate = 1.000` on all three splits.

## Findings
1. **The decoder ties majority exactly (fallback 100%)** — its surface-override logic
   never fired. Root cause: the anchor-location step reused `mine_edge_pred_atoms`, which
   yields a **median 75 words per predicate** (of 39), so the "exactly one anchor position"
   gate can essentially never hold on real CFQ. The surface-adjacency hypothesis is
   **untested**, blocked on anchoring — a spec flaw (wrong tool for positional anchoring),
   not a decoder bug. The never-worse fallback behaved exactly as designed.
2. **Roles are slot-correlated, not independent** — `per_slot_marginal` (0.32–0.40) is
   *worse* than the joint (0.34–0.43) on every split. Predicting subject/object roles
   independently loses their correlation.
3. **Coarse global question features don't capture the variation** —
   `global_feature_prior` (mention-count bucket) barely moves and is *negative* on mcd3.

## What this says
The bottleneck for cashing in the anchored-edge role headroom is **predicate→surface
anchoring** — aligning a predicate to its relation phrase in the question. That is the
general *grounding* problem, not a CFQ quirk. A precision anchor (top-1 distinctive word
per predicate, required to appear once) is the fix to actually test the hypothesis;
deferred in favor of general-learner threads, where grounding is better tackled than as a
CFQ one-off. The eval harness (4 no-leak baselines, oracle-predicate protocol) is reusable
for that retry.

## Tag discipline
| Claim | Tag |
|---|---|
| surface role decoder beats the majority prior | **open** (untested — 100% no-op under the coarse anchor) |
| argument roles are slot-correlated (marginal < joint) | **empirical** (mcd1/2/3) |
| coarse global features don't capture role variation | **empirical** (mcd1/2/3) |
| predicate→surface anchoring is the bottleneck | **empirical** (median 75 candidate anchor words/predicate) |
