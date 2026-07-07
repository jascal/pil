# estate2: the world-state fold — qa2/qa3, and three admission lessons

## The form

`estate2` extends the estate register to a full world-state simulation: `loc[entity]` from
movement verbs; `holder[object]` and `loc[object]` from take/drop verbs (movement propagates
location to held objects; **drops freeze it**); a per-object location **history** for qa3's
"before the X" queries. Two modes = two family members (is / before), arbitration selects.

## Self-grounding, extended to verb SEMANTICS

Entities, locations, objects, movement verbs: mined as for estate. The new piece — **take vs
drop is learned by a universal EM-style flip**: start all grab-verbs as TAKE; flip each verb if
the world-state fold then reproduces more of the corpus's own inline answers; two sweeps. Both
corpora converge at self-agreement **1.0**, correctly sorting got/grabbed/took/picked-up from
dropped/discarded/left/put-down. Word-level probes on the UNSEEN test benchmarks:
**qa2 1000/1000, qa3 998/1000** (`probe_estate2.py`).

## The admission saga — three lessons, each now machinery

The tensor fold scores 0.836 standalone on qa2 judge queries, yet the served package stayed at
0.498. Three distinct mechanisms, uncovered in sequence:

1. **Tie-shadowing**: at sleep 0 the pointer (stale-copy of the previous inline answer, +0.301)
   out-raced estate2 (+0.234); thereafter estate2's marginal read exactly 0.0000 — both hit
   confidence 1.0 and the incumbent wins ties by rule order. Greedy forward selection can
   install a locally-better heuristic that permanently blocks a pointwise-better rule.
2. **Single-move local optimum**: backward elimination alone found nothing — the improvement
   requires remove-pointer AND add-estate2 simultaneously. **Swap moves** (full stepwise
   selection) added.
3. **Optimize the emitted artifact**: the in-learner cover contained an unemittable rule
   (moveloc [0]) holding its query numbers up while the SERVED package sat 0.3 lower — the
   selection loop now drops EMIT_INFO-less rules first (the optimize-what-you-ship principle,
   the same lesson as query-shaped judging, one level up).

## Honest state — the parity investigation, resolved and reframed

**There was no learner-vs-package gap.** The `WYLY_PARITY` dump (now standing instrumentation)
proved the live cover and the served package agree query-for-query (0.511 == 0.511, all 1000);
the suspected ~0.3 gap was a wrong inference from in-learner state. What the hunt actually
found, in layers:

1. **Fit-statistics tables cap identity maps**: estate2's dgate had learned identity for only
   4 of 6 values (support/det filters); unkeyed firings fell to counts and lost. Fixed: the
   identity table is now synthesized **analytically** — a key per value, per-value fired
   accuracy as calibrated confidence (C10-style).
2. **Query-miscalibrated incumbents** can hold a cover in a local optimum single swaps cannot
   escape; two escape mechanisms added — **eviction** (query fired-acc < 0.5) and
   **restart-from-champion** (rebuild greedily on a standalone winner).
3. Neither fired on qa2 — which isolates the remaining suspect precisely: **the counts tier's
   window-fit confidences at query tails** (the one arbitration participant that is not a
   removable rule). estate2's raw feature+gate reads 0.836 on judge queries, but its in-learner
   standalone cover ({rule + counts}) reads ≤0.53. The named next measurement: per-query
   arbitration traces; if counts outrank calibrated rules at answer slots, the counts tier
   needs query-calibration too.

| | qa2 | qa3 |
|---|---|---|
| probe (form ceiling, word level) | 1.000 | 0.998 |
| tensor feature on judge queries | 0.836 | — |
| served package | 0.498 | 0.429 |

qa3 carries an additional named limit: its stories run ~600 tokens — beyond the L=256 window —
so the fold sees truncated histories; the fix is an L=512 extraction + teacher pass (future).
The estate2 FORM remains validated at ceiling; the remaining distance is arbitration
calibration, not the rule.

## Postscript: the calibration goal — qa2 served at 0.778

The counts-tier measurement was total: at every one of the 1000 query tails (':'), the counts
tier fired at confidence 0.489, answered ' Where', and was **0.000 accurate** — the Q-colon
conflation living in the counts tier itself. Per-tail query calibration (fired accuracy replaces
window confidence where n ≥ 10, in cover arbitration and in the emitted k=1 confidences) let
**estate2 win admission at sleep 0 (+0.4246)**; the learner cover reached **0.809**.

The parity dump then caught two more real defects on the way to serving:
1. estate2's emission wrote member sets as **word strings** where the runtime compares token
   ids — the served fold never fired (fixed: token-id emission).
2. **The stratum plumbing had two holes** (rosetta #43, sgiandubh #28): the loader's backfill
   desynced on kinds that don't map 1:1 into idioms, and the pointer/induction consider sites
   never passed stratum — stratum-2 rules were serving at stratum 1 (a stratum-2 pointer
   answered ' Where' at conf 0.82 on 334 queries).

**Served qa2: 0.498 → 0.778** (learner 0.809; the residual is query-set noise + fall-through
differences). qa3 remains window-truncated (~600-token stories vs L=256; L=512 pipeline future).

## The calibration principle, complete

Every arbitration participant is now calibrated on the deployment distribution — admitted rules
(query-blend marginals), stratum-2 qualification (query fired-accuracy), incumbents (eviction +
champion restarts), and the counts tier (per-tail calibration) — C10's premise enforced end to
end, with the WYLY_PARITY dump as the standing guard that caught every serving-side divergence.

## Postscript 2: the L=512 pipeline — truncation lifted, and the residual moved again

The window-truncation limit is gone: qa3 re-extracted at L=512 (full ~600-token stories fit;
teacher gold 0.603 → 0.618; the VAL_REGION fit-exclusion margin is now window-aware). En route,
a general tokenization hole was found and fixed: **word-level mined member sets must map through
the tokenizer** — 'journeyed' splits into two Qwen tokens, so the mined verb never reached class
space and a quarter of movements were invisible to the fold (feature 0.790-of-fired). Multi-token
words now map to their first-token signatures (`mkset` fallback).

The honest scoreboard after both fixes: **served qa3 stays 0.429**. estate2/before proposes at
only +0.005 marginal despite a ~0.64-overall feature — the remaining gap between a feature that
substantially beats the cover and a near-zero admission marginal is the named open (the
before-mode gate/arbitration integration on qa3 queries: slot keying, identity-table reach, or
an incumbent overlap not yet traced). qa1 = 1.000 and qa2 = 0.778 stand; qa3 is the EAV
family's remaining frontier, now with three diagnosed-and-cleared layers behind it (window,
verbs, and the counts/stratum calibrations shared with qa2).