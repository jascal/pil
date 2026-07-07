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