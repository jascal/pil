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

## Honest state

| | qa2 | qa3 |
|---|---|---|
| probe (form ceiling) | 1.000 | 0.998 |
| teacher on train windows | 0.665 | 0.603 |
| served package | 0.498 | 0.429 |

After all three fixes the swap still does not fire: the in-learner cover scores ≥0.83 on the
same judge queries where the served package scores 0.51 — a **learner-vs-package parity gap**
in one of this package's emitted kinds (candidates: cmember, induction, counts-tier
confidences, pointer cells). That is the named next investigation; the estate2 form itself is
validated at ceiling and shipped in the schema and both runtimes (rosetta PR #42, sgiandubh
PR #27).
