# Hub proof (join battery): constructive assembly bridges the TYPED output-vocabulary wall; the lexical star wall is located, not bridged (slice #93)

**Status:** measured + registered (2026-07-12). Pre-registered in
`PIL_CFQ_JOIN_PREREG.md` ("the HUB PROOF") BEFORE numbers; the star prediction failed
honestly and is reported under the prereg's own `<0.7` branch. Implementation:
`experiments/campaign_join_battery.py` (two new arms) + `tests/test_join_battery_hub.py`
(3 tests). Verified: ruff clean, 350 existing tests + 3 new pass, leak guard checked by
signature inspection.

## The reframing this slice establishes (corrects the ledger)

The consolidation note (`consolidation_76_90.md`) and steering said *"the count-aggregate
atom is the fix"* for the #77 hub/star wall. **Refuted.** Reading the code: `expand()`
(L189-195) can only union **fixed train-mined `tgt` multisets**, and the prediction target
IS the serialized predicate-combination signature string (`canon_sig`). So SIG's 0.000 on
held-out stars is an **OUTPUT-VOCABULARY wall (the atom can fire, it cannot *emit* a
never-seen signature string), not a firing wall.** A count aggregate emits a scalar — zero
predicate identity — so it is the wrong output type for this battery; the sudoku
count-aggregate register and this relational battery test **different capabilities**.

## Two arms (both fire hub-shaped; only one constructs)

- **CONSTRUCT** — ONE atom that fires by compatibility on any key and builds its target **at
  fire time** from TRAIN-mined slot tables (`mine_slot_tables` → `assemble_joins_by_type`,
  reused from `campaign_cfq_typed_join.py`). Earns admission on IID val (constructs those
  signatures correctly) and the same rule generalizes.
- **CAGG** — designed-negative: fires hub-shaped on a scalar peer-set feature (`len(key)`)
  but carries a FIXED train-mined majority `tgt`. Isolates the cause.

## Results

**Regime S** (typed regime; `test_comp` = chain/cycle only — regime S structurally keeps the
rare valid star keys in train via force-cover, at every seed, so it has NO held-out stars):

| arm | rules (n_adm) | test_iid | test_comp |
|---|---|---|---|
| CONSTRUCT | **1** | 1.00 | **1.00** |
| SIG | 66 | 1.00 | 0.95 |
| CAGG | 1 | 0.05 | **0.02** |

→ Constructive fire-time assembly **bridges the typed compositional output-vocabulary wall
with a single general rule**, where 66 memorized-signature atoms reach 0.95 and a
scalar-keyed fixed-vocabulary atom reaches 0.02. The count-aggregate framing is refuted.

**Regime L** (the ACTUAL star wall — `_build_l_star` coin-flips ~492 held-out all-variable
stars into `test_comp`; their full-triple signatures never occur in train, `bound=0.0`;
reproduces #77):

| arm | star comp | chain comp | test_comp | n_adm |
|---|---|---|---|---|
| SIGW (ceiling) | 0.00 | 0.844 | 0.4286 | 28 |
| SIG | 0.00 | 0.656 | 0.333 | 21 |
| **CONSTRUCT** | **0.00** | **0.00** | **0.00** | **0** |
| CAGG | 0.00 | 0.00 | 0.00 | 1 |

→ **Every arm scores 0.0 on held-out stars, including CONSTRUCT.** CONSTRUCT is
`n_admitted=0`: the **typed** constructor is structurally **inert** on lexical all-variable
stars (no explicit types to group by → all singletons → no signature → no atom → never
admitted). The star wall stands.

## What bridging the star wall requires (the open direction)

The regime-L star is **lexical / all-variable**: the center `?x0` is the subject of all
three predicates, and the whole predicate triple is held out. Bridging it needs a
**LEXICAL / STRUCTURAL constructor** — a rule that learns the join *shape* ("an all-var hub
joins any predicates sharing the center subject") **independent of the specific predicates or
types**, and constructs the held-out star signature from that structure. That is the
**structural / relational-shape generalization** dimension (generalize the shape, not the
tokens). This slice locates it as the precise, measured requirement; it is a **new
pre-registered slice** reopening the star wall on a structural basis (permitted by steering's
"don't reopen #77 without a hub-shaped trunk task").

## Tags

| Claim | Tag |
|---|---|
| SIG's held-out-star 0.000 is an OUTPUT-VOCABULARY wall, not a firing wall (`expand` unions fixed train-mined tgt; target is the signature string) | **proved-by-construction** (verified in code) + **empirical** (CAGG hub-fires yet scores 0.02) |
| constructive fire-time assembly bridges the TYPED compositional output-vocabulary wall (regime S: comp 1.00 with 1 rule vs SIG 66 / CAGG 0.02) | **empirical** (this battery); mechanism **proved-by-construction** (no gold path — leak-guarded, `_construct_tgt` reads only key + train tables) |
| "the count-aggregate atom is the fix for the hub wall" (prior ledger claim) | **REFUTED** (CAGG comp 0.02 — scalar cannot emit a relational signature) |
| the typed constructor bridges the regime-L lexical star wall | **NOT shown** — CONSTRUCT `n_admitted=0` in L; needs a lexical/structural constructor. **OPEN.** |

## Process note (pre-registration hygiene)

The prereg mis-operationalized "held-out stars" as regime S. Checking `test_comp`'s topology
BEFORE claiming caught the mismatch (regime S has zero held-out stars — a structural bias of
the force-cover partition, not a seed fluke); the star wall was then measured where it lives
(regime L). The registered `<0.7` branch fired: report the residual, which is the typed
constructor's inertness on lexical stars. No threshold was tuned to force the prediction.
